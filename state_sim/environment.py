from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict

import numpy as np

from .curriculum import default_curriculum
from .game import Action, EntityManager, Renderer, SimState, ZoneManager


class AlbionStateSim:
    def __init__(
        self, seed: int | None = None, max_steps: int = 220, grid_size: int = 5
    ):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.move_action_count = self.grid_size * self.grid_size
        self.action_count = self.move_action_count + len(Action)

        self.curriculum = default_curriculum()
        self.phase_index = 0
        self.phase = self.curriculum[self.phase_index]

        # Initialize managers
        self.zone_manager = ZoneManager(self.rng)
        self.entity_manager = EntityManager(self.rng)
        self.renderer = Renderer()

        self.navigation_task_enabled = False
        self.navigation_start_zone = "forest_cross"
        self.navigation_goal_zone = "yellow_forest_2"
        self.navigation_complete = False
        self.task_type = "navigation"
        self.task_types = ["navigation", "gather", "mixed"]
        self.task_type_to_id = {
            task_type: idx for idx, task_type in enumerate(self.task_types)
        }
        self.fsm_states = [
            "explore",
            "approach_gate",
            "interact_gate",
            "arrived",
            "approach_resource",
            "gather",
            "return_city",
            "bank",
        ]
        self.fsm_state_to_id = {state: idx for idx, state in enumerate(self.fsm_states)}

        self.state = self._initial_state()
        self.zone_entities: dict[str, dict[str, object]] = {}

        self._episode_deaths = 0
        self._episode_returns = 0
        self._episode_return_steps: list[int] = []
        self._episode_gather_events = 0
        self._episode_gather_value = 0.0

        self.map_l1_radius = 0.47
        self.single_zone_mode = False
        self.single_zone_id = "forest_cross"
        self.single_zone_resource_scale = 1.0
        self.gather_spawn_zones = [
            zone_id
            for zone_id, zone_type in self.zone_manager.zone_types.items()
            if zone_type != "city"
        ]
        self.gather_success_target_events = 99.0
        self.gather_channel_ticks_required = 3
        self.resource_respawn_ticks = 20
        self.inventory_full_gather_events = 99
        self._gather_target_resource_id: int | None = None
        self._gather_stay_ticks = 0
        self._forced_gather_spawn_zone: str | None = None
        self._gate_spawn_cooldown = (
            0  # ticks after zone transition before gate re-interactable
        )

        # Vision radius
        self._vision_radius = 0.30  # entities within this radius are "observed"

        # Exploration / stagnation tracking
        self._exploration_grid_size = 10
        self._visited_cells: set[tuple[int, int]] = set()
        self._prev_pos: tuple[float, float] = (0.5, 0.5)
        self._stagnation_ticks = 0
        self._stagnation_threshold = 0.025  # min movement to not be "stagnant"
        self._stagnation_penalty = 0.015
        self._exploration_bonus = 0.04
        self._no_move_penalty = 0.008

    def set_single_zone_training(
        self,
        zone_id: str,
        enabled: bool = True,
        resource_scale: float = 1.0,
    ) -> None:
        if zone_id not in self.zone_manager.zone_graph:
            raise ValueError(f"unknown zone_id: {zone_id}")
        self.single_zone_mode = enabled
        self.single_zone_id = zone_id
        self.single_zone_resource_scale = max(0.5, resource_scale)

    def set_gather_spawn_zones(self, zone_ids: list[str]) -> None:
        valid = [
            zone_id
            for zone_id in zone_ids
            if zone_id in self.zone_manager.zone_graph
            and self.zone_manager.zone_types.get(zone_id) != "city"
        ]
        if not valid:
            raise ValueError(
                "gather spawn zones must include at least one non-city zone"
            )
        self.gather_spawn_zones = valid

    def force_next_gather_spawn_zone(self, zone_id: str | None) -> None:
        if zone_id is None:
            self._forced_gather_spawn_zone = None
            return
        if zone_id not in self.zone_manager.zone_graph:
            raise ValueError(f"unknown gather spawn zone: {zone_id}")
        if self.zone_manager.zone_types.get(zone_id) == "city":
            raise ValueError(f"gather spawn zone must be non-city: {zone_id}")
        self._forced_gather_spawn_zone = zone_id

    def set_phase(self, phase_index: int) -> None:
        if phase_index < 0 or phase_index >= len(self.curriculum):
            raise ValueError(f"phase_index must be in [0, {len(self.curriculum) - 1}]")
        self.phase_index = phase_index
        self.phase = self.curriculum[phase_index]

    def set_navigation_task(
        self, start_zone: str, goal_zone: str, enabled: bool = True
    ) -> None:
        if start_zone not in self.zone_manager.zone_graph:
            raise ValueError(f"unknown start zone: {start_zone}")
        if goal_zone not in self.zone_manager.zone_graph:
            raise ValueError(f"unknown goal zone: {goal_zone}")
        self.navigation_start_zone = start_zone
        self.navigation_goal_zone = goal_zone
        self.navigation_task_enabled = enabled
        if enabled:
            self.task_type = "navigation"

    def set_task_type(self, task_type: str) -> None:
        if task_type not in self.task_type_to_id:
            raise ValueError(
                f"unknown task_type: {task_type}. valid={list(self.task_type_to_id.keys())}"
            )
        self.task_type = task_type

    def _infer_fsm_state(self) -> str:
        if self.navigation_task_enabled or self.task_type == "navigation":
            nearest_gate = self._nearest_dist(
                self.zone_entities.get(self.state.zone_id, {}).get("gates", [])
            )
            if self.state.zone_id == self.navigation_goal_zone:
                return "arrived"
            if nearest_gate < 0.15:
                return "interact_gate"
            return "approach_gate"

        nearest_resource = self._nearest_dist(self._resource_positions())
        if self.state.zone_type == "city" and self.state.inventory_load > 0.05:
            if self._distance_to_bank() <= 0.16:
                return "bank"
            return "return_city"
        if self.state.inventory_load > 0.78:
            return "return_city"
        if nearest_resource < 0.16:
            return "gather"
        if nearest_resource < 0.35:
            return "approach_resource"
        return "explore"

    def _initial_state(self) -> SimState:
        return SimState(
            zone_id="lymhurst",
            zone_type="city",
            biome="forest",
            x=0.5,
            y=0.5,
            hp=1.0,
            mounted=True,
            inventory_load=0.0,
            resource_value=0.0,
            xp_value=0.0,
            threat_score=0.0,
            at_city=True,
            alive=True,
            step_count=0,
            banked_value=0.0,
            last_return_steps=0,
        )

    def reset(self) -> dict[str, float | int | str | bool]:
        self.state = self._initial_state()
        self._episode_deaths = 0
        self._episode_returns = 0
        self._episode_return_steps = []
        self._episode_gather_events = 0
        self._episode_gather_value = 0.0
        self.navigation_complete = False
        self._visited_cells = set()
        self._prev_pos = (0.5, 0.5)
        self._stagnation_ticks = 0

        if self.single_zone_mode:
            self.state.zone_id = self.single_zone_id
        elif self.navigation_task_enabled:
            self.state.zone_id = self.navigation_start_zone
        elif self.task_type == "gather":
            candidates = self.gather_spawn_zones or [
                "forest_cross",
                "green_forest_1",
                "forest_gate",
            ]
            if self._forced_gather_spawn_zone in candidates:
                self.state.zone_id = str(self._forced_gather_spawn_zone)
            else:
                self.state.zone_id = self.rng.choice(candidates)
            self._forced_gather_spawn_zone = None
        elif self.phase_index >= 2:
            self.state.zone_id = self.rng.choice(
                ["forest_cross", "green_forest_1", "yellow_forest_1"]
            )

        self.state.zone_type = self.zone_manager.zone_types[self.state.zone_id]
        self.state.biome = self.zone_manager.zone_biomes[self.state.zone_id]
        self.state.at_city = self.state.zone_type == "city"
        self._gather_target_resource_id = None
        self._gather_stay_ticks = 0
        self._gate_spawn_cooldown = 0

        self._build_zone_entities()
        self.state.x, self.state.y = self.zone_manager.spawn_at_zone_gate(
            self.state.zone_id, self.zone_entities
        )
        self._gate_spawn_cooldown = 4  # buffer ticks after spawning near gate
        return self._obs()

    def _zone_distance(self, src: str, dst: str) -> int:
        if src == dst:
            return 0
        visited = {src}
        queue = deque([(src, 0)])
        while queue:
            node, depth = queue.popleft()
            for nxt in self.zone_manager.zone_graph.get(node, []):
                if nxt == dst:
                    return depth + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        return 999

    def _build_zone_entities(self) -> None:
        self.zone_entities = {}
        for zone_id, neighbors in self.zone_manager.zone_graph.items():
            gate_slots = self.zone_manager.rotated_square_gate_slots()
            gate_positions: list[tuple[float, float]] = [
                gate_slots[i % len(gate_slots)] for i in range(max(1, len(neighbors)))
            ]

            zone_type = self.zone_manager.zone_types[zone_id]
            biome = self.zone_manager.zone_biomes[zone_id]

            resources: list[dict[str, object]] = []
            mobs: list[tuple[float, float]] = []
            obstacles: list[dict[str, object]] = []
            bank_pos = (0.5, 0.56)

            if zone_type != "city":
                resource_scale = (
                    self.single_zone_resource_scale
                    if self.single_zone_mode and zone_id == self.single_zone_id
                    else 1.0
                )
                res_count = max(8, int(self.rng.randint(11, 18) * resource_scale))
                mob_count = max(1, int(self.rng.randint(2, 5) * resource_scale))
                obstacle_count = max(3, int(self.rng.randint(4, 7) * resource_scale))

                for _ in range(res_count):
                    resources.append(
                        self.entity_manager.spawn_resource_node(
                            zone_type,
                            biome,
                            self.zone_manager.biome_resource_profile,
                            self.zone_manager.sample_point_in_diamond,
                        )
                    )

                for _ in range(mob_count):
                    mobs.append(self.zone_manager.sample_point_in_diamond(margin=0.07))

                resource_points = [res["pos"] for res in resources]
                for _ in range(obstacle_count):
                    obstacle = self.entity_manager.spawn_obstacle_with_clearance(
                        biome,
                        resource_points,
                        mobs,
                        gate_positions,
                        obstacles,
                        self.zone_manager.sample_point_in_diamond,
                    )
                    if obstacle is not None:
                        obstacles.append(obstacle)

            self.zone_entities[zone_id] = {
                "gates": gate_positions,
                "resources": resources,
                "mobs": mobs,
                "obstacles": obstacles,
                "bank": bank_pos,
                "resource_respawns": [],
            }

    def _process_resource_respawns(self) -> None:
        for zone_id, entities in self.zone_entities.items():
            respawns = entities.get("resource_respawns", [])
            if not isinstance(respawns, list):
                continue
            if not respawns:
                continue

            due_respawns = [t for t in respawns if int(t) <= self.state.step_count]
            if not due_respawns:
                continue

            entities["resource_respawns"] = [
                t for t in respawns if int(t) > self.state.step_count
            ]

            resources = entities.get("resources", [])
            if not isinstance(resources, list):
                continue
            zone_type = self.zone_manager.zone_types.get(zone_id, "blue")
            biome = self.zone_manager.zone_biomes.get(zone_id, "forest")
            for _ in due_respawns:
                resources.append(
                    self.entity_manager.spawn_resource_node(
                        zone_type,
                        biome,
                        self.zone_manager.biome_resource_profile,
                        self.zone_manager.sample_point_in_diamond,
                    )
                )

    def _nearest_dist(self, points: list[tuple[float, float]]) -> float:
        if not points:
            return 1.0
        return min(
            math.hypot(px - self.state.x, py - self.state.y) for px, py in points
        )

    def _nearest_direction(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float, float]:
        """Return (dist, dx, dy) to nearest point. dx/dy are normalised direction."""
        if not points:
            return 1.0, 0.0, 0.0
        best_d = 999.0
        best_dx, best_dy = 0.0, 0.0
        for px, py in points:
            dx = px - self.state.x
            dy = py - self.state.y
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                if d > 1e-6:
                    best_dx = dx / d
                    best_dy = dy / d
                else:
                    best_dx, best_dy = 0.0, 0.0
        return best_d, best_dx, best_dy

    def _resource_positions(self) -> list[tuple[float, float]]:
        resources = self.zone_entities.get(self.state.zone_id, {}).get("resources", [])
        return [res["pos"] for res in resources]  # type: ignore[index]

    def _obstacle_positions(self) -> list[tuple[float, float]]:
        obstacles = self.zone_entities.get(self.state.zone_id, {}).get("obstacles", [])
        return [obs["pos"] for obs in obstacles]  # type: ignore[index]

    # ------ Vision filtering helpers ------ #

    def _points_in_vision(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Return only points within ``_vision_radius`` of the agent."""
        r = self._vision_radius
        ax, ay = self.state.x, self.state.y
        return [(px, py) for px, py in points if math.hypot(px - ax, py - ay) <= r]

    def _obs(self) -> dict[str, float | int | str | bool]:
        obs = asdict(self.state)
        entities = self.zone_entities.get(self.state.zone_id, {})
        gates = entities.get("gates", [])
        mobs = entities.get("mobs", [])
        resources = self._resource_positions()
        obstacles = self._obstacle_positions()

        goal_distance = self._zone_distance(
            self.state.zone_id, self.navigation_goal_zone
        )
        fsm_state = self._infer_fsm_state()

        # ----- View-limited directions (only within vision radius) ----- #
        vis_resources = self._points_in_vision(resources)
        vis_gates = self._points_in_vision(gates)
        vis_mobs = self._points_in_vision(mobs)
        vis_obstacles = self._points_in_vision(obstacles)

        res_dist, res_dx, res_dy = self._nearest_direction(vis_resources)
        gate_dist, gate_dx, gate_dy = self._nearest_direction(vis_gates)

        bank = entities.get("bank", (0.5, 0.56))
        bank_pos = [(float(bank[0]), float(bank[1]))]
        vis_bank = self._points_in_vision(bank_pos)
        bank_dist, bank_dx, bank_dy = self._nearest_direction(vis_bank)

        obs.update(
            {
                # View-limited: only entities within vision radius
                "nearest_resource_dist": res_dist,
                "nearest_resource_dx": res_dx,
                "nearest_resource_dy": res_dy,
                "nearest_mob_dist": self._nearest_dist(vis_mobs),
                "nearest_gate_dist": gate_dist,
                "nearest_gate_dx": gate_dx,
                "nearest_gate_dy": gate_dy,
                "nearest_obstacle_dist": self._nearest_dist(vis_obstacles),
                "nearest_bank_dist": bank_dist,
                "nearest_bank_dx": bank_dx,
                "nearest_bank_dy": bank_dy,
                # Zone-level counts (general zone knowledge)
                "resource_count": float(len(resources)),
                "mob_count": float(len(mobs)),
                "obstacle_count": float(len(obstacles)),
                "goal_zone": self.navigation_goal_zone,
                "goal_distance": float(goal_distance),
                "navigation_enabled": self.navigation_task_enabled,
                "task_type": self.task_type,
                "task_type_id": float(self.task_type_to_id[self.task_type]),
                "fsm_state": fsm_state,
                "fsm_state_id": float(self.fsm_state_to_id[fsm_state]),
                "exploration_ratio": len(self._visited_cells)
                / max(1, self._exploration_grid_size**2),
                "stagnation_ticks": float(self._stagnation_ticks),
                "gather_progress": min(
                    1.0,
                    float(self._episode_gather_events)
                    / max(1.0, float(self.inventory_full_gather_events)),
                ),
            }
        )
        return obs

    def _collides_obstacle(self, x: float, y: float) -> bool:
        obstacles = self.zone_entities.get(self.state.zone_id, {}).get("obstacles", [])
        for obs in obstacles:  # type: ignore[assignment]
            ox, oy = obs["pos"]
            radius = float(obs["radius"])
            if math.hypot(x - ox, y - oy) <= radius + 0.018:
                return True
        return False

    def _move_to_click_cell(self, action: int) -> None:
        row = action // self.grid_size
        col = action % self.grid_size
        target_x = (col + 0.5) / self.grid_size
        target_y = (row + 0.5) / self.grid_size

        speed = 0.28 if self.state.mounted else 0.16
        next_x = min(1.0, max(0.0, self.state.x + (target_x - self.state.x) * speed))
        next_y = min(1.0, max(0.0, self.state.y + (target_y - self.state.y) * speed))

        next_x, next_y = self.zone_manager.project_to_diamond(next_x, next_y)

        if self._collides_obstacle(next_x, next_y):
            slide_x = min(
                1.0, max(0.0, self.state.x + (target_x - self.state.x) * speed * 0.40)
            )
            slide_y = self.state.y
            slide_x, slide_y = self.zone_manager.project_to_diamond(slide_x, slide_y)
            if not self._collides_obstacle(slide_x, slide_y):
                self.state.x, self.state.y = slide_x, slide_y
            else:
                self.state.threat_score = min(1.0, self.state.threat_score + 0.03)
        else:
            self.state.x, self.state.y = next_x, next_y

        self.state.at_city = self.state.zone_type == "city"

    def _distance_to_bank(self) -> float:
        entities = self.zone_entities.get(self.state.zone_id, {})
        bank = entities.get("bank", (0.5, 0.5))
        bx, by = bank  # type: ignore[misc]
        return math.hypot(self.state.x - bx, self.state.y - by)

    def _apply_passive_danger(self) -> float:
        """Albion-accurate mob danger:
        - Mounted + moving: very small chance of a single light hit (mount absorbs).
        - Unmounted or standing still near a mob: can take real damage.
        - Only standing still near a mob for several ticks can knock out.
        """
        entities = self.zone_entities.get(self.state.zone_id, {})
        mobs = entities.get("mobs", [])
        nearest_mob = self._nearest_dist(mobs)

        damage_taken = 0.0

        # Are we close to a mob?
        mob_aggro_range = 0.18
        near_mob = nearest_mob < mob_aggro_range

        if near_mob:
            # Stagnation near mob = higher danger
            is_moving = self._stagnation_ticks < 2

            if self.state.mounted and is_moving:
                # Mounted passing through mob: one light hit, very small damage
                hit_chance = 0.12
                if self.rng.random() < hit_chance:
                    damage_taken = self.rng.uniform(0.01, 0.04)
                    self.state.hp = max(0.0, self.state.hp - damage_taken)
            elif self.state.mounted and not is_moving:
                # Mounted but standing near mob: moderate danger (dismount risk)
                hit_chance = 0.25
                if self.rng.random() < hit_chance:
                    damage_taken = self.rng.uniform(0.03, 0.10)
                    self.state.hp = max(0.0, self.state.hp - damage_taken)
                    self.state.threat_score = min(1.0, self.state.threat_score + 0.06)
            else:
                # Unmounted near mob: real danger
                hit_chance = 0.35 + self.state.threat_score * 0.15
                if self.rng.random() < hit_chance:
                    damage_taken = self.rng.uniform(0.05, 0.18)
                    self.state.hp = max(0.0, self.state.hp - damage_taken)
                    self.state.threat_score = min(1.0, self.state.threat_score + 0.10)
        else:
            # No mob nearby: threat decays
            self.state.threat_score = max(0.0, self.state.threat_score - 0.06)

        # Background zone risk (PvP, environment) — very low
        bg_risk = self.zone_manager.zone_risk(self.state.zone_type) * (
            1.0 if not self.state.mounted else 0.3
        )
        if self.rng.random() < bg_risk:
            damage_taken += self.rng.uniform(0.01, 0.05)
            self.state.hp = max(0.0, self.state.hp - damage_taken)

        if self.state.hp <= 0.0:
            self._episode_deaths += 1
            self.state.alive = False

        return damage_taken

    def _try_interact_gate(self) -> bool:
        # Gate spawn cooldown: prevent accidental re-entry after zone transition
        if self._gate_spawn_cooldown > 0:
            return False

        entities = self.zone_entities.get(self.state.zone_id, {})
        gates = entities.get("gates", [])
        neighbors = self.zone_manager.zone_graph.get(self.state.zone_id, [])
        if not gates or not neighbors:
            return False

        nearest_idx = None
        nearest_dist = 999.0
        for idx, (gx, gy) in enumerate(gates):
            dist = math.hypot(self.state.x - gx, self.state.y - gy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx is None or nearest_dist > 0.14:
            return False

        previous_zone = self.state.zone_id
        next_zone = neighbors[nearest_idx % len(neighbors)]
        self.state.zone_id = next_zone
        self.state.zone_type = self.zone_manager.zone_types[next_zone]
        self.state.biome = self.zone_manager.zone_biomes[next_zone]
        self.state.at_city = self.state.zone_type == "city"
        self._gather_target_resource_id = None
        self._gather_stay_ticks = 0
        self.state.x, self.state.y = self.zone_manager.spawn_at_zone_gate(
            next_zone, self.zone_entities, preferred_neighbor=previous_zone
        )
        self.state.threat_score = max(0.0, self.state.threat_score - 0.10)
        self._gate_spawn_cooldown = 4  # prevent re-entry for 4 ticks

        if (
            self.navigation_task_enabled
            and self.state.zone_id == self.navigation_goal_zone
        ):
            self.navigation_complete = True

        return True

    def _try_interact_resource_or_bank(self) -> tuple[float, float]:
        entities = self.zone_entities.get(self.state.zone_id, {})

        if self.state.zone_type == "city" and self._distance_to_bank() <= 0.16:
            self._gather_target_resource_id = None
            self._gather_stay_ticks = 0
            bank_gain = self.state.resource_value
            if bank_gain > 0.0:
                self.state.banked_value += bank_gain
                self.state.resource_value = 0.0
                self.state.inventory_load = 0.0
                self.state.last_return_steps = self.state.step_count
                self._episode_returns += 1
                self._episode_return_steps.append(self.state.step_count)
                return bank_gain, 0.0
            return 0.0, 0.0

        resources = entities.get("resources", [])
        if not resources:
            return 0.0, 0.0

        nearest_idx = None
        nearest_dist = 999.0
        for idx, res in enumerate(resources):
            rx, ry = res["pos"]
            dist = math.hypot(self.state.x - rx, self.state.y - ry)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx is None or nearest_dist > 0.16:
            self._gather_target_resource_id = None
            self._gather_stay_ticks = 0
            return 0.0, 0.0

        target = resources[nearest_idx]
        target_id = int(target.get("id", -1))
        if target_id < 0:
            self._gather_target_resource_id = None
            self._gather_stay_ticks = 0
            return 0.0, 0.0

        if self._gather_target_resource_id == target_id:
            self._gather_stay_ticks += 1
        else:
            self._gather_target_resource_id = target_id
            self._gather_stay_ticks = 1

        if self._gather_stay_ticks < self.gather_channel_ticks_required:
            return -1.0, 0.0  # sentinel: channeling in progress

        self._gather_stay_ticks = 0

        tier = int(target["tier"])
        enchant = int(target["enchant"])
        gather_amount = 1 if target["charges"] > 0 else 0
        if gather_amount == 0:
            return 0.0, 0.0

        value_gain = (
            self.entity_manager.resource_value_per_charge(
                str(target["resource_type"]), tier, enchant
            )
            * gather_amount
        )
        load_gain = 0.018 * (1.1 ** max(0, tier - 1))

        target["charges"] = int(target["charges"]) - gather_amount
        self.state.resource_value += value_gain
        self.state.inventory_load = min(1.0, self.state.inventory_load + load_gain)
        self.state.threat_score = min(1.0, self.state.threat_score + 0.01)
        self._episode_gather_events += gather_amount
        self._episode_gather_value += value_gain

        if target["charges"] <= 0:
            depleted_id = int(target.get("id", -1))
            resources.pop(nearest_idx)
            respawns = entities.get("resource_respawns", [])
            if isinstance(respawns, list):
                respawns.append(self.state.step_count + self.resource_respawn_ticks)
            if self._gather_target_resource_id == depleted_id:
                self._gather_target_resource_id = None
                self._gather_stay_ticks = 0

        return value_gain, 0.0

    def _try_attack_mob(self) -> tuple[float, float]:
        if self.phase_index < 6 or self.state.zone_type == "city":
            return 0.0, 0.0

        entities = self.zone_entities.get(self.state.zone_id, {})
        mobs = entities.get("mobs", [])
        if not mobs:
            return 0.0, 0.0

        nearest_idx = None
        nearest_dist = 999.0
        for idx, (mx, my) in enumerate(mobs):
            dist = math.hypot(self.state.x - mx, self.state.y - my)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx is None or nearest_dist > 0.18:
            return 0.0, 0.0

        win_rate = 0.72 - self.state.threat_score * 0.2
        if self.state.hp < 0.35:
            win_rate -= 0.2

        if self.rng.random() < win_rate:
            xp_gain = self.rng.uniform(12.0, 28.0)
            silver_drop = self.rng.uniform(3.0, 10.0)
            self.state.xp_value += xp_gain
            self.state.resource_value += silver_drop
            self.state.threat_score = min(1.0, self.state.threat_score + 0.07)

            mobs.pop(nearest_idx)
            if self.rng.random() < 0.5:
                mobs.append(
                    (self.rng.uniform(0.16, 0.84), self.rng.uniform(0.16, 0.84))
                )
            return silver_drop, xp_gain

        self.state.hp = max(0.0, self.state.hp - self.rng.uniform(0.08, 0.25))
        self.state.threat_score = min(1.0, self.state.threat_score + 0.14)
        if self.state.hp <= 0.0:
            self._episode_deaths += 1
            self.state.alive = False
        return 0.0, 0.0

    def _handle_mount_toggle(self) -> tuple[float, float]:
        self.state.mounted = not self.state.mounted
        return 0.0, 0.0

    def step(self, action: int) -> tuple[
        dict[str, float | int | str | bool],
        float,
        bool,
        dict[str, float | int | bool | str],
    ]:
        if not self.state.alive:
            return self._obs(), 0.0, True, {"reason": "agent_dead"}

        distance_before = self._zone_distance(
            self.state.zone_id, self.navigation_goal_zone
        )

        self.state.step_count += 1
        self._process_resource_respawns()
        # Tick down gate spawn cooldown
        if self._gate_spawn_cooldown > 0:
            self._gate_spawn_cooldown -= 1
        resource_gain = 0.0
        xp_gain = 0.0
        idle_penalty = 0.0
        banked_before = self.state.banked_value

        if 0 <= int(action) < self.move_action_count:
            self._move_to_click_cell(int(action))
        else:
            special_action = int(action) - self.move_action_count
            if special_action == int(Action.INTERACT):
                # In gather mode, prioritize resource interaction over gates
                # when a resource is within interaction range.
                # BUT: if HP is critically low, allow gate escape for retreat.
                resource_first = False
                if (
                    self.task_type == "gather"
                    and self.state.zone_type != "city"
                    and self.state.hp >= 0.50
                ):
                    res_dist = self._nearest_dist(self._resource_positions())
                    if res_dist <= 0.16:
                        resource_first = True

                if resource_first:
                    resource_gain, xp_gain = self._try_interact_resource_or_bank()
                    if resource_gain == -1.0:
                        # Channeling in progress — don't fall through to gate
                        resource_gain = 0.0
                    elif resource_gain == 0.0 and xp_gain == 0.0:
                        self._try_interact_gate()
                else:
                    moved_zone = self._try_interact_gate()
                    if not moved_zone:
                        rg, xg = self._try_interact_resource_or_bank()
                        resource_gain = max(0.0, rg)  # clamp sentinel
                        xp_gain = xg
            elif special_action == int(Action.ATTACK):
                resource_gain, xp_gain = self._try_attack_mob()
            elif special_action == int(Action.MOUNT_TOGGLE):
                resource_gain, xp_gain = self._handle_mount_toggle()
            else:
                idle_penalty = 1.0

        if self.state.inventory_load >= 0.95 and self.state.zone_type != "city":
            self.state.threat_score = min(1.0, self.state.threat_score + 0.05)
        if self.state.zone_type == "city":
            # City: fast heal to full
            self.state.hp = min(1.0, self.state.hp + 0.08)
            self.state.threat_score = max(0.0, self.state.threat_score - 0.10)
        else:
            # Natural HP regen — fast when out of combat (like Albion)
            if self.state.threat_score < 0.15:
                self.state.hp = min(1.0, self.state.hp + 0.025)  # fast regen
            elif self.state.threat_score < 0.35:
                self.state.hp = min(1.0, self.state.hp + 0.010)  # slow regen

        damage_taken = self._apply_passive_danger()
        died = not self.state.alive
        bank_gain = max(0.0, self.state.banked_value - banked_before)

        distance_after = self._zone_distance(
            self.state.zone_id, self.navigation_goal_zone
        )
        nav_delta = float(distance_before - distance_after)

        # --- Exploration / stagnation tracking ---
        move_dist = math.hypot(
            self.state.x - self._prev_pos[0], self.state.y - self._prev_pos[1]
        )
        exploration_reward = 0.0
        stagnation_pen = 0.0

        # Resource approach shaping: reward for moving closer to nearest resource
        resource_approach_reward = 0.0
        if (
            self.task_type == "gather"
            and self.state.zone_type != "city"
            and resource_gain == 0  # only when not already gathering
        ):
            res_positions = self._resource_positions()
            if res_positions:
                prev_nearest = min(
                    math.hypot(self._prev_pos[0] - rx, self._prev_pos[1] - ry)
                    for rx, ry in res_positions
                )
                curr_nearest = min(
                    math.hypot(self.state.x - rx, self.state.y - ry)
                    for rx, ry in res_positions
                )
                delta = prev_nearest - curr_nearest  # positive = getting closer
                if delta > 0.005:
                    resource_approach_reward = min(0.02, delta * 0.15)

        cell_x = min(
            self._exploration_grid_size - 1,
            int(self.state.x * self._exploration_grid_size),
        )
        cell_y = min(
            self._exploration_grid_size - 1,
            int(self.state.y * self._exploration_grid_size),
        )
        cell = (cell_x, cell_y)
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            exploration_reward = self._exploration_bonus

        is_channeling = self._gather_stay_ticks > 0
        if move_dist < self._stagnation_threshold and not is_channeling:
            self._stagnation_ticks += 1
            if self._stagnation_ticks > 3:
                stagnation_pen = self._stagnation_penalty * min(
                    3.0, self._stagnation_ticks / 5.0
                )
        else:
            self._stagnation_ticks = max(0, self._stagnation_ticks - 1)

        # Penalty for not moving at all
        no_move_pen = self._no_move_penalty if move_dist < 0.001 else 0.0

        self._prev_pos = (self.state.x, self.state.y)

        # ---- Gather-centric reward: 1.0 per gather event (flat, clear signal) ----
        gather_bonus = 1.0 if (self.task_type == "gather" and resource_gain > 0) else 0.0

        # ---- Step penalty: small time pressure (encourages speed) ----
        step_penalty = 0.004

        # Inventory full condition
        inventory_full = (
            self._episode_gather_events >= self.inventory_full_gather_events
        )

        if self.task_type == "gather":
            reward = (
                gather_bonus
                + resource_approach_reward
                - step_penalty
                - stagnation_pen
                - no_move_pen
            )
            if inventory_full:
                # Speed bonus: reward finishing inventory faster
                speed = float(self.inventory_full_gather_events) / max(1, self.state.step_count)
                reward += 5.0 + speed * 3.0
        else:
            # Navigation / mixed: keep phase-based formula
            reward = (
                resource_gain * self.phase.reward_resource / 100.0
                + xp_gain * self.phase.reward_xp / 100.0
                + bank_gain * self.phase.reward_bank / 100.0
                - damage_taken * self.phase.penalty_damage
                - idle_penalty * self.phase.penalty_idle
                - self.phase.step_penalty
                - (self.phase.penalty_death if died else 0.0)
                + exploration_reward
                - stagnation_pen
                - no_move_pen
                + resource_approach_reward
            )

        if self.navigation_task_enabled:
            reward += nav_delta * 0.09
            reward += 2.5 if self.navigation_complete else 0.0

        done = died or self.state.step_count >= self.max_steps
        if self.task_type == "gather" and inventory_full:
            done = True
        if self.navigation_task_enabled and self.navigation_complete:
            done = True

        info: dict[str, float | int | bool | str] = {
            "phase": self.phase.name,
            "resource_gain": resource_gain,
            "xp_gain": xp_gain,
            "bank_gain": bank_gain,
            "damage_taken": damage_taken,
            "dead": died,
            "at_city": self.state.at_city,
            "inventory_load": self.state.inventory_load,
            "banked_value": self.state.banked_value,
            "zone_id": self.state.zone_id,
            "biome": self.state.biome,
            "navigation_goal": self.navigation_goal_zone,
            "goal_distance": float(distance_after),
            "navigation_success": self.navigation_complete,
            "task_type": self.task_type,
            "fsm_state": self._infer_fsm_state(),
            "inventory_full": inventory_full,
            "exploration_ratio": len(self._visited_cells)
            / max(1, self._exploration_grid_size**2),
            "stagnation_ticks": float(self._stagnation_ticks),
        }

        if done:
            info.update(self.episode_metrics())

        return self._obs(), reward, done, info

    def episode_metrics(self) -> dict[str, float]:
        max_steps = max(1, self.state.step_count)
        success = 1.0 if self.state.banked_value > 0.0 and self.state.alive else 0.0
        gather_events = float(self._episode_gather_events)
        inventory_full = gather_events >= float(self.inventory_full_gather_events)

        # gather_success: 1.0 only when inventory is full (99 events).
        # Efficiency bonus: completing faster is better.
        gather_success = 0.0
        if inventory_full:
            # Base 1.0 for filling inventory; higher if done quickly
            efficiency = max(
                0.0, 1.0 - float(self.state.step_count) / float(self.max_steps)
            )
            gather_success = min(1.0, 0.7 + 0.3 * efficiency)
        else:
            # Partial credit proportional to progress
            gather_success = 0.6 * min(
                1.0, gather_events / max(1.0, float(self.inventory_full_gather_events))
            )

        avg_return_steps = (
            sum(self._episode_return_steps) / len(self._episode_return_steps)
            if self._episode_return_steps
            else float(max_steps)
        )
        nav_success = 1.0 if self.navigation_complete else 0.0
        exploration_ratio = len(self._visited_cells) / max(
            1, self._exploration_grid_size**2
        )
        return {
            "success_rate": success,
            "death_rate": float(self._episode_deaths > 0),
            "avg_return_steps": avg_return_steps,
            "episode_steps": float(self.state.step_count),
            "episode_banked_value": self.state.banked_value,
            "episode_gather_events": float(self._episode_gather_events),
            "episode_gather_value": self._episode_gather_value,
            "episode_xp": self.state.xp_value,
            "navigation_success": nav_success,
            "gather_success": gather_success,
            "inventory_full": float(inventory_full),
            "exploration_ratio": exploration_ratio,
        }

    def render(self, mode: str = "rgb_array", size: int = 420):
        world_size = int(size * 3)
        world = np.zeros((world_size, world_size, 3), dtype=np.uint8)
        world[:] = self.zone_manager.zone_color(self.state.zone_type)

        yy, xx = np.ogrid[:world_size, :world_size]
        nx = xx / max(1, world_size - 1)
        ny = yy / max(1, world_size - 1)
        mask = (np.abs(nx - 0.5) + np.abs(ny - 0.5)) <= self.map_l1_radius
        world[~mask] = (0, 0, 0)

        entities = self.zone_entities.get(self.state.zone_id, {})
        gates = entities.get("gates", [])
        resources = entities.get("resources", [])
        mobs = entities.get("mobs", [])
        obstacles = entities.get("obstacles", [])

        for gate in gates:
            self.renderer.draw_circle(world, gate, 12, (220, 180, 70))

        for res in resources:
            res_pos = res["pos"]
            res_color = EntityManager.resource_color(str(res["resource_type"]))
            radius = 12 if bool(res["plentiful"]) else 9
            self.renderer.draw_circle(world, res_pos, radius, res_color)

        for mob in mobs:
            self.renderer.draw_circle(world, mob, 11, (50, 50, 220))

        for obs in obstacles:
            ox, oy = obs["pos"]
            rr = int(float(obs["radius"]) * world_size)
            self.renderer.draw_circle(world, (ox, oy), max(8, rr), (95, 95, 95))

        self.renderer.draw_circle(
            world, (self.state.x, self.state.y), 13, (240, 240, 240)
        )

        if self.state.zone_type == "city":
            bank = entities.get("bank", (0.5, 0.56))
            self.renderer.draw_circle(world, bank, 14, (220, 220, 80))

        canvas = self.renderer.crop_agent_view(
            world, size, self.state.x, self.state.y
        )
        canvas = self.renderer.apply_visibility_mask(canvas)
        self.renderer.draw_circle(canvas, (0.5, 0.5), 9, (255, 255, 255))

        self.renderer.draw_label(
            canvas,
            "PLAYER",
            (int(size * 0.5) + 10, int(size * 0.5) - 10),
            (255, 255, 255),
        )

        for gate in gates:
            gx, gy = self.renderer.world_to_view_point(
                gate, size, world_size, self.state.x, self.state.y
            )
            if -20 <= gx <= size + 20 and -20 <= gy <= size + 20:
                self.renderer.draw_label(canvas, "GATE", (gx + 6, gy - 6), (220, 180, 70))

        for res in resources:
            rx, ry = self.renderer.world_to_view_point(
                res["pos"], size, world_size, self.state.x, self.state.y
            )
            if -20 <= rx <= size + 20 and -20 <= ry <= size + 20:
                label = f"{str(res['resource_type'])}:T{int(res['tier'])}"
                self.renderer.draw_label(canvas, label, (rx + 6, ry - 6), (80, 220, 80))

        for mob in mobs:
            mx, my = self.renderer.world_to_view_point(
                mob, size, world_size, self.state.x, self.state.y
            )
            if -20 <= mx <= size + 20 and -20 <= my <= size + 20:
                self.renderer.draw_label(canvas, "MOB", (mx + 6, my - 6), (50, 50, 220))

        for obs in obstacles:
            ox, oy = self.renderer.world_to_view_point(
                obs["pos"], size, world_size, self.state.x, self.state.y
            )
            if -20 <= ox <= size + 20 and -20 <= oy <= size + 20:
                kind = str(obs.get("kind", "obstacle"))
                self.renderer.draw_label(
                    canvas, f"OBS:{kind}", (ox + 6, oy - 6), (150, 150, 150)
                )

        if self.state.zone_type == "city":
            bank = entities.get("bank", (0.5, 0.56))
            bx, by = self.renderer.world_to_view_point(
                bank, size, world_size, self.state.x, self.state.y
            )
            self.renderer.draw_label(canvas, "BANK", (bx + 6, by - 6), (220, 220, 80))

        try:
            import cv2

            cv2.putText(
                canvas,
                f"zone={self.state.zone_id} biome={self.state.biome}",
                (8, size - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"hp={self.state.hp:.2f} load={self.state.inventory_load:.2f} banked={self.state.banked_value:.1f}",
                (8, size - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"goal={self.navigation_goal_zone} dist={self._zone_distance(self.state.zone_id, self.navigation_goal_zone)}",
                (8, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )

            if mode == "human":
                try:
                    cv2.imshow("AlbionStateSim", canvas)
                    cv2.waitKey(1)
                except Exception:
                    self.renderer.render_with_matplotlib(canvas)
        except Exception:
            if mode == "human":
                self.renderer.render_with_matplotlib(canvas)

        return canvas

    def close(self) -> None:
        self.renderer.close()
