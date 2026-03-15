"""
Spatial World Model - External memory for agent exploration.

This module provides a symbolic world model that the agent builds during runtime
as it explores the environment. Unlike the GRU hidden state, this memory:
- Is built incrementally through exploration
- Resets each episode (no cross-episode memory)
- Stores discrete, queryable information (positions, entity types)
- Helps agent make informed decisions based on what it has seen
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    """Represents a discovered entity in the world."""

    x: float
    y: float
    timestamp: int
    confidence: float = 1.0


@dataclass
class ResourceNode(Entity):
    """Resource node discovered during exploration."""

    resource_type: str = "unknown"
    tier: int = 0


@dataclass
class Gate(Entity):
    """Gate connecting zones."""

    neighbor_zone: str | None = None


class ZoneModel:
    """
    Model of a single zone built during exploration.
    """

    def __init__(self, zone_id: str, grid_size: int = 10):
        self.zone_id = zone_id
        self.grid_size = grid_size

        # Entity tracking
        self.resources: list[ResourceNode] = []
        self.gates: list[Gate] = []
        self.bank: Entity | None = None

        # Exploration tracking
        self.visited_cells: set[tuple[int, int]] = set()
        self.total_cells = grid_size * grid_size

    def add_resource(self, x: float, y: float, timestamp: int, merge_radius: float = 0.08):
        """Add or update resource node."""
        # Check if resource already known (merge nearby)
        for res in self.resources:
            if math.hypot(res.x - x, res.y - y) < merge_radius:
                # Update existing
                res.x = x
                res.y = y
                res.timestamp = timestamp
                res.confidence = min(1.0, res.confidence + 0.1)
                return

        # New resource
        self.resources.append(ResourceNode(x=x, y=y, timestamp=timestamp))

    def add_gate(self, x: float, y: float, timestamp: int, merge_radius: float = 0.08):
        """Add or update gate."""
        for gate in self.gates:
            if math.hypot(gate.x - x, gate.y - y) < merge_radius:
                gate.x = x
                gate.y = y
                gate.timestamp = timestamp
                gate.confidence = min(1.0, gate.confidence + 0.1)
                return

        self.gates.append(Gate(x=x, y=y, timestamp=timestamp))

    def add_bank(self, x: float, y: float, timestamp: int):
        """Set bank position (only one per zone)."""
        self.bank = Entity(x=x, y=y, timestamp=timestamp)

    def mark_visited(self, x: float, y: float):
        """Mark a cell as visited for exploration tracking."""
        cell_x = min(self.grid_size - 1, max(0, int(x * self.grid_size)))
        cell_y = min(self.grid_size - 1, max(0, int(y * self.grid_size)))
        self.visited_cells.add((cell_x, cell_y))

    @property
    def explored_ratio(self) -> float:
        """Fraction of zone explored."""
        return len(self.visited_cells) / max(1, self.total_cells)

    def get_nearest_resource(
        self, agent_pos: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Return position of nearest known resource."""
        if not self.resources:
            return None

        ax, ay = agent_pos
        nearest = min(
            self.resources, key=lambda r: math.hypot(r.x - ax, r.y - ay)
        )
        return (nearest.x, nearest.y)

    def get_nearest_gate(
        self, agent_pos: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Return position of nearest known gate."""
        if not self.gates:
            return None

        ax, ay = agent_pos
        nearest = min(self.gates, key=lambda g: math.hypot(g.x - ax, g.y - ay))
        return (nearest.x, nearest.y)

    def find_unexplored_area(
        self, agent_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Find nearest unexplored cell for exploration.
        Returns center of cell in normalized coordinates.
        """
        ax, ay = agent_pos
        agent_cell = (
            min(self.grid_size - 1, max(0, int(ax * self.grid_size))),
            min(self.grid_size - 1, max(0, int(ay * self.grid_size))),
        )

        # Find closest unvisited cell
        min_dist = float("inf")
        target_cell = agent_cell

        for cx in range(self.grid_size):
            for cy in range(self.grid_size):
                if (cx, cy) not in self.visited_cells:
                    dist = math.hypot(cx - agent_cell[0], cy - agent_cell[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_cell = (cx, cy)

        # Convert cell to normalized coordinates (center of cell)
        tx = (target_cell[0] + 0.5) / self.grid_size
        ty = (target_cell[1] + 0.5) / self.grid_size
        return (tx, ty)

    def prune_stale_entities(self, current_tick: int, max_age: int = 300):
        """Remove entities not seen recently."""
        self.resources = [
            r for r in self.resources if current_tick - r.timestamp <= max_age
        ]
        self.gates = [g for g in self.gates if current_tick - g.timestamp <= max_age]


class SpatialWorldModel:
    """
    Global world model built during episode through exploration.
    Resets each episode - agent must rebuild map each time.
    """

    def __init__(self, exploration_grid_size: int = 10):
        self.zones: dict[str, ZoneModel] = {}
        self.current_zone: str | None = None
        self.grid_size = exploration_grid_size
        self.current_tick = 0

    def reset(self):
        """Clear all spatial memory. Called at episode start."""
        self.zones.clear()
        self.current_zone = None
        self.current_tick = 0

    def update_from_observation(
        self, obs: dict[str, float | int | str | bool], vision_radius: float = 0.30
    ):
        """
        Update world model based on current observation.
        Called every step during episode.

        Args:
            obs: observation dict from environment
            vision_radius: how far agent can see
        """
        zone_id = str(obs["zone_id"])
        agent_pos = (float(obs["x"]), float(obs["y"]))
        self.current_zone = zone_id
        self.current_tick = int(obs.get("step_count", 0))

        # Ensure zone exists
        if zone_id not in self.zones:
            self.zones[zone_id] = ZoneModel(zone_id, self.grid_size)

        zone = self.zones[zone_id]

        # Mark current position as visited
        zone.mark_visited(agent_pos[0], agent_pos[1])

        # Update entities from vision
        res_dist = float(obs.get("nearest_resource_dist", 1.0))
        if res_dist <= vision_radius:
            res_dx = float(obs.get("nearest_resource_dx", 0.0))
            res_dy = float(obs.get("nearest_resource_dy", 0.0))
            res_x = agent_pos[0] + res_dx * res_dist
            res_y = agent_pos[1] + res_dy * res_dist
            zone.add_resource(res_x, res_y, self.current_tick)

        gate_dist = float(obs.get("nearest_gate_dist", 1.0))
        if gate_dist <= vision_radius:
            gate_dx = float(obs.get("nearest_gate_dx", 0.0))
            gate_dy = float(obs.get("nearest_gate_dy", 0.0))
            gate_x = agent_pos[0] + gate_dx * gate_dist
            gate_y = agent_pos[1] + gate_dy * gate_dist
            zone.add_gate(gate_x, gate_y, self.current_tick)

        bank_dist = float(obs.get("nearest_bank_dist", 1.0))
        zone_type = str(obs.get("zone_type", ""))
        if zone_type == "city" and bank_dist <= vision_radius:
            bank_dx = float(obs.get("nearest_bank_dx", 0.0))
            bank_dy = float(obs.get("nearest_bank_dy", 0.0))
            bank_x = agent_pos[0] + bank_dx * bank_dist
            bank_y = agent_pos[1] + bank_dy * bank_dist
            zone.add_bank(bank_x, bank_y, self.current_tick)

        # Prune stale entities
        zone.prune_stale_entities(self.current_tick, max_age=300)

    def get_features(
        self, obs: dict[str, float | int | str | bool]
    ) -> dict[str, float]:
        """
        Extract features from world model for agent observation.

        Returns:
            Dict of world model features to add to observation
        """
        zone_id = str(obs["zone_id"])
        agent_pos = (float(obs["x"]), float(obs["y"]))

        features = {}

        if zone_id not in self.zones:
            # Unknown zone - return defaults
            features.update(
                {
                    "wm_explored_ratio": 0.0,
                    "wm_known_resources": 0.0,
                    "wm_known_gates": 0.0,
                    "wm_exploration_target_dx": 0.0,
                    "wm_exploration_target_dy": 0.0,
                    "wm_exploration_target_dist": 1.0,
                    "wm_nearest_known_resource_dx": 0.0,
                    "wm_nearest_known_resource_dy": 0.0,
                    "wm_nearest_known_resource_dist": 1.0,
                }
            )
            return features

        zone = self.zones[zone_id]

        # Exploration state
        features["wm_explored_ratio"] = zone.explored_ratio
        features["wm_known_resources"] = min(1.0, len(zone.resources) / 12.0)
        features["wm_known_gates"] = min(1.0, len(zone.gates) / 6.0)

        # Direction to exploration target
        explore_target = zone.find_unexplored_area(agent_pos)
        dx = explore_target[0] - agent_pos[0]
        dy = explore_target[1] - agent_pos[1]
        dist = math.hypot(dx, dy)
        features["wm_exploration_target_dx"] = dx / dist if dist > 0 else 0.0
        features["wm_exploration_target_dy"] = dy / dist if dist > 0 else 0.0
        features["wm_exploration_target_dist"] = min(1.0, dist)

        # Direction to nearest known resource (from world model, not vision)
        nearest_res = zone.get_nearest_resource(agent_pos)
        if nearest_res:
            dx = nearest_res[0] - agent_pos[0]
            dy = nearest_res[1] - agent_pos[1]
            dist = math.hypot(dx, dy)
            features["wm_nearest_known_resource_dx"] = dx / dist if dist > 0 else 0.0
            features["wm_nearest_known_resource_dy"] = dy / dist if dist > 0 else 0.0
            features["wm_nearest_known_resource_dist"] = min(1.0, dist)
        else:
            features["wm_nearest_known_resource_dx"] = 0.0
            features["wm_nearest_known_resource_dy"] = 0.0
            features["wm_nearest_known_resource_dist"] = 1.0

        return features

    def query_nearest_resource(
        self, agent_pos: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Query nearest known resource in current zone."""
        if self.current_zone is None or self.current_zone not in self.zones:
            return None
        return self.zones[self.current_zone].get_nearest_resource(agent_pos)

    def query_exploration_target(
        self, agent_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Get next exploration target in current zone."""
        if self.current_zone is None or self.current_zone not in self.zones:
            # Default to random nearby
            return (0.5, 0.5)
        return self.zones[self.current_zone].find_unexplored_area(agent_pos)
