from __future__ import annotations

import numpy as np

from state_sim.environment import AlbionStateSim

# Zone risk as continuous scalar (generalizable, no identity)
_ZONE_TYPE_RISK: dict[str, float] = {
    "city": 0.0,
    "blue": 0.2,
    "yellow": 0.4,
    "red": 0.7,
    "black": 1.0,
}


class ObsEncoder:
    def __init__(self, env: AlbionStateSim, use_world_model: bool = False):
        # Zone properties (generalizable) — NO zone_id identity
        self.zone_types = ["city", "blue", "yellow", "red", "black"]
        self.zone_type_to_idx = {z: i for i, z in enumerate(self.zone_types)}
        self.biomes = ["forest", "highland", "mountain", "steppe", "swamp"]
        self.biome_to_idx = {b: i for i, b in enumerate(self.biomes)}
        self.task_types = ["navigation", "gather", "mixed"]
        self.task_type_to_idx = {t: i for i, t in enumerate(self.task_types)}
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
        self.fsm_state_to_idx = {s: i for i, s in enumerate(self.fsm_states)}

        # Keep env reference for goal zone property lookup
        self._zone_manager = env.zone_manager

        # Base features: 28 (no memory-based directions, no world model)
        # 5+5+1+5+5+1+3+8+28 = 61
        self.obs_dim = (
            len(self.zone_types)  # current zone_type one-hot (5)
            + len(self.biomes)  # current biome one-hot (5)
            + 1  # current zone_risk continuous
            + len(self.zone_types)  # goal zone_type one-hot (5)
            + len(self.biomes)  # goal biome one-hot (5)
            + 1  # goal zone_risk continuous
            + len(self.task_types)  # task_type one-hot (3)
            + len(self.fsm_states)  # fsm_state one-hot (8)
            + 28  # base continuous features (no mem_* features)
        )

    def reset(self):
        """Nothing to reset (no world model)."""
        pass

    def encode(self, obs: dict[str, float | int | str | bool]) -> np.ndarray:

        # --- Zone properties (generalizable, NO zone_id identity) ---
        zone_type_vec = np.zeros(len(self.zone_types), dtype=np.float32)
        biome_vec = np.zeros(len(self.biomes), dtype=np.float32)
        goal_zone_type_vec = np.zeros(len(self.zone_types), dtype=np.float32)
        goal_biome_vec = np.zeros(len(self.biomes), dtype=np.float32)
        task_type_vec = np.zeros(len(self.task_types), dtype=np.float32)
        fsm_state_vec = np.zeros(len(self.fsm_states), dtype=np.float32)

        zone_type = str(obs["zone_type"])
        biome = str(obs.get("biome", "forest"))
        task_type = str(obs.get("task_type", "navigation"))
        fsm_state = str(obs.get("fsm_state", "explore"))

        # Current zone properties
        zone_type_vec[self.zone_type_to_idx.get(zone_type, 1)] = 1.0
        biome_vec[self.biome_to_idx.get(biome, 0)] = 1.0
        zone_risk = np.float32(_ZONE_TYPE_RISK.get(zone_type, 0.2))

        # Goal zone properties (look up from zone manager)
        goal_zone_id = str(obs.get("goal_zone", obs.get("zone_id", "")))
        goal_zt = self._zone_manager.zone_types.get(goal_zone_id, "blue")
        goal_bi = self._zone_manager.zone_biomes.get(goal_zone_id, "forest")
        goal_zone_type_vec[self.zone_type_to_idx.get(goal_zt, 1)] = 1.0
        goal_biome_vec[self.biome_to_idx.get(goal_bi, 0)] = 1.0
        goal_risk = np.float32(_ZONE_TYPE_RISK.get(goal_zt, 0.2))

        task_type_vec[self.task_type_to_idx.get(task_type, 0)] = 1.0
        fsm_state_vec[self.fsm_state_to_idx.get(fsm_state, 0)] = 1.0

        cont = np.array(
            [
                float(obs["x"]),
                float(obs["y"]),
                float(obs["hp"]),
                1.0 if bool(obs["mounted"]) else 0.0,
                float(obs["inventory_load"]),
                min(1.0, float(obs["resource_value"]) / 500.0),
                min(1.0, float(obs["xp_value"]) / 500.0),
                float(obs["threat_score"]),
                # View-limited direction features
                float(obs.get("nearest_resource_dist", 1.0)),
                float(obs.get("nearest_resource_dx", 0.0)),
                float(obs.get("nearest_resource_dy", 0.0)),
                float(obs.get("nearest_mob_dist", 1.0)),
                float(obs.get("nearest_gate_dist", 1.0)),
                float(obs.get("nearest_gate_dx", 0.0)),
                float(obs.get("nearest_gate_dy", 0.0)),
                float(obs.get("nearest_obstacle_dist", 1.0)),
                float(obs.get("nearest_bank_dist", 1.0)),
                float(obs.get("nearest_bank_dx", 0.0)),
                float(obs.get("nearest_bank_dy", 0.0)),
                # Counts & general
                min(1.0, float(obs.get("resource_count", 0.0)) / 12.0),
                min(1.0, float(obs.get("mob_count", 0.0)) / 8.0),
                min(1.0, float(obs.get("obstacle_count", 0.0)) / 10.0),
                min(1.0, float(obs.get("goal_distance", 10.0)) / 10.0),
                1.0 if bool(obs.get("navigation_enabled", False)) else 0.0,
                min(1.0, float(obs.get("step_count", 0.0)) / 300.0),
                # Exploration / stagnation / gather progress
                float(obs.get("exploration_ratio", 0.0)),
                min(1.0, float(obs.get("stagnation_ticks", 0.0)) / 20.0),
                float(obs.get("gather_progress", 0.0)),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                zone_type_vec,
                biome_vec,
                [zone_risk],
                goal_zone_type_vec,
                goal_biome_vec,
                [goal_risk],
                task_type_vec,
                fsm_state_vec,
                cont,
            ],
            axis=0,
        )
