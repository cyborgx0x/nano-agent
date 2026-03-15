"""
Vectorized environment wrapper for parallel rollout collection.

Runs N AlbionStateSim instances in the same process, steps them all at once,
returns batched numpy arrays instead of single-step dicts.
This lets the model forward pass run on [N, obs_dim] batches → much better GPU
utilisation compared to N sequential [1, obs_dim] calls.
"""
from __future__ import annotations

import numpy as np

from state_sim.environment import AlbionStateSim
from state_sim.ppo.encoder import ObsEncoder


class VecEnv:
    """
    Synchronous vectorized wrapper around N AlbionStateSim environments.

    All envs are configured identically (same task/zone/phase) but seeded
    differently so they explore different trajectories.

    Usage:
        vec = VecEnv(num_envs=8, base_seed=42, max_steps=600,
                     inventory_full_target=99)
        vec.configure_gather("forest_cross", zone_ids)
        obs_batch = vec.reset()           # [N, obs_dim]
        obs_batch, rews, dones, infos = vec.step(actions)  # actions: [N]
    """

    def __init__(
        self,
        num_envs: int,
        base_seed: int,
        max_steps: int,
        inventory_full_target: int,
    ):
        self.num_envs = num_envs
        self.envs: list[AlbionStateSim] = [
            AlbionStateSim(seed=base_seed + i, max_steps=max_steps)
            for i in range(num_envs)
        ]
        for env in self.envs:
            env.inventory_full_gather_events = inventory_full_target

        # Encoders — one per env so each has its own state
        self.encoders: list[ObsEncoder] = [ObsEncoder(env) for env in self.envs]

        self._last_obs: list[dict] = [{}] * num_envs

    # ------------------------------------------------------------------ #
    # Configuration helpers (mirror AlbionStateSim's API across all envs) #
    # ------------------------------------------------------------------ #

    def set_phase(self, phase_index: int) -> None:
        for env in self.envs:
            env.set_phase(phase_index)

    def set_task_type(self, task_type: str) -> None:
        for env in self.envs:
            env.set_task_type(task_type)

    def set_single_zone_training(
        self, zone_id: str, enabled: bool = True, resource_scale: float = 1.0
    ) -> None:
        for env in self.envs:
            env.set_single_zone_training(zone_id, enabled, resource_scale)

    def set_gather_spawn_zones(self, zone_ids: list[str]) -> None:
        for env in self.envs:
            env.set_gather_spawn_zones(zone_ids)

    def set_navigation_task(
        self, start_zone: str, goal_zone: str, enabled: bool = True
    ) -> None:
        for env in self.envs:
            env.set_navigation_task(start_zone, goal_zone, enabled)

    def force_gather_zones(self, zone_cycle: list[str], offset: int) -> None:
        """Assign each env a zone from the cycle (round-robin)."""
        for i, env in enumerate(self.envs):
            zone = zone_cycle[(offset + i) % len(zone_cycle)]
            env.force_next_gather_spawn_zone(zone)

    # ------------------------------------------------------------------ #
    # Core VecEnv interface                                               #
    # ------------------------------------------------------------------ #

    @property
    def obs_dim(self) -> int:
        return self.encoders[0].obs_dim

    @property
    def action_count(self) -> int:
        return self.envs[0].action_count

    def reset(self) -> np.ndarray:
        """Reset all envs, return obs batch [N, obs_dim]."""
        obs_list = []
        for env, enc in zip(self.envs, self.encoders):
            enc.reset()
            obs = env.reset()
            obs_list.append(enc.encode(obs))
            self._last_obs[self.envs.index(env)] = obs
        return np.stack(obs_list, axis=0)  # [N, obs_dim]

    def reset_env(self, i: int) -> np.ndarray:
        """Reset env i only, return obs vec [obs_dim]."""
        env = self.envs[i]
        enc = self.encoders[i]
        enc.reset()
        obs = env.reset()
        self._last_obs[i] = obs
        vec = enc.encode(obs)
        return vec

    def step(
        self, actions: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Step all envs with the given actions.

        Returns:
            obs_batch   [N, obs_dim]
            rewards     [N]
            dones       [N]  (bool-valued float)
            infos       list[dict] length N
        """
        obs_vecs = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for i, (env, enc, action) in enumerate(
            zip(self.envs, self.encoders, actions)
        ):
            next_obs, reward, done, info = env.step(action)
            rewards[i] = reward
            dones[i] = float(done)
            infos[i] = info

            if done:
                # Auto-reset and encode fresh obs
                enc.reset()
                reset_obs = env.reset()
                obs_vecs.append(enc.encode(reset_obs))
                self._last_obs[i] = reset_obs
            else:
                obs_vecs.append(enc.encode(next_obs))
                self._last_obs[i] = next_obs

        obs_batch = np.stack(obs_vecs, axis=0)  # [N, obs_dim]
        return obs_batch, rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
