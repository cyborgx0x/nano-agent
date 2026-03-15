from __future__ import annotations

import random
from statistics import mean

import numpy as np
import torch

from state_sim.environment import AlbionStateSim
from state_sim.ppo import ActorCritic, ObsEncoder


def _sample_navigation_task(env: AlbionStateSim) -> tuple[str, str]:
    zones = [z for z, t in env.zone_types.items() if t != "city"]
    start_zone = random.choice(zones)
    goal_zone = random.choice([z for z in zones if z != start_zone])
    return start_zone, goal_zone


def _configure_eval_task(
    env: AlbionStateSim,
    objective: str,
    mixed_navigation_ratio: float,
    gather_spawn_zones: list[str] | None,
) -> str:
    objective = objective.lower()

    if objective == "navigation":
        start_zone, goal_zone = _sample_navigation_task(env)
        env.set_navigation_task(
            start_zone=start_zone, goal_zone=goal_zone, enabled=True
        )
        env.set_task_type("navigation")
        return "navigation"

    if objective == "gather":
        if gather_spawn_zones:
            env.set_single_zone_training("forest_cross", enabled=False)
            env.set_gather_spawn_zones(gather_spawn_zones)
        env.set_navigation_task(
            start_zone="forest_cross", goal_zone="yellow_forest_2", enabled=False
        )
        env.set_task_type("gather")
        return "gather"

    if objective == "mixed":
        if random.random() < mixed_navigation_ratio:
            start_zone, goal_zone = _sample_navigation_task(env)
            env.set_navigation_task(
                start_zone=start_zone, goal_zone=goal_zone, enabled=True
            )
            env.set_task_type("navigation")
            return "navigation"

        env.set_navigation_task(
            start_zone="forest_cross", goal_zone="yellow_forest_2", enabled=False
        )
        env.set_task_type("gather")
        return "gather"

    raise ValueError(f"unknown objective={objective}")


def _episode_success(
    info: dict[str, float | int | bool | str], episode_mode: str
) -> float:
    if episode_mode == "navigation":
        return float(info.get("navigation_success", False))
    if episode_mode == "gather":
        return float(info.get("gather_success", 0.0))
    return float(info.get("success_rate", 0.0))


def evaluate_objective_checkpoint(
    checkpoint_path: str,
    objective: str,
    episodes: int,
    max_steps: int,
    seed: int,
    mixed_navigation_ratio: float,
    gather_eval_zones: list[str] | None = None,
    gather_round_robin: bool = True,
    inventory_full_target: int = 99,
) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = AlbionStateSim(seed=seed, max_steps=max_steps)
    env.inventory_full_gather_events = inventory_full_target
    obs_encoder = ObsEncoder(env)

    ckpt = torch.load(checkpoint_path, map_location=device)
    memory_size = int(ckpt.get("memory_size", 128))
    model = ActorCritic(
        obs_encoder.obs_dim, env.action_count, memory_size=memory_size
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    successes: list[float] = []
    returns: list[float] = []
    lengths: list[float] = []
    inventory_fulls: list[float] = []

    eval_zones = gather_eval_zones[:] if gather_eval_zones else []
    for ep in range(episodes):
        episode_mode = _configure_eval_task(
            env,
            objective,
            mixed_navigation_ratio,
            eval_zones if objective.lower() == "gather" else None,
        )
        if objective.lower() == "gather" and eval_zones and gather_round_robin:
            env.force_next_gather_spawn_zone(eval_zones[ep % len(eval_zones)])
        obs = env.reset()

        done = False
        ep_return = 0.0
        ep_len = 0
        hidden = model.init_hidden(1, device)

        while not done:
            obs_vec = obs_encoder.encode(obs)
            obs_t = torch.from_numpy(obs_vec).float().to(device).unsqueeze(0)

            with torch.no_grad():
                logits, _, hidden = model(obs_t, hidden)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        successes.append(_episode_success(info, episode_mode))
        returns.append(ep_return)
        lengths.append(float(ep_len))
        inventory_fulls.append(float(info.get("inventory_full", 0.0)))

    env.close()

    return {
        "eval_success": mean(successes) if successes else 0.0,
        "eval_return": mean(returns) if returns else 0.0,
        "eval_length": mean(lengths) if lengths else 0.0,
        "eval_inventory_full_rate": mean(inventory_fulls) if inventory_fulls else 0.0,
    }
