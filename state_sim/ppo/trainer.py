from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from state_sim.environment import AlbionStateSim

from .algorithms import Transition, compute_gae
from .config import PPOTrainConfig
from .encoder import ObsEncoder
from .network import ActorCritic


def _split_train_holdout_zones(
    zones: list[str], holdout_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    unique = sorted(set(zones))
    if len(unique) <= 2:
        return unique, []

    rng = random.Random(seed + 17)
    shuffled = unique[:]
    rng.shuffle(shuffled)

    holdout_count = max(1, int(round(len(shuffled) * max(0.0, holdout_ratio))))
    holdout_count = min(holdout_count, len(shuffled) - 1)

    holdout = sorted(shuffled[:holdout_count])
    train = sorted(shuffled[holdout_count:])
    return train, holdout


def _evaluate_gather_holdout(
    model: ActorCritic,
    obs_dim: int,
    action_dim: int,
    episodes: int,
    max_steps: int,
    seed: int,
    eval_zones: list[str],
    memory_size: int = 128,
    inventory_full_target: int = 99,
) -> float:
    if not eval_zones or episodes <= 0:
        return 0.0

    eval_env = AlbionStateSim(seed=seed, max_steps=max_steps)
    eval_env.inventory_full_gather_events = inventory_full_target
    eval_obs_encoder = ObsEncoder(eval_env)

    eval_model = ActorCritic(obs_dim, action_dim, memory_size=memory_size)
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()

    eval_env.set_single_zone_training("forest_cross", enabled=False)
    eval_env.set_gather_spawn_zones(eval_zones)
    eval_env.set_navigation_task(
        start_zone="forest_cross", goal_zone="yellow_forest_2", enabled=False
    )
    eval_env.set_task_type("gather")

    successes: list[float] = []
    zone_cycle = eval_zones[:]
    for ep in range(episodes):
        zone_id = zone_cycle[ep % len(zone_cycle)]
        eval_env.force_next_gather_spawn_zone(zone_id)
        obs = eval_env.reset()
        # Reset world model for eval episode
        eval_obs_encoder.reset()

        done = False
        info: dict[str, float | int | bool | str] = {}
        hidden = eval_model.init_hidden(1)
        while not done:
            obs_vec = eval_obs_encoder.encode(obs)
            obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0)
            with torch.no_grad():
                logits, _, hidden = eval_model(obs_t, hidden)
                action = int(torch.argmax(logits, dim=-1).item())
            obs, _, done, info = eval_env.step(action)

        successes.append(float(info.get("gather_success", 0.0)))

    eval_env.close()
    return mean(successes) if successes else 0.0


def _target_cell(x: float, y: float, grid_size: int) -> int:
    row = min(grid_size - 1, max(0, int(y * grid_size)))
    col = min(grid_size - 1, max(0, int(x * grid_size)))
    return row * grid_size + col


def _shortest_path_first_step(graph: dict[str, list[str]], src: str, dst: str) -> str:
    if src == dst:
        return src
    queue = deque([src])
    parent: dict[str, str | None] = {src: None}

    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == dst:
                cur = nxt
                while parent[cur] is not None and parent[cur] != src:
                    cur = parent[cur]  # type: ignore[index]
                return cur
            queue.append(nxt)
    return src


def _teacher_action(
    obs: dict[str, float | int | str | bool], env: AlbionStateSim
) -> int:
    zone_id = str(obs["zone_id"])
    goal = str(obs["goal_zone"])
    nearest_gate = float(obs.get("nearest_gate_dist", 1.0))

    if zone_id == goal:
        return env.move_action_count + 3
    if nearest_gate < 0.15:
        return env.move_action_count + 0

    next_zone = _shortest_path_first_step(env.zone_manager.zone_graph, zone_id, goal)
    neighbors = env.zone_manager.zone_graph.get(zone_id, [])
    if next_zone in neighbors:
        idx = neighbors.index(next_zone)
        angle = 2 * math.pi * idx / max(1, len(neighbors)) - math.pi / 2
        tx = 0.5 + 0.40 * math.cos(angle)
        ty = 0.5 + 0.40 * math.sin(angle)
        return _target_cell(tx, ty, env.grid_size)

    x = float(obs["x"])
    y = float(obs["y"])
    return _target_cell(
        min(0.95, max(0.05, x + random.uniform(-0.2, 0.2))),
        min(0.95, max(0.05, y + random.uniform(-0.2, 0.2))),
        env.grid_size,
    )


def _teacher_action_gather(env: AlbionStateSim) -> int:
    entities = env.zone_entities.get(env.state.zone_id, {})

    # Retreat to city when HP is low
    if env.state.hp < 0.50 and env.state.zone_type != "city":
        gates = entities.get("gates", [])
        if gates:
            # Find gate leading toward a city
            neighbors = env.zone_manager.zone_graph.get(env.state.zone_id, [])
            best_gate = None
            best_dist = 999.0
            for idx, gate in enumerate(gates):
                neigh = neighbors[idx % len(neighbors)] if neighbors else None
                if neigh and env.zone_manager.zone_types.get(neigh) == "city":
                    gd = math.hypot(
                        env.state.x - float(gate[0]), env.state.y - float(gate[1])
                    )
                    if gd < best_dist:
                        best_dist = gd
                        best_gate = gate
            if best_gate is None:
                # No direct city neighbor — go to nearest gate anyway
                nearest_gate = min(
                    gates,
                    key=lambda g: math.hypot(
                        env.state.x - float(g[0]), env.state.y - float(g[1])
                    ),
                )
                best_gate = nearest_gate
                best_dist = math.hypot(
                    env.state.x - float(best_gate[0]),
                    env.state.y - float(best_gate[1]),
                )
            if best_dist <= 0.13:
                return env.move_action_count + 0  # INTERACT gate
            return _target_cell(float(best_gate[0]), float(best_gate[1]), env.grid_size)

    if env.state.zone_type == "city":
        # Heal at city if HP is low
        if env.state.hp < 0.85:
            return env.move_action_count + 3  # IDLE to heal

        # Bank if we have resources
        if env.state.resource_value > 0.0 and env._distance_to_bank() <= 0.16:
            return env.move_action_count + 0  # INTERACT to bank
        if env.state.resource_value > 0.0:
            bank = entities.get("bank", (0.5, 0.56))
            return _target_cell(float(bank[0]), float(bank[1]), env.grid_size)

        # Leave city through a gate
        gates = entities.get("gates", [])
        if not gates:
            return env.move_action_count + 3

        nearest_gate = min(
            gates,
            key=lambda g: math.hypot(
                env.state.x - float(g[0]), env.state.y - float(g[1])
            ),
        )
        gate_dist = math.hypot(
            env.state.x - float(nearest_gate[0]), env.state.y - float(nearest_gate[1])
        )
        if gate_dist <= 0.13:
            return env.move_action_count + 0
        return _target_cell(
            float(nearest_gate[0]), float(nearest_gate[1]), env.grid_size
        )

    resources = entities.get("resources", [])
    if resources:
        nearest_resource = min(
            resources,
            key=lambda r: math.hypot(
                env.state.x - float(r["pos"][0]), env.state.y - float(r["pos"][1])
            ),
        )
        rx, ry = nearest_resource["pos"]
        res_dist = math.hypot(env.state.x - float(rx), env.state.y - float(ry))
        if res_dist <= 0.15:
            return env.move_action_count + 0
        return _target_cell(float(rx), float(ry), env.grid_size)

    x = min(0.95, max(0.05, env.state.x + random.uniform(-0.2, 0.2)))
    y = min(0.95, max(0.05, env.state.y + random.uniform(-0.2, 0.2)))
    return _target_cell(x, y, env.grid_size)


def _sample_task_by_curriculum(
    env: AlbionStateSim, zones: list[str], progress: float
) -> tuple[str, str]:
    if progress < 0.35:
        candidates: list[tuple[str, str]] = []
        for z in zones:
            for n in env.zone_manager.zone_graph[z]:
                if n in zones and n != z:
                    candidates.append((z, n))
        return random.choice(candidates) if candidates else (zones[0], zones[1])

    if progress < 0.70:
        candidates = []
        for s in zones:
            for g in zones:
                if s == g:
                    continue
                if env._zone_distance(s, g) <= 2:
                    candidates.append((s, g))
        return random.choice(candidates) if candidates else (zones[0], zones[-1])

    start_zone = random.choice(zones)
    goal_zone = random.choice([z for z in zones if z != start_zone])
    return start_zone, goal_zone


def _configure_episode_task(
    env: AlbionStateSim,
    zones: list[str],
    progress: float,
    objective: str,
    mixed_navigation_ratio: float,
    gather_focus_zone: str,
    gather_focus_resource_scale: float,
    gather_train_mode: str,
    gather_zone_for_episode: str | None,
) -> str:
    objective = objective.lower()

    if objective == "navigation":
        env.set_single_zone_training(gather_focus_zone, enabled=False)
        env.set_gather_spawn_zones(zones)
        start_zone, goal_zone = _sample_task_by_curriculum(env, zones, progress)
        env.set_navigation_task(
            start_zone=start_zone, goal_zone=goal_zone, enabled=True
        )
        env.set_task_type("navigation")
        return "navigation"

    if objective == "gather":
        if gather_train_mode == "single":
            env.set_gather_spawn_zones([gather_focus_zone])
            env.set_single_zone_training(
                gather_focus_zone,
                enabled=True,
                resource_scale=gather_focus_resource_scale,
            )
        else:
            env.set_single_zone_training(gather_focus_zone, enabled=False)
            env.set_gather_spawn_zones(zones)
            if gather_zone_for_episode is not None:
                env.force_next_gather_spawn_zone(gather_zone_for_episode)
        env.set_navigation_task(
            start_zone="forest_cross", goal_zone="yellow_forest_2", enabled=False
        )
        env.set_task_type("gather")
        return "gather"

    if objective == "mixed":
        env.set_single_zone_training(gather_focus_zone, enabled=False)
        env.set_gather_spawn_zones(zones)
        if random.random() < mixed_navigation_ratio:
            start_zone, goal_zone = _sample_task_by_curriculum(env, zones, progress)
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

    raise ValueError(
        f"unknown objective={objective}. expected: navigation|gather|mixed"
    )


def _episode_success(
    info: dict[str, float | int | bool | str], episode_mode: str
) -> float:
    if episode_mode == "navigation":
        return float(info.get("navigation_success", False))
    if episode_mode == "gather":
        return float(info.get("gather_success", 0.0))
    return float(info.get("success_rate", 0.0))


def train_ppo(config: PPOTrainConfig) -> dict[str, float]:
    if config.num_envs > 1:
        return train_ppo_vec(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = AlbionStateSim(seed=config.seed, max_steps=config.max_steps)
    env.inventory_full_gather_events = config.inventory_full_target
    obs_encoder = ObsEncoder(env)
    model = ActorCritic(
        obs_encoder.obs_dim, env.action_count, memory_size=config.memory_size
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                mode=config.wandb_mode,
                config={
                    "algo": "ppo",
                    "task": f"state_sim_{config.objective}",
                    "episodes": config.episodes,
                    "max_steps": config.max_steps,
                    "lr": config.lr,
                    "gamma": config.gamma,
                    "gae_lambda": config.gae_lambda,
                    "clip_eps": config.clip_eps,
                    "update_epochs": config.update_epochs,
                    "minibatch_size": config.minibatch_size,
                    "update_every_episodes": config.update_every_episodes,
                    "objective": config.objective,
                    "mixed_navigation_ratio": config.mixed_navigation_ratio,
                },
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    zones = [z for z, t in env.zone_manager.zone_types.items() if t != "city"]
    train_zones, holdout_zones = _split_train_holdout_zones(
        zones, config.gather_holdout_ratio, config.seed
    )

    episode_returns: list[float] = []
    episode_successes: list[float] = []
    episode_lengths: list[int] = []
    buffer: list[Transition] = []

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_success = -1.0
    best_checkpoint_score = -1.0
    best_holdout_success = -1.0
    gather_teacher_boost_level = 0.0

    gather_zone_cycle = train_zones[:] if train_zones else zones[:]
    if not gather_zone_cycle:
        gather_zone_cycle = [config.gather_focus_zone]
    cycle_index = 0

    for ep in range(config.episodes):
        progress = (ep + 1) / max(1, config.episodes)
        gather_zone_for_episode: str | None = None
        if (
            config.objective.lower() == "gather"
            and config.gather_train_mode == "multi"
            and config.gather_zone_scheduler == "round_robin"
        ):
            gather_zone_for_episode = gather_zone_cycle[
                cycle_index % len(gather_zone_cycle)
            ]
            cycle_index += 1

        # Set appropriate phase for the objective
        if config.objective.lower() == "gather":
            env.set_phase(5)  # phase_5_economy_loop: reward_resource=1.2
        elif config.objective.lower() == "navigation":
            env.set_phase(2)  # phase_2_inter_zone_transition
        else:
            phase_idx = min(5, int(progress * 6))
            env.set_phase(phase_idx)

        episode_mode = _configure_episode_task(
            env,
            train_zones if config.objective.lower() == "gather" else zones,
            progress,
            config.objective,
            config.mixed_navigation_ratio,
            config.gather_focus_zone,
            config.gather_focus_resource_scale,
            config.gather_train_mode,
            gather_zone_for_episode,
        )
        if episode_mode == "navigation":
            teacher_prob = max(
                config.nav_teacher_min,
                config.nav_teacher_start * (1.0 - progress),
            )
        elif episode_mode == "gather":
            gather_decay = max(0.0, 1.0 - (progress**config.gather_teacher_decay_power))
            base_teacher = config.gather_teacher_start * gather_decay
            teacher_prob = max(
                config.gather_teacher_min,
                base_teacher,
            )
            teacher_prob = min(1.0, teacher_prob + gather_teacher_boost_level)
            gather_teacher_boost_level *= config.gather_teacher_boost_decay
        else:
            teacher_prob = max(
                config.mixed_teacher_min,
                config.mixed_teacher_start * (1.0 - progress),
            )

        obs_dict = env.reset()
        # Reset world model at episode start
        obs_encoder.reset()

        done = False
        ep_return = 0.0
        ep_steps = 0
        hidden = model.init_hidden(1)

        while not done:
            obs_vec = obs_encoder.encode(obs_dict)
            obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0)

            with torch.no_grad():
                logits, value, new_hidden = model(obs_tensor, hidden)
                dist = Categorical(logits=logits)
                sampled_action = dist.sample()

                if random.random() < teacher_prob:
                    if episode_mode == "navigation":
                        chosen_action = _teacher_action(obs_dict, env)
                    else:
                        chosen_action = _teacher_action_gather(env)
                    action = torch.tensor(
                        [chosen_action], dtype=torch.long
                    )
                    logprob = dist.log_prob(action)
                else:
                    action = sampled_action
                    logprob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(int(action.item()))
            buffer.append(
                Transition(
                    obs=obs_vec.tolist(),
                    action=int(action.item()),
                    logprob=float(logprob.item()),
                    value=float(value.item()),
                    reward=float(reward),
                    done=1.0 if done else 0.0,
                    hidden=hidden.squeeze(0).detach().cpu().tolist()
                    if hidden is not None
                    else [],
                )
            )

            hidden = new_hidden.detach() if new_hidden is not None else None
            obs_dict = next_obs
            ep_return += reward
            ep_steps += 1

        success = _episode_success(info, episode_mode)
        episode_returns.append(ep_return)
        episode_successes.append(success)
        episode_lengths.append(ep_steps)

        if (ep + 1) % config.update_every_episodes == 0 and buffer:
            obs_batch = torch.tensor(
                np.array([t.obs for t in buffer]), dtype=torch.float32
            )
            action_batch = torch.tensor(
                [t.action for t in buffer], dtype=torch.long
            )
            old_logprob_batch = torch.tensor(
                [t.logprob for t in buffer], dtype=torch.float32
            )
            value_batch = torch.tensor(
                [t.value for t in buffer], dtype=torch.float32
            )
            reward_batch = torch.tensor(
                [t.reward for t in buffer], dtype=torch.float32
            )
            done_batch = torch.tensor(
                [t.done for t in buffer], dtype=torch.float32
            )
            # Hidden batch (None for feedforward, dummy for compatibility)
            if buffer[0].hidden:
                hidden_batch = torch.tensor(
                    np.array(
                        [
                            t.hidden if t.hidden else [0.0] * config.memory_size
                            for t in buffer
                        ]
                    ),
                    dtype=torch.float32,
                )
            else:
                # Feedforward model - no hidden state needed
                hidden_batch = None

            advantages, returns = compute_gae(
                reward_batch, value_batch, done_batch, config.gamma, config.gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            idx = np.arange(len(buffer))
            for _ in range(config.update_epochs):
                np.random.shuffle(idx)
                for start in range(0, len(idx), config.minibatch_size):
                    mb_idx = idx[start : start + config.minibatch_size]
                    mb_idx_t = torch.tensor(mb_idx, dtype=torch.long)

                    mb_obs = obs_batch[mb_idx_t]
                    mb_actions = action_batch[mb_idx_t]
                    mb_old_logprobs = old_logprob_batch[mb_idx_t]
                    mb_adv = advantages[mb_idx_t]
                    mb_returns = returns[mb_idx_t]
                    mb_hidden = hidden_batch[mb_idx_t] if hidden_batch is not None else None

                    logits, values, _ = model(mb_obs, mb_hidden)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = (new_logprobs - mb_old_logprobs).exp()
                    s1 = ratio * mb_adv
                    s2 = (
                        torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                        * mb_adv
                    )
                    policy_loss = -torch.min(s1, s2).mean()
                    value_loss = nn.functional.mse_loss(values.squeeze(-1), mb_returns)
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                    optimizer.step()

            buffer.clear()

        if (ep + 1) % config.log_interval == 0:
            window = min(config.log_interval, len(episode_returns))
            success_win = mean(episode_successes[-window:])
            return_win = mean(episode_returns[-window:])
            length_win = mean(episode_lengths[-window:])

        # Run holdout eval on its own interval (not gated by log_interval)
        holdout_success = 0.0
        if (
            config.objective.lower() == "gather"
            and holdout_zones
            and config.gather_eval_episodes > 0
            and (ep + 1) % max(1, config.gather_eval_interval) == 0
        ):
            holdout_success = _evaluate_gather_holdout(
                model,
                obs_encoder.obs_dim,
                env.action_count,
                config.gather_eval_episodes,
                config.max_steps,
                config.seed + ep + 1,
                holdout_zones,
                memory_size=config.memory_size,
                inventory_full_target=config.inventory_full_target,
            )

        if (ep + 1) % config.log_interval == 0:
            if config.objective.lower() == "gather" and holdout_zones:
                checkpoint_score = success_win * 0.55 + holdout_success * 0.45
                best_holdout_success = max(best_holdout_success, holdout_success)
                if (
                    best_success > 0.0
                    and success_win
                    < best_success * config.gather_teacher_recovery_ratio
                ):
                    gather_teacher_boost_level = max(
                        gather_teacher_boost_level,
                        config.gather_teacher_recovery_boost,
                    )
                print(
                    f"episode={ep+1} success_{window}={success_win:.2f} "
                    f"holdout={holdout_success:.2f} return_{window}={return_win:.3f} len_{window}={length_win:.1f}"
                )
            else:
                checkpoint_score = success_win
                print(
                    f"episode={ep+1} success_{window}={success_win:.2f} "
                    f"return_{window}={return_win:.3f} len_{window}={length_win:.1f}"
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode": ep + 1,
                        "success_window": success_win,
                        "holdout_success": holdout_success,
                        "return_window": return_win,
                        "length_window": length_win,
                        "teacher_prob": teacher_prob,
                        "buffer_size": len(buffer),
                        "gather_events_last": float(
                            info.get("episode_gather_events", 0)
                        ),
                        "inventory_full_last": float(info.get("inventory_full", 0)),
                        "exploration_ratio_last": float(
                            info.get("exploration_ratio", 0)
                        ),
                    }
                )

            best_success = max(best_success, success_win)
            if checkpoint_score > best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                ckpt = (
                    Path(config.checkpoint_dir)
                    / f"map_to_map_ppo_{config.objective}_best.pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "obs_dim": obs_encoder.obs_dim,
                        "action_dim": env.action_count,
                        "memory_size": config.memory_size,
                        "episode": ep + 1,
                        "success_window": success_win,
                        "holdout_success": holdout_success,
                        "checkpoint_score": checkpoint_score,
                        "objective": config.objective,
                    },
                    ckpt,
                )

    final_success = mean(episode_successes) if episode_successes else 0.0
    final_return = mean(episode_returns) if episode_returns else 0.0
    tail_window = min(config.log_interval, len(episode_successes))
    tail_success_window = (
        mean(episode_successes[-tail_window:]) if tail_window > 0 else 0.0
    )
    print(
        f"final episodes={config.episodes} success={final_success:.2f} return={final_return:.3f}"
    )

    if wandb_run is not None:
        wandb_run.summary["final_success"] = final_success
        wandb_run.summary["final_return"] = final_return
        wandb_run.summary["best_success_window"] = best_success
        wandb_run.summary["best_checkpoint_score"] = best_checkpoint_score
        wandb_run.summary["best_holdout_success"] = best_holdout_success
        wandb_run.summary["tail_success_window"] = tail_success_window
        wandb_run.finish()

    env.close()

    return {
        "objective": config.objective,
        "final_success": final_success,
        "final_return": final_return,
        "best_success_window": best_success,
        "best_checkpoint_score": best_checkpoint_score,
        "best_holdout_success": best_holdout_success,
        "tail_success_window": tail_success_window,
        "checkpoint_path": str(
            Path(config.checkpoint_dir) / f"map_to_map_ppo_{config.objective}_best.pt"
        ),
    }

# ============================================================ #
#  Vectorized PPO trainer — uses N parallel envs               #
# ============================================================ #

def train_ppo_vec(config: PPOTrainConfig) -> dict[str, float]:
    """
    PPO trainer with vectorised environment collection.

    Each 'episode' (outer-loop index) = one rollout of `rollout_steps` steps
    collected from `num_envs` parallel envs simultaneously.

    Total transitions per update = num_envs * rollout_steps.
    """
    from .vecenv import VecEnv  # local import to keep module boundary clean

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ------ Reference env for zone info / action_count only ------ #
    ref_env = AlbionStateSim(seed=config.seed, max_steps=config.max_steps)
    ref_env.inventory_full_gather_events = config.inventory_full_target
    ref_enc = ObsEncoder(ref_env)
    obs_dim = ref_enc.obs_dim
    action_dim = ref_env.action_count

    zones = [z for z, t in ref_env.zone_manager.zone_types.items() if t != "city"]
    train_zones, holdout_zones = _split_train_holdout_zones(
        zones, config.gather_holdout_ratio, config.seed
    )

    # ------ Model, optimiser ------ #
    model = ActorCritic(obs_dim, action_dim, memory_size=config.memory_size)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # ------ WandB (optional) ------ #
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                mode=config.wandb_mode,
                config={
                    "algo": "ppo_vec",
                    "num_envs": config.num_envs,
                    "rollout_steps": config.rollout_steps,
                    "task": f"state_sim_{config.objective}",
                    "episodes": config.episodes,
                    "max_steps": config.max_steps,
                    "lr": config.lr,
                    "gamma": config.gamma,
                    "gae_lambda": config.gae_lambda,
                    "clip_eps": config.clip_eps,
                    "update_epochs": config.update_epochs,
                    "minibatch_size": config.minibatch_size,
                    "objective": config.objective,
                },
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    # ------ VecEnv ------ #
    spawn_zones = train_zones if train_zones else zones
    vec = VecEnv(
        num_envs=config.num_envs,
        base_seed=config.seed,
        max_steps=config.max_steps,
        inventory_full_target=config.inventory_full_target,
    )
    vec.set_phase(5 if config.objective.lower() == "gather" else 2)
    vec.set_task_type(config.objective.lower() if config.objective.lower() != "mixed" else "gather")
    vec.set_single_zone_training("forest_cross", enabled=False)
    vec.set_gather_spawn_zones(spawn_zones if spawn_zones else ["forest_cross"])
    vec.set_navigation_task("forest_cross", "yellow_forest_2", enabled=False)

    # ------ Stats / checkpointing ------ #
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    episode_returns: list[float] = []
    episode_successes: list[float] = []
    episode_lengths: list[int] = []

    # Per-env live episode tracking
    live_returns = np.zeros(config.num_envs, dtype=np.float32)
    live_lengths = np.zeros(config.num_envs, dtype=np.int32)

    best_success = -1.0
    best_checkpoint_score = -1.0
    best_holdout_success = -1.0
    zone_cycle_offset = 0

    # Initial reset
    if config.objective.lower() == "gather" and config.gather_zone_scheduler == "round_robin":
        vec.force_gather_zones(spawn_zones, zone_cycle_offset)
        zone_cycle_offset += config.num_envs
    obs_np = vec.reset()  # [N, obs_dim]

    total_rollouts = config.episodes  # 1 outer-loop iter = 1 rollout

    for rollout in range(total_rollouts):
        progress = (rollout + 1) / max(1, total_rollouts)

        # Teacher probability (same decay schedule as single-env)
        gather_decay = max(0.0, 1.0 - (progress ** config.gather_teacher_decay_power))
        teacher_prob = max(
            config.gather_teacher_min,
            config.gather_teacher_start * gather_decay,
        )

        # ------ Collect rollout_steps steps from all N envs ------ #
        # Pre-allocate flat numpy buffers — no Python list.append, no np.stack later.
        T = config.rollout_steps
        N = config.num_envs
        buf_obs      = np.empty((T, N, obs_dim), dtype=np.float32)
        buf_actions  = np.empty((T, N),          dtype=np.int64)
        buf_logprobs = np.empty((T, N),          dtype=np.float32)
        buf_values   = np.empty((T, N),          dtype=np.float32)
        buf_rewards  = np.empty((T, N),          dtype=np.float32)
        buf_dones    = np.empty((T, N),          dtype=np.float32)

        # Collect entirely on CPU.
        with torch.no_grad():
            for step in range(T):
                obs_t = torch.from_numpy(obs_np).float()   # [N, obs_dim]
                logits, values, _ = model(obs_t)            # [N, A], [N, 1]
                dist = Categorical(logits=logits)

                sampled = dist.sample()                     # [N] tensor

                # Teacher override (independent per-env)
                if config.objective.lower() == "gather" and random.random() < teacher_prob:
                    actions_np = np.array([
                        _teacher_action_gather(env) for env in vec.envs
                    ], dtype=np.int64)
                    actions = torch.tensor(actions_np)      # tensor
                else:
                    actions = sampled
                    actions_np = actions.numpy()

                logprobs = dist.log_prob(actions)           # [N] tensor

                buf_obs[step]      = obs_np
                buf_actions[step]  = actions_np
                buf_logprobs[step] = logprobs.numpy()
                buf_values[step]   = values.squeeze(-1).numpy()

                # Step all envs
                next_obs_np, rewards, dones_f, infos = vec.step(actions_np.tolist())

                buf_rewards[step] = rewards
                buf_dones[step]   = dones_f

                live_returns += rewards
                live_lengths += 1

                # Collect completed episode stats
                for i, (done, info) in enumerate(zip(dones_f, infos)):
                    if done:
                        episode_returns.append(float(live_returns[i]))
                        episode_successes.append(float(info.get("gather_success", 0.0)))
                        episode_lengths.append(int(live_lengths[i]))
                        live_returns[i] = 0.0
                        live_lengths[i] = 0
                        # Assign next zone round-robin
                        if config.objective.lower() == "gather" and spawn_zones:
                            zone = spawn_zones[zone_cycle_offset % len(spawn_zones)]
                            zone_cycle_offset += 1
                            vec.envs[i].force_next_gather_spawn_zone(zone)

                obs_np = next_obs_np

        # ------ Bootstrap last value for GAE ------ #
        with torch.no_grad():
            last_obs_t = torch.from_numpy(obs_np).float()
            _, last_values, _ = model(last_obs_t)
            last_val_np = last_values.squeeze(-1).numpy()

        # ------ Compute GAE across rollout (flatten N×T) ------ #
        # buf_* are already [T, N, ...] ndarrays — aliased directly, no copy/stack needed
        obs_arr  = buf_obs       # [T, N, obs_dim]
        act_arr  = buf_actions   # [T, N]
        lp_arr   = buf_logprobs  # [T, N]
        val_arr  = buf_values    # [T, N]
        rew_arr  = buf_rewards   # [T, N]
        done_arr = buf_dones     # [T, N]

        # Process each env's trajectory independently for correct GAE
        all_advantages = np.zeros((T, N), dtype=np.float32)
        all_returns    = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            rew_t  = torch.tensor(rew_arr[:, n],  dtype=torch.float32)
            val_t  = torch.tensor(val_arr[:, n],  dtype=torch.float32)
            done_t = torch.tensor(done_arr[:, n], dtype=torch.float32)
            # Append bootstrap value
            val_with_bootstrap = torch.cat([val_t, torch.tensor([last_val_np[n]])])
            adv, ret = compute_gae(rew_t, val_with_bootstrap[:-1], done_t, config.gamma, config.gae_lambda)
            all_advantages[:, n] = adv.numpy()
            all_returns[:, n]    = ret.numpy()

        # Flatten to [T*N]
        flat_obs      = obs_arr.reshape(T * N, -1)
        flat_actions  = act_arr.reshape(T * N)
        flat_logprobs = lp_arr.reshape(T * N)
        flat_adv      = all_advantages.reshape(T * N)
        flat_returns  = all_returns.reshape(T * N)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # Move to tensors
        obs_batch      = torch.tensor(flat_obs,     dtype=torch.float32)
        action_batch   = torch.tensor(flat_actions, dtype=torch.long)
        old_lp_batch   = torch.tensor(flat_logprobs,dtype=torch.float32)
        adv_batch      = torch.tensor(flat_adv,     dtype=torch.float32)
        returns_batch  = torch.tensor(flat_returns, dtype=torch.float32)

        # ------ PPO update (same as single-env) ------ #
        model.train()
        idx = np.arange(T * N)
        for _ in range(config.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), config.minibatch_size):
                mb_idx = torch.tensor(idx[start: start + config.minibatch_size],
                                      dtype=torch.long)
                mb_obs     = obs_batch[mb_idx]
                mb_act     = action_batch[mb_idx]
                mb_old_lp  = old_lp_batch[mb_idx]
                mb_adv     = adv_batch[mb_idx]
                mb_ret     = returns_batch[mb_idx]

                logits, values, _ = model(mb_obs)
                dist_mb = Categorical(logits=logits)
                new_lp = dist_mb.log_prob(mb_act)
                entropy = dist_mb.entropy().mean()

                ratio = (new_lp - mb_old_lp).exp()
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss  = nn.functional.mse_loss(values.squeeze(-1), mb_ret)
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                optimizer.step()

        # ------ Logging ------ #
        if (rollout + 1) % config.log_interval == 0 and episode_returns:
            window = min(config.log_interval * config.num_envs, len(episode_returns))
            success_win = mean(episode_successes[-window:])
            return_win  = mean(episode_returns[-window:])
            length_win  = mean(episode_lengths[-window:])

            # Holdout eval
            holdout_success = 0.0
            if (
                config.objective.lower() == "gather"
                and holdout_zones
                and config.gather_eval_episodes > 0
                and (rollout + 1) % max(1, config.gather_eval_interval) == 0
            ):
                holdout_success = _evaluate_gather_holdout(
                    model,
                    obs_dim,
                    action_dim,
                    config.gather_eval_episodes,
                    config.max_steps,
                    config.seed + rollout + 1,
                    holdout_zones,
                    memory_size=config.memory_size,
                    inventory_full_target=config.inventory_full_target,
                )

            total_steps = (rollout + 1) * config.rollout_steps * config.num_envs
            print(
                f"rollout={rollout+1} steps={total_steps} "
                f"success_{window}={success_win:.2f} holdout={holdout_success:.2f} "
                f"return={return_win:.3f} ep_len={length_win:.1f}"
            )

            checkpoint_score = (
                success_win * 0.55 + holdout_success * 0.45
                if holdout_zones and config.objective.lower() == "gather"
                else success_win
            )
            best_success = max(best_success, success_win)
            best_holdout_success = max(best_holdout_success, holdout_success)

            if checkpoint_score > best_checkpoint_score and episode_successes:
                best_checkpoint_score = checkpoint_score
                ckpt = Path(config.checkpoint_dir) / f"map_to_map_ppo_{config.objective}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "memory_size": config.memory_size,
                        "episode": rollout + 1,
                        "objective": config.objective,
                    },
                    ckpt,
                )

            if wandb_run is not None:
                wandb_run.log({
                    "rollout": rollout + 1,
                    "total_steps": total_steps,
                    f"success_win_{window}": success_win,
                    "holdout_success": holdout_success,
                    f"return_win_{window}": return_win,
                    f"length_win_{window}": length_win,
                })

    vec.close()
    ref_env.close()

    final_success = mean(episode_successes[-50:]) if len(episode_successes) >= 50 else (
        mean(episode_successes) if episode_successes else 0.0
    )
    final_return = mean(episode_returns[-50:]) if len(episode_returns) >= 50 else (
        mean(episode_returns) if episode_returns else 0.0
    )

    print(
        f"final rollouts={total_rollouts} total_steps="
        f"{total_rollouts * config.rollout_steps * config.num_envs} "
        f"success={final_success:.2f} return={final_return:.3f}"
    )

    if wandb_run is not None:
        wandb_run.summary["final_success"] = final_success
        wandb_run.summary["final_return"] = final_return
        wandb_run.summary["best_success_window"] = best_success
        wandb_run.finish()

    return {
        "objective": config.objective,
        "final_success": final_success,
        "final_return": final_return,
        "best_success_window": best_success,
        "best_checkpoint_score": best_checkpoint_score,
        "best_holdout_success": best_holdout_success,
        "tail_success_window": final_success,
        "checkpoint_path": str(
            Path(config.checkpoint_dir) / f"map_to_map_ppo_{config.objective}_best.pt"
        ),
    }