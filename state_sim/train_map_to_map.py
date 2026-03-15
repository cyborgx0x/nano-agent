from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from state_sim.environment import Action, AlbionStateSim


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


def _greedy_action(
    obs: dict[str, float | int | str | bool], env: AlbionStateSim
) -> int:
    zone_id = str(obs["zone_id"])
    x = float(obs["x"])
    y = float(obs["y"])
    nearest_gate = float(obs["nearest_gate_dist"])
    goal = str(obs["goal_zone"])

    if zone_id == goal:
        return env.move_action_count + int(Action.IDLE)

    if nearest_gate < 0.15:
        return env.move_action_count + int(Action.INTERACT)

    next_zone = _shortest_path_first_step(env.zone_graph, zone_id, goal)
    neighbors = env.zone_graph[zone_id]
    if next_zone in neighbors:
        target_idx = neighbors.index(next_zone)
        angle = (
            2 * 3.141592653589793 * target_idx / max(1, len(neighbors))
        ) - 3.141592653589793 / 2
        target_x = 0.5 + 0.40 * __import__("math").cos(angle)
        target_y = 0.5 + 0.40 * __import__("math").sin(angle)
        return _target_cell(target_x, target_y, env.grid_size)

    jitter_x = min(0.95, max(0.05, x + random.uniform(-0.2, 0.2)))
    jitter_y = min(0.95, max(0.05, y + random.uniform(-0.2, 0.2)))
    return _target_cell(jitter_x, jitter_y, env.grid_size)


def _state_key(
    obs: dict[str, float | int | str | bool],
) -> tuple[str, str, int, int, int]:
    zone = str(obs["zone_id"])
    goal = str(obs["goal_zone"])
    gate_bin = 1 if float(obs["nearest_gate_dist"]) < 0.18 else 0
    hp_bin = min(4, max(0, int(float(obs["hp"]) * 5)))
    threat_bin = min(4, max(0, int(float(obs["threat_score"]) * 5)))
    return zone, goal, gate_bin, hp_bin, threat_bin


def train_navigation(
    episodes: int = 500,
    max_steps: int = 120,
    alpha: float = 0.18,
    gamma: float = 0.97,
    epsilon_start: float = 0.75,
    epsilon_end: float = 0.08,
    log_interval: int = 25,
    use_wandb: bool = False,
    wandb_project: str = "nano-agent-state-sim",
    wandb_entity: str | None = None,
    wandb_mode: str = "offline",
) -> None:
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                mode=wandb_mode,
                config={
                    "episodes": episodes,
                    "max_steps": max_steps,
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "algorithm": "tabular_q_learning",
                    "task": "map_to_map_navigation",
                },
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    env = AlbionStateSim(seed=42, max_steps=max_steps)

    zones = [z for z, t in env.zone_types.items() if t != "city"]
    q: dict[tuple[str, str, int, int, int], list[float]] = defaultdict(
        lambda: [0.0] * env.action_count
    )

    nav_successes: list[float] = []
    returns: list[float] = []

    for ep in range(episodes):
        start_zone = random.choice(zones)
        goal_zone = random.choice([z for z in zones if z != start_zone])
        env.set_navigation_task(
            start_zone=start_zone, goal_zone=goal_zone, enabled=True
        )
        obs = env.reset()

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - ep / episodes
        )

        done = False
        ep_return = 0.0
        while not done:
            state = _state_key(obs)

            if random.random() < epsilon:
                if random.random() < 0.35:
                    action = _greedy_action(obs, env)
                else:
                    action = random.randrange(env.action_count)
            else:
                values = q[state]
                action = int(max(range(env.action_count), key=lambda a: values[a]))

            next_obs, reward, done, info = env.step(action)
            ep_return += reward

            next_state = _state_key(next_obs)
            td_target = reward + (0.0 if done else gamma * max(q[next_state]))
            q[state][action] += alpha * (td_target - q[state][action])

            obs = next_obs

        nav_successes.append(float(info.get("navigation_success", False)))
        returns.append(ep_return)

        if (ep + 1) % log_interval == 0:
            window = min(log_interval, len(nav_successes))
            success_win = mean(nav_successes[-window:])
            return_win = mean(returns[-window:])
            print(
                f"episode={ep+1} "
                f"success_{window}={success_win:.2f} "
                f"return_{window}={return_win:.3f} "
                f"epsilon={epsilon:.3f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode": ep + 1,
                        "success_window": success_win,
                        "return_window": return_win,
                        "epsilon": epsilon,
                        "q_table_states": len(q),
                    }
                )

    final_success = mean(nav_successes)
    final_return = mean(returns)
    print(
        f"final episodes={episodes} "
        f"success={final_success:.2f} "
        f"return={final_return:.3f}"
    )

    if wandb_run is not None:
        wandb_run.summary["final_success"] = final_success
        wandb_run.summary["final_return"] = final_return
        wandb_run.summary["q_table_states"] = len(q)
        wandb_run.finish()

    env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train map-to-map navigation policy")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon-start", type=float, default=0.75)
    parser.add_argument("--epsilon-end", type=float, default=0.08)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nano-agent-state-sim")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["offline", "online", "disabled"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_navigation(
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        log_interval=args.log_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )


if __name__ == "__main__":
    main()
