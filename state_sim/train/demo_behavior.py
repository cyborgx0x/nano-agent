from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from state_sim.environment import AlbionStateSim
from state_sim.ppo import ActorCritic, ObsEncoder


def _sample_navigation_task(env: AlbionStateSim) -> tuple[str, str]:
    zones = [z for z, t in env.zone_types.items() if t != "city"]
    start_zone = random.choice(zones)
    goal_zone = random.choice([z for z in zones if z != start_zone])
    return start_zone, goal_zone


def _configure_task(
    env: AlbionStateSim, objective: str, mixed_navigation_ratio: float
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

    raise ValueError("objective must be one of: navigation, gather, mixed")


def _episode_success(
    info: dict[str, float | int | bool | str], episode_mode: str
) -> float:
    if episode_mode == "navigation":
        return float(info.get("navigation_success", False))
    if episode_mode == "gather":
        return float(info.get("gather_success", 0.0))
    return float(info.get("success_rate", 0.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo trained PPO behavior with labeled render"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--objective",
        type=str,
        default="gather",
        choices=["navigation", "gather", "mixed"],
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render-size", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--video-path", type=str, default="runs/state_sim/demo_behavior.mp4"
    )
    parser.add_argument("--mixed-navigation-ratio", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = AlbionStateSim(seed=args.seed, max_steps=args.max_steps)
    encoder = ObsEncoder(env)
    model = ActorCritic(encoder.obs_dim, env.action_count).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    import cv2

    Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (args.render_size, args.render_size),
    )

    success_list: list[float] = []

    try:
        for ep in range(args.episodes):
            episode_mode = _configure_task(
                env, args.objective, args.mixed_navigation_ratio
            )
            obs = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0

            while not done:
                obs_vec = encoder.encode(obs)
                obs_t = torch.from_numpy(obs_vec).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(obs_t)
                    action = int(torch.argmax(logits, dim=-1).item())

                obs, reward, done, info = env.step(action)
                ep_return += float(reward)
                ep_steps += 1

                frame = env.render(mode="rgb_array", size=args.render_size)
                writer.write(frame)

            success = _episode_success(info, episode_mode)
            success_list.append(success)
            print(
                f"demo episode={ep+1} mode={episode_mode} success={success:.0f} "
                f"return={ep_return:.3f} len={ep_steps}"
            )

    finally:
        writer.release()
        env.close()

    avg_success = sum(success_list) / max(1, len(success_list))
    print(
        f"demo done: episodes={args.episodes} avg_success={avg_success:.3f} video={args.video_path}"
    )


if __name__ == "__main__":
    main()
