from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from state_sim.environment import AlbionStateSim
from state_sim.train_map_to_map_ppo import ActorCritic, ObsEncoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoint on map-to-map task"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/state_sim/map_to_map_ppo_best.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--video-path",
        type=str,
        default="runs/state_sim/eval_episode.mp4",
        help="Output video path when --save-video is set",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--start-zone",
        type=str,
        default=None,
        help="Optional fixed start zone for all episodes",
    )
    parser.add_argument(
        "--goal-zone",
        type=str,
        default=None,
        help="Optional fixed goal zone for all episodes",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env = AlbionStateSim(seed=args.seed, max_steps=args.max_steps)
    obs_encoder = ObsEncoder(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_encoder.obs_dim, env.action_count).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    zones = [z for z, t in env.zone_types.items() if t != "city"]

    video_writer = None
    if args.save_video:
        try:
            import cv2

            Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.video_path,
                fourcc,
                float(args.fps),
                (args.render_size, args.render_size),
            )
        except Exception as exc:
            env.close()
            raise RuntimeError(f"Unable to create video writer: {exc}") from exc

    success_flags: list[float] = []
    returns: list[float] = []
    lengths: list[float] = []

    for ep in range(args.episodes):
        if args.start_zone and args.goal_zone:
            start_zone, goal_zone = args.start_zone, args.goal_zone
        else:
            start_zone = random.choice(zones)
            goal_zone = random.choice([z for z in zones if z != start_zone])

        env.set_navigation_task(
            start_zone=start_zone, goal_zone=goal_zone, enabled=True
        )
        obs = env.reset()

        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            obs_vec = obs_encoder.encode(obs)
            obs_t = torch.from_numpy(obs_vec).float().to(device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            if args.render or args.save_video:
                frame = env.render(mode="rgb_array", size=args.render_size)
                if args.render:
                    try:
                        env.render(mode="human", size=args.render_size)
                    except RuntimeError:
                        pass
                if video_writer is not None:
                    video_writer.write(frame)

        success_flags.append(float(info.get("navigation_success", False)))
        returns.append(ep_return)
        lengths.append(float(ep_len))

        if (ep + 1) % 20 == 0:
            print(
                f"eval episode={ep+1} "
                f"success_20={mean(success_flags[-20:]):.2f} "
                f"return_20={mean(returns[-20:]):.3f} "
                f"len_20={mean(lengths[-20:]):.1f}"
            )

    if video_writer is not None:
        video_writer.release()

    print(
        f"final_eval episodes={args.episodes} "
        f"success={mean(success_flags):.3f} "
        f"return={mean(returns):.3f} "
        f"length={mean(lengths):.1f}"
    )

    env.close()


if __name__ == "__main__":
    evaluate(_parse_args())
