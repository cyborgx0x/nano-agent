from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from state_sim.ppo import PPOTrainConfig, train_ppo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO trainer for state_sim map-to-map task"
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-every-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nano-agent-state-sim")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--checkpoint-dir", type=str, default="runs/state_sim")
    parser.add_argument(
        "--objective",
        type=str,
        default="navigation",
        choices=["navigation", "gather", "mixed"],
    )
    parser.add_argument("--mixed-navigation-ratio", type=float, default=0.5)
    parser.add_argument(
        "--gather-train-mode",
        type=str,
        default="multi",
        choices=["multi", "single"],
    )
    parser.add_argument(
        "--gather-zone-scheduler",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
    )
    parser.add_argument("--gather-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--gather-eval-interval", type=int, default=30)
    parser.add_argument("--gather-eval-episodes", type=int, default=30)
    parser.add_argument("--gather-teacher-min", type=float, default=0.35)
    parser.add_argument("--gather-teacher-decay-power", type=float, default=2.0)
    parser.add_argument("--gather-teacher-recovery-boost", type=float, default=0.20)
    parser.add_argument("--gather-teacher-recovery-ratio", type=float, default=0.75)
    parser.add_argument("--gather-teacher-boost-decay", type=float, default=0.96)
    parser.add_argument("--inventory-full-target", type=int, default=20)
    parser.add_argument("--memory-size", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PPOTrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        update_every_episodes=args.update_every_episodes,
        log_interval=args.log_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        checkpoint_dir=args.checkpoint_dir,
        objective=args.objective,
        mixed_navigation_ratio=args.mixed_navigation_ratio,
        gather_train_mode=args.gather_train_mode,
        gather_zone_scheduler=args.gather_zone_scheduler,
        gather_holdout_ratio=args.gather_holdout_ratio,
        gather_eval_interval=args.gather_eval_interval,
        gather_eval_episodes=args.gather_eval_episodes,
        gather_teacher_min=args.gather_teacher_min,
        gather_teacher_decay_power=args.gather_teacher_decay_power,
        gather_teacher_recovery_boost=args.gather_teacher_recovery_boost,
        gather_teacher_recovery_ratio=args.gather_teacher_recovery_ratio,
        gather_teacher_boost_decay=args.gather_teacher_boost_decay,
        inventory_full_target=args.inventory_full_target,
        memory_size=args.memory_size,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
    )
    train_ppo(config)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    main()
