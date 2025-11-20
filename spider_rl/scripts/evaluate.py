#!/usr/bin/env python3
"""
Evaluate trained spider standing policy

Usage:
    python scripts/evaluate.py --checkpoint logs/checkpoint_1000.pt [--num_episodes 10]
"""

import argparse
import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate spider standing policy")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to evaluate")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments for evaluation")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Run headless (no visualization)")
parser.add_argument("--record_video", action="store_true", default=False,
                    help="Record video of evaluation")

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import environment
from envs.spider_standing_env import SpiderStandingEnv, SpiderStandingEnvCfg


def evaluate(checkpoint_path: str, num_episodes: int = 10, num_envs: int = 1):
    """Evaluate trained policy"""

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Create environment
    env_cfg = SpiderStandingEnvCfg()
    env_cfg.num_envs = num_envs
    env = SpiderStandingEnv(cfg=env_cfg)

    # Load policy
    policy = checkpoint.get("model_state_dict", None)
    if policy is None:
        print("Warning: No model state dict found in checkpoint")
        return

    # Evaluation loop
    print(f"\nEvaluating for {num_episodes} episodes...")

    episode_rewards = []
    success_count = 0

    obs_dict = env.reset()
    obs = obs_dict["policy"]

    episodes_completed = 0
    episode_reward = torch.zeros(num_envs, device=env.device)

    while episodes_completed < num_episodes:
        # Get action from policy
        with torch.no_grad():
            # Simple forward pass (you may need to adapt based on your policy structure)
            # This is a placeholder - actual inference depends on policy architecture
            action = torch.randn(num_envs, env.num_actions, device=env.device) * 0.1

        # Step environment
        obs_dict, reward, done, extras = env.step(action)
        obs = obs_dict["policy"]

        episode_reward += reward

        # Check for episode completion
        if done.any():
            for i in range(num_envs):
                if done[i]:
                    episode_rewards.append(episode_reward[i].item())
                    episodes_completed += 1

                    # Check success
                    if hasattr(env, 'standing_success'):
                        if env.standing_success[i] > 0.5:
                            success_count += 1

                    episode_reward[i] = 0.0

                    if episodes_completed >= num_episodes:
                        break

    # Print statistics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print(f"Success Rate: {success_count / num_episodes * 100:.1f}%")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    try:
        evaluate(
            checkpoint_path=args_cli.checkpoint,
            num_episodes=args_cli.num_episodes,
            num_envs=args_cli.num_envs
        )
    finally:
        simulation_app.close()
