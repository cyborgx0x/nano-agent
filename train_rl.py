"""
Training script for Deep Reinforcement Learning agent
Trains agent to play Albion Online gathering task
"""

import argparse
import os
import yaml
from datetime import datetime

import torch
from rl_agent import create_agent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RL agent for Albion Online gathering'
    )

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='dqn',
                       choices=['dqn', 'ppo', 'a2c'],
                       help='RL algorithm to use')

    # Environment
    parser.add_argument('--env-type', type=str, default='simplified',
                       choices=['full', 'simplified'],
                       help='Environment type')
    parser.add_argument('--model-path', type=str, default='model.pt',
                       help='Path to YOLO detection model')

    # Training parameters
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=10000,
                       help='Replay buffer size (for DQN)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Checkpointing
    parser.add_argument('--checkpoint-freq', type=int, default=5000,
                       help='Checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes')

    # Output
    parser.add_argument('--log-dir', type=str, default='logs/rl',
                       help='Log directory')
    parser.add_argument('--save-dir', type=str, default='models/rl',
                       help='Model save directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'play'],
                       help='Mode: train, eval, or play')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to saved model (for eval/play)')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of episodes for eval mode')

    # Other
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Set random seed
    if args.seed is not None:
        import numpy as np
        import random
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Print configuration
    print("=" * 70)
    print("Deep RL Training - Albion Online Gathering")
    print("=" * 70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Environment: {args.env_type}")
    print(f"Mode: {args.mode}")
    if args.mode == 'train':
        print(f"Timesteps: {args.timesteps:,}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Device: {args.device}")
    print("=" * 70)
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")

    # Create agent
    print(f"\nCreating {args.algorithm.upper()} agent...")

    agent = create_agent(
        algorithm=args.algorithm,
        env_type=args.env_type,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        device=args.device,
        verbose=args.verbose
    )

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.name}_{args.algorithm}_{timestamp}"

    log_dir = os.path.join(args.log_dir, exp_name)
    save_dir = os.path.join(args.save_dir, exp_name)

    # Execute mode
    if args.mode == 'train':
        # Train agent
        print("\n🚀 Starting training...")

        agent.train(
            total_timesteps=args.timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes
        )

        print("\n✓ Training completed!")
        print(f"\nLogs saved to: {log_dir}")
        print(f"Models saved to: {save_dir}")

        # TensorBoard info
        print("\nTo view training progress:")
        print(f"  tensorboard --logdir={log_dir}")

    elif args.mode == 'eval':
        # Evaluate agent
        if args.load_model is None:
            print("Error: --load-model required for eval mode")
            return

        print(f"\nLoading model: {args.load_model}")
        agent.load(args.load_model)

        print(f"\nEvaluating for {args.n_episodes} episodes...")
        mean_reward, std_reward = agent.evaluate(
            n_episodes=args.n_episodes,
            deterministic=True
        )

        print("\n📊 Evaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    elif args.mode == 'play':
        # Play one episode
        if args.load_model is None:
            print("Error: --load-model required for play mode")
            return

        print(f"\nLoading model: {args.load_model}")
        agent.load(args.load_model)

        print("\n🎮 Playing episode...")
        total_reward, info = agent.play_episode(
            deterministic=True,
            render=True
        )

        print("\n📊 Episode Results:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Items gathered: {info.get('total_gathered', 0)}")

    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
