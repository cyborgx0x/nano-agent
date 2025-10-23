"""
Example script demonstrating RL agent usage
Shows basic training, evaluation, and deployment
"""

import sys
sys.path.append('..')

from rl_agent import create_agent
import time


def example_1_quick_training():
    """Example 1: Quick training test"""
    print("=" * 70)
    print("Example 1: Quick DQN Training (1000 steps)")
    print("=" * 70)

    # Create agent
    agent = create_agent(
        algorithm='dqn',
        env_type='simplified',
        learning_rate=1e-3,
        verbose=1
    )

    # Quick training
    agent.train(
        total_timesteps=1000,
        log_dir='logs/rl/example1',
        save_dir='models/rl/example1',
        checkpoint_freq=500,
        eval_freq=500,
        eval_episodes=2
    )

    print("\nTraining completed! Check logs/rl/example1/ for results")


def example_2_evaluate_agent():
    """Example 2: Evaluate a trained agent"""
    print("=" * 70)
    print("Example 2: Evaluate Trained Agent")
    print("=" * 70)

    # Create agent
    agent = create_agent(
        algorithm='dqn',
        env_type='simplified',
        verbose=1
    )

    # Load trained model
    model_path = 'models/rl/example1/dqn_final.zip'
    try:
        agent.load(model_path)

        # Evaluate
        mean_reward, std_reward = agent.evaluate(n_episodes=5)

        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Run example_1_quick_training() first!")


def example_3_play_episode():
    """Example 3: Watch agent play one episode"""
    print("=" * 70)
    print("Example 3: Watch Agent Play")
    print("=" * 70)

    # Create agent
    agent = create_agent(
        algorithm='dqn',
        env_type='simplified',
        verbose=2  # Verbose output
    )

    # Load trained model
    model_path = 'models/rl/example1/dqn_final.zip'
    try:
        agent.load(model_path)

        # Play one episode
        total_reward, info = agent.play_episode(
            deterministic=True,
            render=False
        )

        print(f"\nEpisode Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Items gathered: {info['total_gathered']}")
        print(f"  Steps taken: {info['step']}")

    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Run example_1_quick_training() first!")


def example_4_custom_training():
    """Example 4: Custom training configuration"""
    print("=" * 70)
    print("Example 4: Custom PPO Training")
    print("=" * 70)

    # Create PPO agent with custom settings
    agent = create_agent(
        algorithm='ppo',
        env_type='simplified',
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1
    )

    # Custom training parameters
    agent.train(
        total_timesteps=5000,
        log_dir='logs/rl/example4_ppo',
        save_dir='models/rl/example4_ppo',
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=3
    )

    print("\nCustom training completed!")


def example_5_compare_algorithms():
    """Example 5: Compare different algorithms"""
    print("=" * 70)
    print("Example 5: Compare Algorithms (DQN vs PPO vs A2C)")
    print("=" * 70)

    algorithms = ['dqn', 'ppo', 'a2c']
    results = {}

    for algo in algorithms:
        print(f"\nTraining {algo.upper()}...")

        agent = create_agent(
            algorithm=algo,
            env_type='simplified',
            learning_rate=1e-3,
            verbose=0
        )

        # Train
        agent.train(
            total_timesteps=2000,
            log_dir=f'logs/rl/compare_{algo}',
            save_dir=f'models/rl/compare_{algo}',
            checkpoint_freq=1000,
            eval_freq=1000,
            eval_episodes=3
        )

        # Evaluate
        mean_reward, std_reward = agent.evaluate(n_episodes=5)
        results[algo] = (mean_reward, std_reward)

        print(f"{algo.upper()}: {mean_reward:.2f} ± {std_reward:.2f}")

    # Print comparison
    print("\n" + "=" * 70)
    print("Algorithm Comparison Results:")
    print("=" * 70)
    for algo, (mean, std) in results.items():
        print(f"{algo.upper():6s}: {mean:7.2f} ± {std:5.2f}")

    best_algo = max(results.items(), key=lambda x: x[1][0])[0]
    print(f"\nBest algorithm: {best_algo.upper()}")


def example_6_production_deployment():
    """Example 6: Production deployment workflow"""
    print("=" * 70)
    print("Example 6: Production Deployment")
    print("=" * 70)

    # Step 1: Train
    print("\nStep 1: Training agent...")
    agent = create_agent(
        algorithm='dqn',
        env_type='simplified',
        learning_rate=1e-4,
        verbose=1
    )

    agent.train(
        total_timesteps=10000,
        log_dir='logs/rl/production',
        save_dir='models/rl/production',
        checkpoint_freq=2000,
        eval_freq=2000,
        eval_episodes=5
    )

    # Step 2: Evaluate
    print("\nStep 2: Evaluating performance...")
    mean_reward, std_reward = agent.evaluate(n_episodes=10)

    if mean_reward > 20.0:  # Success threshold
        print(f"\n✓ Agent meets performance criteria!")
        print(f"  Mean reward: {mean_reward:.2f}")

        # Step 3: Deploy
        print("\nStep 3: Deploying agent...")
        # Save for production
        import shutil
        shutil.copy(
            'models/rl/production/dqn_final.zip',
            'models/production_agent.zip'
        )
        print("  Agent deployed to: models/production_agent.zip")

    else:
        print(f"\n✗ Agent performance insufficient")
        print(f"  Mean reward: {mean_reward:.2f} (need > 20.0)")
        print("  Recommendation: Train for more timesteps")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RL Examples')
    parser.add_argument('example', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='Example number to run')
    args = parser.parse_args()

    examples = {
        1: example_1_quick_training,
        2: example_2_evaluate_agent,
        3: example_3_play_episode,
        4: example_4_custom_training,
        5: example_5_compare_algorithms,
        6: example_6_production_deployment,
    }

    print("\n")
    examples[args.example]()
    print("\n")
