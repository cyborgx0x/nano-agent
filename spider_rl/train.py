#!/usr/bin/env python3
"""
Spider Standing RL Training Script
Phase 1: Standing and Balancing

This script trains an 8-legged spider robot to stand and balance using PPO.

Usage:
    python train.py --config config/spider_ppo_config.yaml [--headless] [--num_envs 4096]
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Parse arguments first (before Isaac Sim initialization)
parser = argparse.ArgumentParser(description="Train spider robot to stand and balance")
parser.add_argument("--config", type=str, default="config/spider_ppo_config.yaml",
                    help="Path to training configuration file")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Force display off at all times")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of parallel environments (overrides config)")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="Device to run training on")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Maximum training iterations (overrides config)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (overrides config)")

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules (after sim initialization)
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Import our environment
from envs.spider_standing_env import SpiderStandingEnv, SpiderStandingEnvCfg

# Import RSL-RL components
try:
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.env import VecEnv
except ImportError:
    print("Error: rsl_rl not found. Install with: pip install rsl_rl")
    sys.exit(1)


class IsaacLabVecEnvWrapper(VecEnv):
    """Wrapper to make Isaac Lab env compatible with RSL-RL"""

    def __init__(self, env: SpiderStandingEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_observations
        self.num_privileged_obs = None
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device

    def get_observations(self):
        """Get current observations"""
        return self.env.obs_buf

    def reset(self):
        """Reset all environments"""
        obs_dict = self.env.reset()
        return obs_dict["policy"]

    def step(self, actions):
        """Take a step in all environments"""
        obs_dict, rewards, dones, extras = self.env.step(actions)
        return obs_dict["policy"], rewards, dones, extras


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict, num_envs: int = None, device: str = "cuda:0") -> SpiderStandingEnv:
    """Create the spider standing environment"""
    # Create environment config
    env_cfg = SpiderStandingEnvCfg()

    # Override num_envs if specified
    if num_envs is not None:
        env_cfg.num_envs = num_envs
        env_cfg.scene.num_envs = num_envs

    # Set device
    env_cfg.sim.device = device

    # Create environment
    env = SpiderStandingEnv(cfg=env_cfg)

    return env


def create_ppo_runner(env: VecEnv, config: dict, device: str = "cuda:0"):
    """Create PPO runner with RSL-RL"""
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    # Create policy
    policy_cfg = config.get("policy", {})
    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
        critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
        activation=policy_cfg.get("activation", "elu"),
        init_noise_std=policy_cfg.get("init_noise_std", 1.0),
    ).to(device)

    # Create PPO algorithm
    algo_cfg = config.get("algorithm", {})
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=algo_cfg.get("num_learning_epochs", 5),
        num_mini_batches=algo_cfg.get("num_mini_batches", 4),
        clip_param=algo_cfg.get("clip_param", 0.2),
        gamma=algo_cfg.get("gamma", 0.99),
        lam=algo_cfg.get("lam", 0.95),
        value_loss_coef=algo_cfg.get("value_loss_coef", 1.0),
        entropy_coef=algo_cfg.get("entropy_coef", 0.01),
        learning_rate=algo_cfg.get("learning_rate", 3e-4),
        max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
        use_clipped_value_loss=algo_cfg.get("use_clipped_value_loss", True),
        schedule=algo_cfg.get("schedule", "adaptive"),
        desired_kl=algo_cfg.get("desired_kl", 0.01),
        device=device,
    )

    # Create runner
    runner_cfg = config.get("runner", {})
    training_cfg = config.get("training", {})

    runner = OnPolicyRunner(
        env=env,
        train_cfg=algo_cfg,
        log_dir=runner_cfg.get("checkpoint_path", "./spider_rl/logs"),
        device=device,
    )

    # Assign the PPO algorithm
    runner.alg = ppo

    # Set runner parameters
    runner.tot_timesteps = 0
    runner.tot_time = 0
    runner.current_learning_iteration = 0

    return runner


def train(runner, config: dict, max_iterations: int = None, checkpoint_path: str = None):
    """Training loop with logging and checkpointing"""
    runner_cfg = config.get("runner", {})
    logging_cfg = config.get("logging", {})

    max_iterations = max_iterations or runner_cfg.get("max_iterations", 5000)
    save_interval = runner_cfg.get("save_interval", 50)
    log_interval = logging_cfg.get("log_interval", 1)
    print_log_interval = logging_cfg.get("print_log_interval", 10)

    # Create log directory
    log_dir = runner_cfg.get("checkpoint_path", "./spider_rl/logs")
    os.makedirs(log_dir, exist_ok=True)

    # Tensorboard writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = logging_cfg.get("log_dir", "./spider_rl/logs/tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        use_tensorboard = True
    except ImportError:
        print("Warning: tensorboard not available, logging to console only")
        writer = None
        use_tensorboard = False

    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        runner.load(checkpoint_path)

    print("=" * 80)
    print("Starting Spider Standing Training")
    print("=" * 80)
    print(f"Number of environments: {runner.env.num_envs}")
    print(f"Number of observations: {runner.env.num_obs}")
    print(f"Number of actions: {runner.env.num_actions}")
    print(f"Max iterations: {max_iterations}")
    print(f"Save interval: {save_interval}")
    print(f"Device: {runner.device}")
    print("=" * 80)

    # Training loop
    for iteration in range(max_iterations):
        # Run one training iteration
        runner.alg.act_std = max(
            runner.alg.init_noise_std * (1.0 - iteration / max_iterations),
            0.1
        )

        # Collect experience
        if iteration == 0:
            runner.env.reset()

        # Run learning step
        mean_value_loss, mean_surrogate_loss = runner.alg.update()

        runner.current_learning_iteration = iteration

        # Compute metrics
        with torch.no_grad():
            # Get environment metrics
            if hasattr(runner.env.env, 'get_metrics'):
                env_metrics = runner.env.env.get_metrics()
            else:
                env_metrics = {}

            # Compute training statistics
            mean_reward = runner.alg.mean_value_loss  # Placeholder
            mean_episode_length = runner.env.max_episode_length

        # Logging
        if iteration % log_interval == 0 and use_tensorboard:
            writer.add_scalar("Loss/value_loss", mean_value_loss, iteration)
            writer.add_scalar("Loss/surrogate_loss", mean_surrogate_loss, iteration)
            writer.add_scalar("Policy/learning_rate", runner.alg.learning_rate, iteration)
            writer.add_scalar("Policy/action_std", runner.alg.act_std, iteration)

            # Log environment metrics
            for key, value in env_metrics.items():
                writer.add_scalar(f"Metrics/{key}", value, iteration)

        # Console logging
        if iteration % print_log_interval == 0:
            print(f"\n[Iteration {iteration}/{max_iterations}]")
            print(f"  Value Loss: {mean_value_loss:.4f}")
            print(f"  Surrogate Loss: {mean_surrogate_loss:.4f}")
            print(f"  Action Std: {runner.alg.act_std:.4f}")

            for key, value in env_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Save checkpoint
        if iteration % save_interval == 0 and iteration > 0:
            checkpoint_file = os.path.join(log_dir, f"checkpoint_{iteration}.pt")
            runner.save(checkpoint_file)
            print(f"\nCheckpoint saved: {checkpoint_file}")

    # Save final checkpoint
    final_checkpoint = os.path.join(log_dir, "checkpoint_final.pt")
    runner.save(final_checkpoint)
    print(f"\nFinal checkpoint saved: {final_checkpoint}")

    if use_tensorboard:
        writer.close()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


def main():
    """Main training function"""
    # Load configuration
    config_path = args_cli.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Override config with CLI arguments
    if args_cli.seed is not None:
        config["seed"] = args_cli.seed

    # Set random seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    print("Creating environment...")
    env = create_environment(
        config,
        num_envs=args_cli.num_envs,
        device=args_cli.device
    )

    # Wrap environment for RSL-RL
    env_wrapped = IsaacLabVecEnvWrapper(env)

    # Create PPO runner
    print("Creating PPO runner...")
    runner = create_ppo_runner(env_wrapped, config, device=args_cli.device)

    # Start training
    train(
        runner,
        config,
        max_iterations=args_cli.max_iterations,
        checkpoint_path=args_cli.checkpoint
    )

    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        simulation_app.close()
