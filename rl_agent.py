"""
Deep Reinforcement Learning Agent for Albion Online
Supports DQN, PPO, and A2C algorithms via stable-baselines3
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import torch

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from game_env import AlbionGatherEnv, SimplifiedAlbionEnv


class GatheringMetricsCallback(BaseCallback):
    """
    Custom callback for logging gathering-specific metrics
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_gathered = []

    def _on_step(self) -> bool:
        # Log custom metrics if episode finished
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals['infos'][i]
                if 'total_gathered' in info:
                    self.total_gathered.append(info['total_gathered'])
                    self.logger.record('gathering/total_gathered', info['total_gathered'])
                    self.logger.record('gathering/inventory_final', info.get('inventory', 0))

                    if self.verbose > 0:
                        print(f"Episode finished! Gathered: {info['total_gathered']} items")

        return True


class AlbionRLAgent:
    """
    Reinforcement Learning agent for Albion Online gathering

    Supports multiple RL algorithms:
    - DQN: Deep Q-Network (good for discrete actions)
    - PPO: Proximal Policy Optimization (stable, popular)
    - A2C: Advantage Actor-Critic (faster than PPO)
    """

    def __init__(
        self,
        algorithm: str = 'dqn',
        env_type: str = 'simplified',
        model_path: str = 'model.pt',
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        buffer_size: int = 10000,
        gamma: float = 0.99,
        device: str = 'auto',
        verbose: int = 1
    ):
        """
        Initialize RL agent

        Args:
            algorithm: 'dqn', 'ppo', or 'a2c'
            env_type: 'full' or 'simplified'
            model_path: Path to YOLO detection model
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            buffer_size: Replay buffer size (for DQN)
            gamma: Discount factor
            device: 'auto', 'cuda', or 'cpu'
            verbose: Verbosity level
        """
        self.algorithm = algorithm.lower()
        self.env_type = env_type
        self.model_path = model_path
        self.verbose = verbose

        # Create environment
        self.env = self._create_env()

        # Algorithm-specific parameters
        self.params = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'device': device,
            'verbose': verbose,
        }

        if algorithm == 'dqn':
            self.params.update({
                'batch_size': batch_size,
                'buffer_size': buffer_size,
                'exploration_fraction': 0.3,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'learning_starts': 1000,
                'target_update_interval': 500,
            })
        elif algorithm == 'ppo':
            self.params.update({
                'batch_size': batch_size,
                'n_steps': 2048,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01,
            })
        elif algorithm == 'a2c':
            self.params.update({
                'n_steps': 5,
                'ent_coef': 0.01,
            })

        # Create model
        self.model = self._create_model()

        print(f"AlbionRLAgent initialized")
        print(f"  Algorithm: {self.algorithm.upper()}")
        print(f"  Environment: {self.env_type}")
        print(f"  Device: {self.params['device']}")

    def _create_env(self):
        """Create and wrap environment"""
        if self.env_type == 'simplified':
            env = SimplifiedAlbionEnv(model_path=self.model_path)
        else:
            env = AlbionGatherEnv(model_path=self.model_path)

        # Wrap in Monitor for logging
        env = Monitor(env)

        return env

    def _create_model(self):
        """Create RL model based on algorithm"""
        if self.algorithm == 'dqn':
            model = DQN(
                'MlpPolicy',
                self.env,
                **self.params
            )
        elif self.algorithm == 'ppo':
            model = PPO(
                'MlpPolicy',
                self.env,
                **self.params
            )
        elif self.algorithm == 'a2c':
            model = A2C(
                'MlpPolicy',
                self.env,
                **self.params
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return model

    def train(
        self,
        total_timesteps: int = 50000,
        log_dir: str = 'logs/rl',
        save_dir: str = 'models/rl',
        checkpoint_freq: int = 5000,
        eval_freq: int = 5000,
        eval_episodes: int = 5
    ):
        """
        Train the RL agent

        Args:
            total_timesteps: Total training steps
            log_dir: Directory for logs
            save_dir: Directory for saved models
            checkpoint_freq: Frequency of checkpoints
            eval_freq: Frequency of evaluation
            eval_episodes: Number of evaluation episodes
        """
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Setup logger
        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(logger)

        # Create callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_dir,
            name_prefix=f"{self.algorithm}_agent"
        )
        callbacks.append(checkpoint_callback)

        # Custom metrics callback
        metrics_callback = GatheringMetricsCallback(verbose=self.verbose)
        callbacks.append(metrics_callback)

        # Eval callback (optional)
        if eval_freq > 0:
            eval_env = Monitor(self._create_env())
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                best_model_save_path=save_dir,
                log_path=log_dir,
                deterministic=True
            )
            callbacks.append(eval_callback)

        callback_list = CallbackList(callbacks)

        # Train
        print(f"\n{'='*60}")
        print(f"Starting RL Training - {self.algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Log directory: {log_dir}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(save_dir, f"{self.algorithm}_final.zip")
        self.model.save(final_path)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Final model saved: {final_path}")
        print(f"{'='*60}\n")

        return self.model

    def load(self, model_path: str):
        """Load a trained model"""
        if self.algorithm == 'dqn':
            self.model = DQN.load(model_path, env=self.env)
        elif self.algorithm == 'ppo':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(model_path, env=self.env)

        print(f"Model loaded from: {model_path}")

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """
        Predict action from observation

        Args:
            observation: Current state
            deterministic: Use deterministic policy (no exploration)

        Returns:
            action: Predicted action
            state: Agent state (for recurrent policies)
        """
        return self.model.predict(observation, deterministic=deterministic)

    def play_episode(self, deterministic: bool = True, render: bool = False):
        """
        Play one episode using learned policy

        Args:
            deterministic: Use deterministic policy
            render: Render environment

        Returns:
            total_reward: Total episode reward
            info: Episode information
        """
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        steps = 0

        print("Starting episode...")

        while not done:
            action, _states = self.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if render:
                self.env.render()

            if self.verbose > 0:
                action_names = ['Cotton', 'Flax', 'Hemp', 'Wait']
                print(f"Step {steps}: Action={action_names[action]} "
                      f"Reward={reward:.2f} Inventory={info.get('inventory', 0)}")

        print(f"\nEpisode finished!")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Items gathered: {info.get('total_gathered', 0)}")

        return total_reward, info

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """
        Evaluate agent over multiple episodes

        Args:
            n_episodes: Number of episodes
            deterministic: Use deterministic policy

        Returns:
            mean_reward: Mean episode reward
            std_reward: Standard deviation of rewards
        """
        rewards = []
        gathered_items = []

        print(f"Evaluating for {n_episodes} episodes...")

        for i in range(n_episodes):
            reward, info = self.play_episode(deterministic=deterministic)
            rewards.append(reward)
            gathered_items.append(info.get('total_gathered', 0))

            print(f"Episode {i+1}/{n_episodes}: "
                  f"Reward={reward:.2f}, Gathered={info.get('total_gathered', 0)}")

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_gathered = np.mean(gathered_items)

        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Mean gathered: {mean_gathered:.2f}")

        return mean_reward, std_reward


def create_agent(algorithm: str = 'dqn', **kwargs) -> AlbionRLAgent:
    """
    Factory function to create RL agent

    Args:
        algorithm: 'dqn', 'ppo', or 'a2c'
        **kwargs: Additional arguments for AlbionRLAgent

    Returns:
        agent: Configured RL agent
    """
    return AlbionRLAgent(algorithm=algorithm, **kwargs)


if __name__ == '__main__':
    # Quick test
    print("Testing RL Agent...")

    # Create agent with simplified environment
    agent = create_agent(
        algorithm='dqn',
        env_type='simplified',
        learning_rate=1e-3,
        verbose=1
    )

    print("\nAgent created successfully!")
    print("To train, run:")
    print("  agent.train(total_timesteps=10000)")
