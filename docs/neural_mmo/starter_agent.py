"""
STARTER AGENT - Simple Neural MMO Agent Examples

This file contains progressively complex agent implementations:
1. RandomAgent - Takes random actions
2. ScriptedAgent - Rule-based survival strategy
3. LearningAgent - Simple RL agent using Stable Baselines3

Use these as templates for your own experiments!
"""

import nmmo
from nmmo.core.env import Env
import numpy as np


# =============================================================================
# AGENT 1: Random Agent (Baseline)
# =============================================================================

class RandomAgent(nmmo.Agent):
    """
    Simplest possible agent - just takes random actions.

    This is the baseline to beat! Surprisingly, random actions
    can survive for a while if lucky with food/water spawns.
    """

    def __call__(self, obs):
        """
        Called each step to get agent's action.

        Args:
            obs: Observation dictionary from environment

        Returns:
            actions: Dictionary of actions to take
        """
        # Sample random action from action space
        return self.action_space.sample()


# =============================================================================
# AGENT 2: Scripted Survival Agent
# =============================================================================

class ScriptedAgent(nmmo.Agent):
    """
    Rule-based agent with simple survival strategy:
    1. Maintain food/water above threshold
    2. Explore to find resources
    3. Avoid combat (pacifist)

    This demonstrates how to parse observations and make decisions.
    """

    def __call__(self, obs):
        """
        Make decisions based on current state.

        Strategy:
        - If low on food/water → forage
        - If healthy → explore
        - If attacked → flee
        """
        # Initialize actions
        actions = {}

        # Parse observation to get agent state
        agent_data = obs['Entity']['Self']

        # Extract stats (indices depend on Neural MMO version)
        # Typical format: [id, row, col, health, food, water, ...]
        health = agent_data[3] if len(agent_data) > 3 else 100
        food = agent_data[4] if len(agent_data) > 4 else 100
        water = agent_data[5] if len(agent_data) > 5 else 100

        # Decision logic
        if food < 50 or water < 50:
            # Low resources → forage (move to nearby tiles)
            actions['Move'] = self._find_resource_direction(obs)
            actions['Use'] = 0  # Try to consume food/water
        else:
            # Healthy → explore (random movement)
            actions['Move'] = np.random.randint(0, 5)

        # Never attack (pacifist strategy)
        actions['Attack'] = 0

        return actions

    def _find_resource_direction(self, obs):
        """
        Simple heuristic: look for tiles with resources.

        In real implementation, you'd parse the Tile observation
        to find grass (food) or water tiles.
        """
        # For now, just move randomly
        # TODO: Implement actual tile parsing
        return np.random.randint(0, 5)


# =============================================================================
# AGENT 3: Learning Agent (Reinforcement Learning)
# =============================================================================

class LearningAgent:
    """
    Wrapper for training agents with Stable Baselines3.

    This demonstrates how to integrate Neural MMO with RL libraries.
    Note: This is a simplified single-agent example.
    """

    def __init__(self, algorithm='PPO'):
        """
        Initialize RL agent.

        Args:
            algorithm: RL algorithm to use (PPO, A2C, DQN, etc.)
        """
        from stable_baselines3 import PPO

        # Create environment
        config = nmmo.config.Default()
        config.PLAYERS = [nmmo.Agent] * 1  # Single agent for simplicity

        self.env = Env(config)

        # Create RL model
        self.model = PPO(
            'MlpPolicy',        # Simple feedforward network
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
        )

    def train(self, timesteps=100000):
        """
        Train the agent.

        Args:
            timesteps: Total number of environment steps
        """
        print(f"🚀 Training for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        print("✅ Training complete!")

    def save(self, path='neuralmmo_agent'):
        """Save trained model."""
        self.model.save(path)
        print(f"💾 Model saved to {path}")

    def load(self, path='neuralmmo_agent'):
        """Load trained model."""
        from stable_baselines3 import PPO
        self.model = PPO.load(path)
        print(f"📂 Model loaded from {path}")

    def evaluate(self, episodes=10):
        """
        Evaluate agent performance.

        Args:
            episodes: Number of episodes to run

        Returns:
            mean_reward: Average reward across episodes
        """
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=episodes
        )
        print(f"📊 Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def run_random_agent():
    """Run a random agent for demonstration."""
    print("🎲 Running RandomAgent...")

    # Create environment with 4 random agents
    config = nmmo.config.Default()
    config.PLAYERS = [RandomAgent] * 4

    env = Env(config)
    obs = env.reset()

    # Run for 200 steps
    for step in range(200):
        actions = {
            agent_id: env.agents[agent_id](obs[agent_id])
            for agent_id in env.agents
        }

        obs, rewards, dones, infos = env.step(actions)

        if step % 50 == 0:
            print(f"Step {step}: {len(env.agents)} agents alive")

    env.close()
    print("✅ Simulation complete!")


def run_scripted_agent():
    """Run a scripted agent for demonstration."""
    print("📜 Running ScriptedAgent...")

    config = nmmo.config.Default()
    config.PLAYERS = [ScriptedAgent] * 4

    env = Env(config)
    obs = env.reset()

    # Run for 200 steps
    survival_times = []
    for step in range(200):
        actions = {
            agent_id: env.agents[agent_id](obs[agent_id])
            for agent_id in env.agents
        }

        obs, rewards, dones, infos = env.step(actions)

        if step % 50 == 0:
            print(f"Step {step}: {len(env.agents)} agents alive")

        if not env.agents:
            print(f"All agents died at step {step}")
            break

    env.close()
    print("✅ Simulation complete!")


def train_learning_agent():
    """Train an RL agent."""
    print("🧠 Training LearningAgent with PPO...")

    agent = LearningAgent(algorithm='PPO')

    # Train
    agent.train(timesteps=50000)

    # Save
    agent.save('my_neuralmmo_agent')

    # Evaluate
    agent.evaluate(episodes=10)

    print("✅ Training and evaluation complete!")


if __name__ == "__main__":
    """
    Run this file to test different agents.

    Uncomment the agent you want to run:
    """

    # Option 1: Random agent (fastest)
    run_random_agent()

    # Option 2: Scripted agent (smarter)
    # run_scripted_agent()

    # Option 3: Learning agent (requires training time)
    # NOTE: This requires stable-baselines3 installed!
    # pip install stable-baselines3
    # train_learning_agent()

    print("\n" + "="*60)
    print("🎓 Next steps:")
    print("1. Modify ScriptedAgent to implement better strategies")
    print("2. Experiment with different reward functions")
    print("3. Try training with more agents (multi-agent RL)")
    print("4. Read concepts.md to learn about emergent behavior")
    print("="*60)
