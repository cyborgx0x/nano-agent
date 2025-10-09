# Neural MMO - Setup Guide

This guide will walk you through installing and running Neural MMO.

---

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (Linux recommended)
- **Python**: 3.8 - 3.11
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core recommended for parallel training

### Knowledge Requirements
- Basic Python programming
- Understanding of NumPy
- Familiarity with Gym/Gymnasium environments (helpful but not required)

---

## Installation

### Step 1: Create Virtual Environment

```bash
# Create a new conda environment (recommended)
conda create -n neuralmmo python=3.10
conda activate neuralmmo

# OR use venv
python -m venv neuralmmo_env
source neuralmmo_env/bin/activate  # Linux/Mac
# neuralmmo_env\Scripts\activate  # Windows
```

### Step 2: Install Neural MMO

```bash
# Install from PyPI (stable)
pip install nmmo

# OR install from GitHub (latest development version)
git clone https://github.com/NeuralMMO/environment.git
cd environment
pip install -e .
```

### Step 3: Install Dependencies

Neural MMO requires:

```bash
# Core dependencies
pip install numpy
pip install gymnasium  # OpenAI Gym successor
pip install pettingzoo  # Multi-agent RL wrapper
pip install vec-noise   # Procedural map generation

# Optional: For training with RL
pip install stable-baselines3  # PPO, A2C, DQN implementations
pip install torch  # PyTorch (for neural networks)

# Optional: For visualization
pip install matplotlib
pip install imageio
```

### Step 4: Verify Installation

```python
import nmmo
from nmmo.core.env import Env

# Create a simple environment
env = Env()
print("✓ Neural MMO installed successfully!")
print(f"Environment version: {nmmo.__version__}")
```

---

## Quick Start Example

### Running a Random Agent

```python
import nmmo
from nmmo.core.env import Env

# Create environment with 4 agents
config = nmmo.config.Default()
config.PLAYERS = [nmmo.Agent] * 4  # 4 random agents

env = Env(config)
obs = env.reset()

# Run for 100 steps
for step in range(100):
    # Random actions for all agents
    actions = {
        agent_id: env.action_space(agent_id).sample()
        for agent_id in env.agents
    }

    obs, rewards, dones, infos = env.step(actions)

    print(f"Step {step}: {len(env.agents)} agents alive")

    if not env.agents:
        print("All agents dead!")
        break

env.close()
```

---

## Configuration Options

Neural MMO is highly configurable. Key settings:

### Map Settings

```python
config.MAP_SIZE = 128           # Map dimensions (128x128 tiles)
config.MAP_GENERATOR = 'perlin' # Procedural generation algorithm
config.TERRAIN_CENTER = 64      # Center coordinates
```

### Agent Settings

```python
config.PLAYERS = [nmmo.Agent] * 16  # Number and types of agents
config.SPAWN_CONCURRENT = 16        # Agents spawned together
config.PLAYER_VISION_RADIUS = 7     # How far agents can see
```

### Survival Settings

```python
config.RESOURCE_SYSTEM_ENABLED = True  # Enable food/water
config.COMBAT_SYSTEM_ENABLED = True    # Enable combat
config.EXCHANGE_SYSTEM_ENABLED = True  # Enable trading
```

### Episode Settings

```python
config.HORIZON = 1024           # Max timesteps per episode
config.IMMORTAL = False         # Agents can die
```

---

## Environment Structure

### Observation Space

Each agent receives observations as a dictionary:

```python
{
    'Entity': {
        'Self': [...],      # Agent's own stats (HP, food, water, etc.)
        'Other': [...],     # Nearby agents
    },
    'Tile': {
        'Continuous': [...], # Terrain features
        'Discrete': [...]    # Tile types (grass, water, etc.)
    },
    'Inventory': [...],      # Items held
    'Market': [...]          # Available trades
}
```

### Action Space

Actions are structured as MultiDiscrete:

```python
{
    'Move': [0-4],      # North, South, East, West, Stay
    'Attack': [0-4],    # Melee, Range, Mage, or None
    'Use': [0-N],       # Which item to use
    'Give': [0-N],      # Trading actions
}
```

---

## Visualization

### Rendering the Environment

```python
import nmmo
from nmmo.core.env import Env

config = nmmo.config.Default()
config.RENDER = True  # Enable rendering

env = Env(config)
obs = env.reset()

for step in range(100):
    actions = {agent_id: env.action_space(agent_id).sample()
               for agent_id in env.agents}
    obs, rewards, dones, infos = env.step(actions)

    # Render to image
    img = env.render(mode='rgb_array')

    # Save or display
    # plt.imshow(img)
    # plt.show()

env.close()
```

### Creating Videos

```python
import imageio

frames = []

for step in range(500):
    actions = {agent_id: env.action_space(agent_id).sample()
               for agent_id in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    frames.append(env.render(mode='rgb_array'))

# Save as GIF
imageio.mimsave('neural_mmo.gif', frames, fps=10)
```

---

## Training with Reinforcement Learning

### Using Stable Baselines 3

```python
from stable_baselines3 import PPO
from nmmo.core.env import Env
import nmmo

# Wrap for single-agent training
config = nmmo.config.Default()
config.PLAYERS = [nmmo.Agent] * 1  # Single agent for simplicity

env = Env(config)

# Train PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("neuralmmo_ppo")

# Test trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

**Note**: Multi-agent training requires more advanced setups (see research papers).

---

## Common Issues

### Issue: "Module not found: vec_noise"

**Solution**:
```bash
pip install vec-noise
```

### Issue: Slow rendering

**Solution**: Rendering is CPU-intensive. Use headless mode for training:
```python
config.RENDER = False
```

### Issue: Out of memory with many agents

**Solution**: Reduce number of agents or map size:
```python
config.PLAYERS = [nmmo.Agent] * 16  # Reduce from 100+ to 16
config.MAP_SIZE = 64  # Reduce from 128
```

---

## Next Steps

1. **Run the starter agent** (`starter_agent.py`)
2. **Read the concepts guide** (`concepts.md`) to understand RL fundamentals
3. **Experiment with custom agents** - implement smarter strategies
4. **Join the Discord** for community support
5. **Read research papers** to learn about advanced techniques

---

## Resources

- **Official Docs**: https://neuralmmo.github.io/
- **GitHub**: https://github.com/NeuralMMO/environment
- **Discord**: https://discord.gg/BkMmFUC
- **Paper**: https://arxiv.org/abs/1903.00784

---

## Troubleshooting Commands

```bash
# Check installation
python -c "import nmmo; print(nmmo.__version__)"

# List installed packages
pip list | grep nmmo

# Reinstall from scratch
pip uninstall nmmo
pip install --no-cache-dir nmmo

# Install development version
git clone https://github.com/NeuralMMO/environment.git
cd environment
pip install -e ".[dev]"
```

Happy coding! 🎮🤖
