# Deep Reinforcement Learning Guide

Complete guide for training RL agents to play Albion Online gathering using vision sensors.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Algorithms](#algorithms)
- [Training](#training)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

**⚠️  IMPORTANT**: The RL environment can run in two modes:

### Option 1: Simulated Environment (Recommended for Starting)

**No game required! Perfect for:**
- Testing algorithms
- Learning RL concepts
- Quick iterations
- Development

```bash
# Verify setup
python verify_setup.py

# Start training immediately
python train_rl.py --algorithm dqn --timesteps 10000
```

**Requirements:**
- Python 3.10+
- Dependencies: `pip install -r requirements.txt requirements-training.txt`
- That's it!

### Option 2: Real Game Environment (For Production)

**Requires actual Albion Online running:**
- Game must be launched and visible
- Character in gathering zone
- `model.pt` file (YOLO model)
- Full setup guide: **[SETUP.md](SETUP.md)**

```bash
# Verify all prerequisites
python verify_setup.py

# With game running:
python train_rl.py --algorithm dqn --timesteps 50000
```

**Quick Check:**
```bash
# Check if ready to train
python verify_setup.py

# Test environment
python game_env.py
```

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

## Overview

This RL system trains an agent to autonomously gather resources in Albion Online using:
- **Sensors**: YOLO object detection + OCR text recognition
- **Actions**: Click on resources (cotton/flax/hemp) or wait
- **Rewards**: Based on successful gathering and efficiency

### Key Features

✅ Multiple RL algorithms (DQN, PPO, A2C)
✅ Gym-compatible environment wrapper
✅ Multi-sensor fusion (vision + text)
✅ Automated training with checkpoints
✅ TensorBoard integration
✅ Simple and advanced environment variants

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RL TRAINING LOOP                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. OBSERVATION (Sensors)                               │
│     ├─ YOLO Detection → Bounding boxes, classes        │
│     ├─ OCR Reader → Inventory count (X/9)              │
│     └─ State Vector → Flatten and normalize            │
│                                                           │
│  2. AGENT (Neural Network)                              │
│     ├─ Input: State vector (5 or 63 dims)              │
│     ├─ Network: MLP (64x64 hidden layers)              │
│     └─ Output: Action probabilities or Q-values        │
│                                                           │
│  3. ACTION EXECUTION                                    │
│     ├─ 0: Click cotton                                  │
│     ├─ 1: Click flax                                    │
│     ├─ 2: Click hemp                                    │
│     └─ 3: Wait                                          │
│                                                           │
│  4. REWARD COMPUTATION                                  │
│     ├─ +5 per item gathered                            │
│     ├─ -0.1 per time step                              │
│     ├─ +50 for completing episode                      │
│     └─ -0.5 for failed actions                         │
│                                                           │
│  5. LEARNING                                            │
│     └─ Update policy based on (state, action, reward)  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Train with Default Settings

```bash
# Simple DQN training
python train_rl.py --algorithm dqn --timesteps 10000

# Or use config file
python train_rl.py --config config/rl_config.yaml
```

### 2. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=logs/rl

# Open browser to http://localhost:6006
```

### 3. Evaluate Trained Agent

```bash
# Evaluate over 10 episodes
python train_rl.py \
    --mode eval \
    --load-model models/rl/exp_dqn_20231023_120000/dqn_final.zip \
    --n-episodes 10
```

### 4. Watch Agent Play

```bash
# Play one episode
python train_rl.py \
    --mode play \
    --load-model models/rl/exp_dqn_20231023_120000/dqn_final.zip
```

## Environment Details

### Observation Space

#### Simplified Environment (5 features)
Fast training, recommended for beginners:

```python
[
    cotton_count,      # Number of cotton resources visible
    flax_count,        # Number of flax resources visible
    hemp_count,        # Number of hemp resources visible
    inventory_count,   # Current inventory (0-9)
    nearest_distance   # Distance to nearest resource (normalized)
]
```

#### Full Environment (63 features)
More information, potentially better performance:

```python
[
    # Detection features (60 dims)
    # Up to 10 detections x 6 features each:
    # [x, y, width, height, class, confidence] x 10

    # Player state (3 dims)
    inventory_count,   # Normalized (0-1)
    last_action,       # Normalized (0-1)
    current_step       # Normalized (0-1)
]
```

### Action Space

Discrete actions (4 total):

| Action | Description | Strategy |
|--------|-------------|----------|
| 0 | Click cotton | Target T2 fiber (low value) |
| 1 | Click flax | Target T3 fiber (medium value) |
| 2 | Click hemp | Target T4 fiber (high value) |
| 3 | Wait | Do nothing for 0.5s |

### Reward Function

```python
reward = base_reward + gather_reward + completion_bonus

base_reward = -0.1              # Time penalty
gather_reward = +5 * items      # Per item gathered
completion_bonus = +50          # If inventory full (9/9)
failed_action_penalty = -0.5    # If action fails
```

### Episode Termination

- **Success**: Inventory full (9/9 items)
- **Truncation**: Max steps reached (default 500)

## Algorithms

### DQN (Deep Q-Network)

**Best for**: Discrete action spaces, sample efficiency

```bash
python train_rl.py \
    --algorithm dqn \
    --timesteps 50000 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --buffer-size 10000
```

**Pros**:
- Experience replay (efficient use of data)
- Stable with good hyperparameters
- Good exploration strategy

**Cons**:
- Requires large replay buffer
- Slower than on-policy methods

### PPO (Proximal Policy Optimization)

**Best for**: Stability, continuous actions (if needed)

```bash
python train_rl.py \
    --algorithm ppo \
    --timesteps 100000 \
    --learning-rate 3e-4 \
    --batch-size 64
```

**Pros**:
- Very stable
- Widely used and well-tested
- Good for continuous actions

**Cons**:
- Requires more samples
- Slower training

### A2C (Advantage Actor-Critic)

**Best for**: Speed, quick iterations

```bash
python train_rl.py \
    --algorithm a2c \
    --timesteps 50000 \
    --learning-rate 7e-4
```

**Pros**:
- Fast training
- Good for quick experiments
- Lower memory usage

**Cons**:
- Less stable than PPO
- Higher variance

## Training

### Basic Training

```bash
# Train DQN for 50k steps
python train_rl.py \
    --algorithm dqn \
    --timesteps 50000 \
    --name my_experiment
```

### Training with Custom Config

Create `my_config.yaml`:
```yaml
algorithm: ppo
timesteps: 100000
learning_rate: 0.0003
batch_size: 64
env_type: simplified
```

Then train:
```bash
python train_rl.py --config my_config.yaml
```

### Hyperparameter Tuning

```bash
# Low learning rate for stable training
python train_rl.py --learning-rate 1e-5

# High learning rate for fast convergence
python train_rl.py --learning-rate 1e-3

# Large batch for stability
python train_rl.py --batch-size 128

# Small batch for faster updates
python train_rl.py --batch-size 16
```

### Resume Training

```bash
# Load checkpoint and continue
python train_rl.py \
    --load-model models/rl/exp/dqn_agent_25000_steps.zip \
    --timesteps 50000
```

### Training with Docker

```bash
# Start training container
docker-compose run --rm training bash

# Inside container
python train_rl.py --algorithm dqn --timesteps 50000

# Monitor with TensorBoard
docker-compose up -d tensorboard
# http://localhost:6006
```

## Evaluation

### Evaluate Model

```bash
# Evaluate over 20 episodes
python train_rl.py \
    --mode eval \
    --load-model models/rl/best_model.zip \
    --n-episodes 20
```

### Play Single Episode

```bash
# Watch agent play with verbose output
python train_rl.py \
    --mode play \
    --load-model models/rl/best_model.zip \
    --verbose 2
```

### Programmatic Evaluation

```python
from rl_agent import create_agent

# Create and load agent
agent = create_agent(algorithm='dqn', env_type='simplified')
agent.load('models/rl/best_model.zip')

# Evaluate
mean_reward, std_reward = agent.evaluate(n_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Play episode
reward, info = agent.play_episode(deterministic=True)
print(f"Items gathered: {info['total_gathered']}")
```

## Advanced Usage

### Custom Environment

```python
from game_env import SimplifiedAlbionEnv
import gymnasium as gym

# Create custom environment
class MyAlbionEnv(SimplifiedAlbionEnv):
    def _compute_reward(self, action, action_success, screenshot):
        # Custom reward logic
        reward = super()._compute_reward(action, action_success, screenshot)

        # Add custom bonuses
        if action == 2:  # Prioritize hemp
            reward += 1.0

        return reward

# Use in training
env = MyAlbionEnv()
```

### Custom Network Architecture

```python
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
import torch.nn as nn

# Custom policy network
policy_kwargs = dict(
    net_arch=[128, 128, 64],  # 3 layers: 128->128->64
    activation_fn=nn.ReLU,
)

# Create agent with custom network
model = DQN(
    MlpPolicy,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    verbose=1
)

model.learn(total_timesteps=50000)
```

### Multi-Processing

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from game_env import SimplifiedAlbionEnv

# Create vectorized environment (4 parallel instances)
def make_env():
    def _init():
        return SimplifiedAlbionEnv()
    return _init

env = SubprocVecEnv([make_env() for _ in range(4)])

# Train with parallel environments
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Curriculum Learning

```python
# Stage 1: Easy task (only cotton, short episodes)
agent.env.max_steps = 100
agent.train(total_timesteps=10000)

# Stage 2: Medium task (cotton + flax)
agent.env.max_steps = 200
agent.train(total_timesteps=20000)

# Stage 3: Full task (all resources)
agent.env.max_steps = 500
agent.train(total_timesteps=50000)
```

## Monitoring and Visualization

### TensorBoard Metrics

Training metrics logged:
- `rollout/ep_rew_mean`: Mean episode reward
- `rollout/ep_len_mean`: Mean episode length
- `train/loss`: Training loss
- `gathering/total_gathered`: Items gathered per episode
- `gathering/inventory_final`: Final inventory count

View with:
```bash
tensorboard --logdir=logs/rl
```

### Custom Logging

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomLogger(BaseCallback):
    def _on_step(self):
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            self.logger.record('custom/success_rate', info['total_gathered'] / 9.0)
        return True

# Use in training
agent.train(total_timesteps=10000, callback=CustomLogger())
```

## Troubleshooting

### Agent Not Learning

**Symptoms**: Reward not improving, random behavior

**Solutions**:
```bash
# 1. Increase training time
python train_rl.py --timesteps 100000

# 2. Adjust learning rate
python train_rl.py --learning-rate 1e-3  # Higher for faster learning

# 3. Use simpler environment
python train_rl.py --env-type simplified

# 4. Check reward function
# Edit game_env.py to increase gather rewards
```

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
```bash
# 1. Use faster algorithm
python train_rl.py --algorithm a2c

# 2. Use simplified environment
python train_rl.py --env-type simplified

# 3. Reduce evaluation frequency
python train_rl.py --eval-freq 10000

# 4. Reduce checkpoint frequency
python train_rl.py --checkpoint-freq 10000
```

### Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
```bash
# 1. Reduce batch size
python train_rl.py --batch-size 16

# 2. Reduce buffer size
python train_rl.py --buffer-size 5000

# 3. Use CPU
python train_rl.py --device cpu
```

### Unstable Training

**Symptoms**: Reward fluctuating wildly

**Solutions**:
```bash
# 1. Use PPO (more stable)
python train_rl.py --algorithm ppo

# 2. Lower learning rate
python train_rl.py --learning-rate 1e-5

# 3. Increase batch size
python train_rl.py --batch-size 64

# 4. Add entropy regularization
# Edit config/rl_config.yaml, increase ent_coef
```

### Environment Errors

**Symptoms**: Detection failures, OCR errors

**Solutions**:
```python
# 1. Check YOLO model
from fiber_detection import FiberDetection
detector = FiberDetection('model.pt')
# Test detection manually

# 2. Verify OCR
from gather_state import GatherState
reader = GatherState()
# Test OCR manually

# 3. Add error handling in environment
# Edit game_env.py to be more robust
```

## Best Practices

### Training Strategy

1. **Start Simple**:
   ```bash
   python train_rl.py --algorithm dqn --env-type simplified --timesteps 10000
   ```

2. **Iterate Quickly**:
   - Train for 10k steps
   - Evaluate performance
   - Adjust hyperparameters
   - Repeat

3. **Scale Up**:
   ```bash
   # Once it works, scale up
   python train_rl.py --timesteps 100000 --env-type full
   ```

### Hyperparameter Recommendations

| Task | Algorithm | Learning Rate | Timesteps |
|------|-----------|---------------|-----------|
| Quick test | DQN | 1e-3 | 10,000 |
| Balanced | DQN | 1e-4 | 50,000 |
| Best performance | PPO | 3e-4 | 100,000 |
| Fast iterations | A2C | 7e-4 | 25,000 |

### Reward Shaping Tips

- **Sparse rewards**: Only reward on success (harder to learn)
- **Dense rewards**: Reward every small step (easier to learn)
- **Shaped rewards**: Guide agent toward goal

Example:
```python
# Dense reward shaping
reward = 0
reward += 5 * items_gathered          # Main objective
reward -= 0.1                         # Time penalty
reward += 1.0 / (distance_to_target + 1)  # Distance shaping
reward += 50 if inventory_full else 0  # Completion bonus
```

## Next Steps

1. **Test Environment**: Run `python game_env.py` to test
2. **Quick Training**: Train for 10k steps to verify setup
3. **Hyperparameter Tuning**: Experiment with learning rates
4. **Evaluate**: Test trained agent's performance
5. **Deploy**: Integrate with main bot pipeline

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep RL Course](https://huggingface.co/learn/deep-rl-course)

## Examples

### Example 1: Train DQN Agent

```bash
# Train for 50k steps
python train_rl.py \
    --algorithm dqn \
    --timesteps 50000 \
    --learning-rate 0.0001 \
    --batch-size 32 \
    --name dqn_baseline
```

### Example 2: Train PPO Agent

```bash
# Train with PPO (more stable)
python train_rl.py \
    --algorithm ppo \
    --timesteps 100000 \
    --learning-rate 0.0003 \
    --name ppo_stable
```

### Example 3: Evaluate and Compare

```bash
# Evaluate DQN
python train_rl.py --mode eval --load-model models/rl/dqn_final.zip --n-episodes 20

# Evaluate PPO
python train_rl.py --mode eval --load-model models/rl/ppo_final.zip --n-episodes 20

# Compare results
```

---

Happy training! 🚀
