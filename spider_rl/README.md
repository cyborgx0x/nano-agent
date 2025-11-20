# Spider Robot Reinforcement Learning - Phase 1: Standing & Balancing

A complete, runnable reinforcement learning training pipeline for an 8-legged spider robot using NVIDIA Isaac Lab and PPO (Proximal Policy Optimization).

## Overview

This project implements **Phase 1** of spider robot locomotion: learning to stand upright and maintain balance while recovering from random external pushes.

### Robot Specifications
- **8 legs** with **3 actuated joints each** (coxa yaw, femur pitch, tibia pitch)
- **24 degrees of freedom (DoF)** total
- Target torso height: ~0.18 m
- Simulated with realistic physics, contacts, and domain randomization

### Task Description
The robot must:
- Keep its torso upright (minimize tilt)
- Maintain target height (0.18 m ± tolerance)
- Recover from random external pushes (10-80 N every 3-5 seconds)
- Stay balanced without falling

## Features

### Environment (`spider_standing_env.py`)
- **Observation Space** (153 dims):
  - Joint positions & velocities × 3-step history (144 dims)
  - IMU angular velocity (3 dims)
  - IMU linear acceleration (3 dims)
  - Torso orientation error (roll, pitch, yaw) (3 dims)

- **Action Space** (24 dims):
  - Target joint positions (PD controller in simulation)

- **Reward Function**:
  ```
  R = exp(-3 × tilt²) - 5 × |height_error| + 1.0 - 0.001 × action_rate - 0.0001 × joint_vel²
  ```

- **Domain Randomization**:
  - Friction: 0.5–1.5
  - Mass: ±40%
  - Joint damping: ±50%
  - Random pushes: every 3–5 sec, 10–80 N, random direction
  - Actuator delay: 10–30 ms

- **Termination Conditions**:
  - Torso height < 0.1 m
  - Tilt > 60°
  - Episode timeout (10 seconds)

### Training Algorithm
- **PPO** (Proximal Policy Optimization) via RSL-RL
- **4096+ parallel environments** (configurable)
- Actor-Critic MLP: [512, 256, 128] hidden layers
- Checkpoints saved every 50 iterations
- Tensorboard logging with custom metrics

### Metrics Logged
- Standing success rate (height ± 10% and tilt < 10°)
- Average torso height
- Average tilt (degrees)
- Value loss, surrogate loss
- Policy learning rate and action std

## Directory Structure

```
spider_rl/
├── assets/
│   └── urdf/
│       └── spider.urdf          # 8-legged spider robot URDF
├── envs/
│   ├── __init__.py
│   └── spider_standing_env.py   # Isaac Lab environment
├── config/
│   └── spider_ppo_config.yaml   # PPO training configuration
├── scripts/
│   └── (evaluation scripts - optional)
├── logs/                         # Training logs and checkpoints
│   ├── tensorboard/
│   └── checkpoint_*.pt
├── train.py                      # Main training script
├── requirements.txt
└── README.md                     # This file
```

## Installation

### Prerequisites
1. **NVIDIA Isaac Lab** (or Isaac Gym Legacy)
   - Follow installation guide: https://isaac-sim.github.io/IsaacLab/
   - Requires NVIDIA GPU with CUDA support
   - Ubuntu 20.04/22.04 recommended

2. **Python 3.8+**

### Setup

1. **Clone repository** (if not already done):
   ```bash
   cd /path/to/nano-agent/spider_rl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Isaac Lab installation**:
   ```bash
   python -c "import omni.isaac.lab; print('Isaac Lab installed successfully')"
   ```

## Usage

### Training from Scratch

```bash
# Basic training (headless mode, 4096 environments)
python train.py --headless

# Custom number of environments
python train.py --headless --num_envs 8192

# With visualization (slower, useful for debugging)
python train.py --num_envs 512

# Specify device
python train.py --headless --device cuda:0

# Custom configuration
python train.py --config config/spider_ppo_config.yaml --headless
```

### Resume Training from Checkpoint

```bash
python train.py --checkpoint logs/checkpoint_500.pt --headless
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config | `config/spider_ppo_config.yaml` |
| `--headless` | Run without GUI (faster) | `False` |
| `--num_envs` | Number of parallel environments | `4096` (from config) |
| `--device` | CUDA device | `cuda:0` |
| `--max_iterations` | Maximum training iterations | `5000` (from config) |
| `--checkpoint` | Resume from checkpoint path | `None` |
| `--seed` | Random seed | `42` (from config) |

### Monitoring Training

#### Tensorboard
```bash
tensorboard --logdir spider_rl/logs/tensorboard
```
Access at: http://localhost:6006

#### Console Output
Training metrics are printed every 10 iterations by default:
```
[Iteration 100/5000]
  Value Loss: 0.0234
  Surrogate Loss: -0.0012
  Action Std: 0.8500
  standing_success_rate: 0.6543
  avg_torso_height: 0.1782
  avg_tilt_deg: 8.32
```

## Configuration

Edit `config/spider_ppo_config.yaml` to customize:

### Key Parameters
- **num_envs**: Number of parallel environments (more = faster but more GPU memory)
- **learning_rate**: PPO learning rate (default: 3e-4)
- **num_learning_epochs**: PPO epochs per iteration (default: 5)
- **max_iterations**: Total training iterations (default: 5000)
- **save_interval**: Checkpoint frequency (default: 50)

### Reward Tuning
Adjust weights in `spider_standing_env.py`:
```python
reward_weights = {
    "orientation": 1.0,      # Tilt penalty
    "height": 5.0,           # Height error penalty
    "alive": 1.0,            # Alive bonus
    "action_rate": 0.001,    # Action smoothness
    "joint_vel": 0.0001,     # Joint velocity penalty
}
```

### Domain Randomization
Configure in `spider_ppo_config.yaml`:
```yaml
domain_randomization:
  mass_randomization: 0.4      # ±40%
  friction_range: [0.5, 1.5]
  damping_randomization: 0.5   # ±50%
  push_interval_range: [3.0, 5.0]
  push_force_range: [10.0, 80.0]
```

## Expected Training Time

- **Hardware**: NVIDIA RTX 3090 / A100
- **Environments**: 4096
- **Time per 1000 iterations**: ~30-60 minutes
- **Convergence**: ~1000-2000 iterations for basic standing
- **Full training (5000 iterations)**: ~2-5 hours

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'omni.isaac.lab'**
   - Ensure Isaac Lab is properly installed
   - Activate the correct conda/virtual environment

2. **CUDA out of memory**
   - Reduce `--num_envs` (try 2048 or 1024)
   - Reduce batch size in config

3. **Slow training**
   - Use `--headless` mode
   - Increase `--num_envs` (if GPU memory allows)
   - Ensure CUDA is being used (`--device cuda:0`)

4. **Robot falls immediately**
   - Check initial joint positions in `default_joint_pos`
   - Adjust reward weights (increase `orientation` and `height` weights)
   - Verify URDF is loaded correctly

5. **URDF loading errors**
   - Ensure path in `spider_standing_env.py` is correct
   - Verify URDF file exists and is valid

## Next Steps (Phase 2+)

After successful standing training:
- **Phase 2**: Forward locomotion (walking)
- **Phase 3**: Multi-directional movement and turning
- **Phase 4**: Terrain adaptation (stairs, slopes, obstacles)

## File Descriptions

### `spider.urdf`
Complete URDF description of 8-legged spider robot with:
- 1 torso (base link)
- 8 legs × 3 joints (coxa, femur, tibia) = 24 actuated joints
- Foot contact spheres for ground interaction
- Realistic inertial properties

### `spider_standing_env.py`
Isaac Lab `DirectRLEnv` implementation with:
- Observation computation (proprioception + IMU)
- Reward function
- Termination conditions
- Domain randomization
- External push disturbances
- History buffers for temporal observations

### `train.py`
Main training script that:
- Initializes Isaac Sim
- Creates environment instances
- Sets up PPO algorithm (RSL-RL)
- Runs training loop
- Handles checkpointing and logging

### `spider_ppo_config.yaml`
Complete training configuration:
- PPO hyperparameters
- Network architecture
- Logging settings
- Domain randomization parameters

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spider_rl_2025,
  title={Spider Robot RL Training with Isaac Lab},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nano-agent}
}
```

## License

MIT License - see repository root for details

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy Training! 🕷️🤖**
