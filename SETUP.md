# Game Environment Setup Guide

This guide covers the prerequisites and setup steps for training RL agents.

## ⚠️ Important: Two Environment Modes

The RL system supports **two modes**:

1. **Real Game Environment** (production) - Requires actual Albion Online running
2. **Simulated Environment** (testing) - Mock environment for development

## Real Game Environment Setup

### Prerequisites

#### 1. Albion Online Game
- **Required**: Albion Online installed and running
- **Platform**: Windows (via Wine/Proton on Linux)
- **Account**: Active game account
- **Character**: Must be in-game and able to gather

#### 2. YOLO Detection Model
- **File**: `model.pt` (22MB)
- **Location**: Project root directory
- **Status**: Should already exist in repository
- **Verify**: Run `ls -lh model.pt`

If missing, you need to train it:
```bash
# See TRAINING.md for instructions
python train.py --data config/dataset.yaml --epochs 100
```

#### 3. System Requirements
- **Python**: 3.10+
- **GPU**: NVIDIA GPU recommended (CUDA support)
- **RAM**: 8GB+ (16GB recommended)
- **Screen**: 1920x1080 or higher resolution

#### 4. Python Dependencies

Already in `requirements.txt`:
```bash
# Install if not using Docker
pip install -r requirements.txt
pip install -r requirements-training.txt
```

Key packages:
- `ultralytics` - YOLOv8
- `easyocr` - Text recognition
- `pyautogui` - Screen capture and mouse control
- `stable-baselines3` - RL algorithms
- `gymnasium` - RL environment interface

### Game Setup Steps

#### Step 1: Launch Albion Online

```bash
# Launch the game
# On Windows: Open Albion Online launcher
# On Linux: Use Steam/Wine
```

#### Step 2: Position Character

**Critical Requirements**:
- Character must be in an **open gathering zone**
- Visible resources (cotton/flax/hemp) on screen
- Inventory should be empty or have space
- Character should be mounted (optional but recommended)

**Recommended Locations**:
- **Tier 2-4 zones** (low risk for testing)
- **Steppe Biomes** (good for fiber resources)
- Avoid high-traffic or PvP zones during training

#### Step 3: Window Setup

**Screen Configuration**:
```
┌─────────────────────────────────────┐
│     Albion Online (Windowed)        │
│  Recommended: 1920x1080 fullscreen  │
│  Or: Windowed mode (no borders)     │
└─────────────────────────────────────┘
```

**Important**:
- Use **windowed fullscreen** or **borderless window**
- Game should be **in focus** during training
- No overlapping windows
- Stable frame rate

#### Step 4: Verify Detection

Test YOLO detection:
```bash
# Run detection test
python predict.py

# Or test directly
python -c "
from fiber_detection import get_detection_fiber
import pyautogui
screenshot = pyautogui.screenshot()
detections = get_detection_fiber(screenshot)
print(f'Detected {len(detections)} resources')
"
```

Expected output:
```
Detected 5 resources
[
  {"name": "cotton", "confidence": 0.89, ...},
  {"name": "flax", "confidence": 0.92, ...},
  ...
]
```

#### Step 5: Verify OCR

Test inventory OCR:
```bash
python -c "
from gather_state import GatherState
reader = GatherState()
count = reader.get_gather_count()
print(f'Inventory: {count}')
"
```

Expected output:
```
Inventory: 0/9
```

#### Step 6: Test Environment

```bash
# Quick environment test
python game_env.py
```

This will:
1. Take a screenshot
2. Run YOLO detection
3. Read OCR
4. Execute a test action

#### Step 7: Start Training

```bash
# With game running and character positioned:
python train_rl.py --algorithm dqn --timesteps 10000
```

### Safety Considerations

**Bot Detection**:
- Albion Online **prohibits bots** in Terms of Service
- This is for **educational/research purposes only**
- Use on **test accounts** or **private servers**
- Do not use on live/production accounts

**System Safety**:
- RL agent **will control your mouse**
- Close important applications first
- Consider using **virtual machine**
- Set up **kill switch** (Ctrl+C to stop)

**Training Safety**:
```bash
# Run in a tmux/screen session
tmux new -s rl_training
python train_rl.py --algorithm dqn --timesteps 50000

# Detach: Ctrl+B then D
# Reattach: tmux attach -t rl_training
```

---

## Simulated Environment (No Game Required)

For **testing and development** without the game:

### Quick Start

```bash
# Use simulated environment
python train_rl.py \
    --algorithm dqn \
    --timesteps 10000 \
    --env-type simulated
```

### Features

- **No game required**: Pure simulation
- **Fast training**: No waiting for game actions
- **Deterministic**: Reproducible results
- **Perfect for**:
  - Algorithm testing
  - Hyperparameter tuning
  - Debugging
  - CI/CD pipelines

### How It Works

The simulated environment:
1. Generates **synthetic observations** (random resource positions)
2. Simulates **gathering success** based on action quality
3. Returns **realistic rewards** without actual game interaction
4. Runs **10-100x faster** than real environment

### Example

```python
from game_env import SimulatedAlbionEnv

# Create simulated environment
env = SimulatedAlbionEnv()

# Train as normal
from rl_agent import create_agent
agent = create_agent(algorithm='dqn', env_type='simulated')
agent.train(total_timesteps=50000)
```

---

## Verification Checklist

Before training, verify:

- [ ] Albion Online is running (or using simulated mode)
- [ ] Character is in gathering zone (real mode only)
- [ ] `model.pt` file exists (real mode only)
- [ ] YOLO detection works (`python predict.py`)
- [ ] OCR reading works (inventory count visible)
- [ ] Python dependencies installed
- [ ] GPU available (check with `nvidia-smi`)
- [ ] Environment test passes (`python game_env.py`)

Run automated verification:
```bash
python verify_setup.py
```

---

## Troubleshooting

### "model.pt not found"

```bash
# Check if file exists
ls -lh model.pt

# If missing, train or download:
# Option 1: Train (requires dataset)
python train.py --data config/dataset.yaml --epochs 100

# Option 2: Use pretrained YOLOv8n (less accurate)
# Edit game_env.py to use 'yolov8n.pt' instead
```

### "No detections found"

**Causes**:
- Resources not visible on screen
- Wrong game window/screen
- Model not trained on current graphics settings

**Solutions**:
```bash
# 1. Verify resources are visible
# Take screenshot and check manually

# 2. Test detection on screenshot
python predict.py

# 3. Lower confidence threshold
# Edit game_env.py, change confidence_threshold=0.5
```

### "OCR not reading inventory"

**Causes**:
- Inventory UI not visible
- Wrong screen region
- Different UI language

**Solutions**:
```bash
# 1. Check gather_state.py OCR region
# 2. Make sure gathering progress is visible
# 3. Test OCR directly
python -c "from gather_state import GatherState; print(GatherState().get_gather_count())"
```

### "Mouse not clicking correctly"

**Causes**:
- Screen scaling/DPI issues
- Multiple monitors
- Virtual machine mouse capture

**Solutions**:
- Use native resolution (1920x1080)
- Disable display scaling
- Use primary monitor only
- Adjust `pyautogui` settings in `game_env.py`

### "Training too slow"

**Solutions**:
```bash
# 1. Use simulated environment for testing
python train_rl.py --env-type simulated

# 2. Use faster algorithm
python train_rl.py --algorithm a2c

# 3. Reduce evaluation frequency
python train_rl.py --eval-freq 10000

# 4. Use GPU
python train_rl.py --device cuda
```

### "Game crashes during training"

**Solutions**:
- Use simulated environment for bulk training
- Train for shorter episodes (reduce max_steps)
- Add error handling (environment automatically resets)
- Use virtual machine for isolation

---

## Docker Setup

### With Real Game (Advanced)

Requires X11 forwarding to access game from container:

```bash
# Allow X11 access
xhost +local:docker

# Run container with X11
docker-compose run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    training python train_rl.py
```

### With Simulated Environment (Recommended)

No special setup needed:

```bash
# Just run normally
docker-compose run --rm training \
    python train_rl.py --env-type simulated --timesteps 50000
```

---

## Quick Setup Script

```bash
#!/bin/bash
# setup_rl.sh - Quick setup verification

echo "=== RL Environment Setup Verification ==="

# Check Python
python3 --version || { echo "❌ Python not found"; exit 1; }
echo "✓ Python installed"

# Check dependencies
python3 -c "import torch" || { echo "❌ PyTorch not installed"; exit 1; }
python3 -c "import gymnasium" || { echo "❌ Gymnasium not installed"; exit 1; }
python3 -c "import stable_baselines3" || { echo "❌ Stable-Baselines3 not installed"; exit 1; }
echo "✓ Dependencies installed"

# Check model
if [ -f "model.pt" ]; then
    echo "✓ model.pt found"
else
    echo "⚠️  model.pt not found (required for real environment)"
fi

# Check GPU
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✓ GPU available"
else
    echo "⚠️  No GPU (training will be slower)"
fi

echo ""
echo "=== Setup Status ==="
echo "Ready to train with simulated environment: ✓"
echo "Ready to train with real game: Check model.pt and game running"
echo ""
echo "Next steps:"
echo "  1. Test simulated: python train_rl.py --env-type simulated --timesteps 1000"
echo "  2. Test real: Launch game, then python game_env.py"
```

---

## Environment Comparison

| Feature | Real Environment | Simulated Environment |
|---------|------------------|----------------------|
| **Game Required** | ✓ Yes | ✗ No |
| **Training Speed** | Slow (0.5-1 FPS) | Fast (100+ FPS) |
| **Accuracy** | High (real data) | Medium (synthetic) |
| **Use Case** | Production | Testing/Development |
| **Setup Time** | 30 min | 2 min |
| **Cost** | Game account | Free |
| **Reproducible** | No (game varies) | Yes (deterministic) |

**Recommendation**:
1. Start with **simulated** for algorithm testing
2. Move to **real** for final training
3. Use **simulated** for CI/CD and testing
4. Use **real** for production deployment

---

## Next Steps

### For Development/Testing
```bash
# Use simulated environment
python train_rl.py --env-type simulated --timesteps 50000
```

### For Production
```bash
# 1. Setup game (follow steps above)
# 2. Verify setup
python verify_setup.py

# 3. Train with real environment
python train_rl.py --algorithm dqn --timesteps 100000
```

### For Deployment
```bash
# 1. Train agent (simulated or real)
# 2. Evaluate performance
python train_rl.py --mode eval --load-model models/rl/best.zip

# 3. Deploy to production
python train_rl.py --mode play --load-model models/rl/best.zip
```
