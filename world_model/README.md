# World Model for Game Agent

V-JEPA-based world model for autonomous game playing. Adapted from Meta's V-JEPA architecture.

## Overview

This module implements a world model that:
1. **Learns** visual representations of game states from gameplay videos
2. **Predicts** future states given actions (mouse/keyboard inputs)
3. **Plans** action sequences using Model Predictive Control (MPC)
4. **Controls** the game agent to achieve goals

## Architecture

```
Game Screenshot → Encoder → Latent Representation
                              ↓
                    Latent + Action → Predictor → Next Latent
                              ↓
                    MPC Planner → Best Action Sequence
                              ↓
                    Execute Action in Game
```

## Directory Structure

```
world_model/
├── models/                    # Core V-JEPA models (from vjepa2)
│   ├── vision_transformer.py # Vision encoder (ViT)
│   ├── predictor.py          # Latent predictor
│   └── utils/                # Model utilities
├── datasets/
│   └── gameplay_dataset.py   # Data collection & PyTorch dataset
├── integration/
│   └── game_controller.py    # Connect model to game control
├── game_world_model.py       # Main world model wrapper
├── utils/                    # Utilities from vjepa2
├── masks/                    # Masking for training
└── configs/                  # Configuration files
```

## Quick Start

### 1. Collect Training Data

```python
from world_model.datasets.gameplay_dataset import GameplayCollector

# Start recording gameplay
collector = GameplayCollector(
    save_dir="./gameplay_data",
    fps=10,                    # 10 frames per second
    max_duration=3600,         # 1 hour max
)

print("Recording gameplay. Press Ctrl+C to stop...")
collector.start_recording()
```

This creates:
- `gameplay_data/<session_name>/gameplay.mp4` - Video file
- `gameplay_data/<session_name>/actions.json` - Mouse/keyboard actions
- `gameplay_data/<session_name>/metadata.json` - Session info

**Recommendation:** Collect 10-20 hours of diverse gameplay.

### 2. Train World Model

```python
import torch
from world_model.models.vision_transformer import VisionTransformer
from world_model.models.predictor import VisionTransformerPredictor
from world_model.game_world_model import GameWorldModel
from world_model.datasets.gameplay_dataset import GameplayDataset
from torch.utils.data import DataLoader

# Load pretrained encoder (from V-JEPA)
encoder = VisionTransformer(
    img_size=(224, 224),
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

# Load pretrained predictor
predictor = VisionTransformerPredictor(
    embed_dim=768,
    predictor_embed_dim=384,
    depth=6,
    num_heads=12,
)

# Load pretrained weights (optional)
# checkpoint = torch.load('vjepa_weights.pth')
# encoder.load_state_dict(checkpoint['encoder'])
# predictor.load_state_dict(checkpoint['predictor'])

# Create dataset
dataset = GameplayDataset(
    data_dir="./gameplay_data",
    sequence_length=8,
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop (simplified)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()),
    lr=1e-4
)

for epoch in range(10):
    for frames, actions in dataloader:
        # TODO: Implement training loop
        # 1. Encode frames to latents
        # 2. Predict next latents given actions
        # 3. Compute loss (prediction error)
        # 4. Backprop
        pass
```

**Note:** Full training script coming soon. For now, you can use pretrained V-JEPA weights.

### 3. Use World Model for Control

```python
from world_model.game_world_model import GameWorldModel
from world_model.integration.game_controller import WorldModelAgent
import numpy as np

# Create world model
world_model = GameWorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=196,  # For 224x224 image with patch_size=16
    transform=your_transform,
    device="cuda:0"
)

# Create agent
agent = WorldModelAgent(world_model)

# Define goal (e.g., screenshot of resource location)
goal_image = np.array(...)  # [H, W, 3] RGB image

# Navigate to goal
success = agent.goto(goal_image)
print(f"Goal reached: {success}")
```

## Usage Examples

### Example 1: Data Collection

```bash
# Collect 1 hour of gameplay
python -m world_model.datasets.gameplay_dataset
```

### Example 2: Explore Latent Space

```python
# Encode different game states
from world_model.game_world_model import GameWorldModel

world_model = GameWorldModel(...)

# Encode current state
current_frame = capture_screenshot()
current_latent = world_model.encode(current_frame)

# Encode goal state
goal_frame = load_goal_image()
goal_latent = world_model.encode(goal_frame)

# Compute similarity
distance = torch.mean(torch.abs(current_latent - goal_latent))
print(f"Distance to goal: {distance:.3f}")
```

### Example 3: Action Planning

```python
# Plan action sequence to reach goal
action = world_model.plan_action(
    current_rep=current_latent,
    goal_rep=goal_latent,
    current_mouse_pos=(500, 500)
)

print(f"Planned action: {action}")
# {'mouse_delta': [dx, dy], 'click': bool, 'key': None}
```

## Configuration

Edit `configs/model_config.yaml`:

```yaml
encoder:
  img_size: [224, 224]
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12

predictor:
  embed_dim: 768
  predictor_embed_dim: 384
  depth: 6
  num_heads: 12

mpc:
  rollout: 3           # Look ahead 3 steps
  samples: 200         # Sample 200 action sequences
  topk: 10            # Keep top 10
  cem_steps: 10       # 10 CEM iterations
  max_mouse_move: 100 # Max pixel movement per step
```

## How It Works

### 1. Visual Encoder
- Converts game screenshots to latent representations
- Uses Vision Transformer (ViT) from V-JEPA
- Input: [224, 224, 3] RGB image
- Output: [196, 768] latent tokens (14x14 grid of patches)

### 2. Latent Predictor
- Predicts next latent state given current state + action
- Uses smaller transformer for efficiency
- Input: current_latent [196, 768] + action [3] (dx, dy, click)
- Output: next_latent [196, 768]

### 3. Model Predictive Control (MPC)
- Plans action sequences using Cross-Entropy Method (CEM)
- Simulates multiple action sequences in latent space
- Selects actions that bring agent closest to goal
- Rollout horizon: 3-5 steps ahead

### 4. Action Execution
- Converts planned actions to mouse/keyboard inputs
- Uses pyautogui for game control
- Rate-limited to avoid detection

## Training Tips

### Data Collection
- **Diversity matters:** Collect gameplay from different scenarios
- **Quality over quantity:** 10 hours of focused gameplay > 50 hours of idle
- **Include failures:** World model learns from mistakes too
- **Multiple sessions:** Avoid long continuous recordings (game changes)

### Training
- **Start with pretrained weights:** Use V-JEPA checkpoint
- **Fine-tune on game:** Adapt to your specific game's visuals
- **Self-supervised learning:** No labels needed (predicts future frames)
- **Monitor prediction error:** Should decrease over epochs

### Evaluation
- **Visual inspection:** Do predicted frames look realistic?
- **Action quality:** Can agent reach simple goals?
- **Generalization:** Try goals not seen during training

## Comparison: Heuristics vs World Model

| Aspect | Heuristic (Option D) | World Model (Option B) |
|--------|---------------------|----------------------|
| **Setup time** | 2-4 weeks | 4-8 weeks |
| **Generalization** | Poor (hardcoded rules) | Good (learns patterns) |
| **Maintenance** | High (update rules) | Low (retrain occasionally) |
| **Interpretability** | High (can read code) | Low (latent space) |
| **Performance** | Depends on rules | Depends on training data |
| **Research value** | Low | High (publishable) |

## Roadmap

- [x] Core architecture setup
- [x] Data collection pipeline
- [x] Game controller integration
- [ ] Training script with V-JEPA loss
- [ ] Pretrained checkpoint (download script)
- [ ] Evaluation metrics
- [ ] Visualization tools (latent space, predictions)
- [ ] Multi-game support
- [ ] Online learning (adapt during gameplay)

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use smaller model (embed_dim=384 instead of 768)
- Enable gradient checkpointing

### "World model makes bad predictions"
- Collect more training data
- Ensure data diversity
- Check if game visuals changed (updates/patches)
- Visualize predictions to debug

### "Agent gets stuck in loops"
- Increase rollout horizon
- Add exploration noise
- Use hybrid mode (fall back to heuristics)

## References

- [V-JEPA Paper](https://arxiv.org/abs/2301.08243)
- [Meta AI Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/)

## License

Adapted from Meta's V-JEPA (MIT License).
See individual files for copyright notices.
