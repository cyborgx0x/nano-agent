# Quick Start Guide

Get your spider robot training in 5 minutes!

## Step 1: Prerequisites

```bash
# Verify Isaac Lab is installed
python -c "import omni.isaac.lab; print('✓ Isaac Lab ready')"

# Verify CUDA is available
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Install Dependencies

```bash
cd spider_rl
pip install -r requirements.txt
```

## Step 3: Start Training

### Basic Training (Recommended)
```bash
python train.py --headless --num_envs 4096
```

### With Visualization (Slower)
```bash
python train.py --num_envs 512
```

### Quick Test Run
```bash
python train.py --num_envs 128 --max_iterations 100
```

## Step 4: Monitor Progress

### Option A: Tensorboard (Real-time)
```bash
# In another terminal
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```

### Option B: Console Output
Watch the console for metrics every 10 iterations:
```
[Iteration 100/5000]
  standing_success_rate: 0.65  ← Goal: > 0.90
  avg_torso_height: 0.178      ← Goal: ~0.18
  avg_tilt_deg: 8.3            ← Goal: < 10
```

## Step 5: Evaluate Trained Model

```bash
python scripts/evaluate.py --checkpoint logs/checkpoint_1000.pt --num_episodes 10
```

## Expected Results

| Training Stage | Iterations | Success Rate | Description |
|----------------|------------|--------------|-------------|
| Initial | 0-200 | 0-20% | Learning basic balance |
| Learning | 200-1000 | 20-60% | Improving stability |
| Converging | 1000-2000 | 60-85% | Fine-tuning control |
| Trained | 2000+ | 85-95%+ | Robust standing |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--num_envs` to 2048 or 1024 |
| Very slow | Add `--headless` flag |
| Import errors | Check Isaac Lab installation |
| Robot falls | Check URDF loaded correctly |

## Next Steps

1. **Tune rewards**: Edit `envs/spider_standing_env.py` reward weights
2. **Adjust domain randomization**: Edit `config/spider_ppo_config.yaml`
3. **Extend to walking**: Implement Phase 2 (locomotion)
4. **Add terrain**: Use Isaac Lab terrain tools

## Useful Commands

```bash
# Resume from checkpoint
python train.py --checkpoint logs/checkpoint_500.pt --headless

# Custom seed for reproducibility
python train.py --seed 12345 --headless

# Plot training curves
python scripts/plot_training.py --log_dir logs/tensorboard

# Evaluate multiple checkpoints
for i in 500 1000 1500 2000; do
    python scripts/evaluate.py --checkpoint logs/checkpoint_${i}.pt
done
```

## Performance Tips

1. **Use headless mode** for 3-5x speedup
2. **Maximize environments** based on GPU memory:
   - 8GB VRAM: 2048 envs
   - 16GB VRAM: 4096 envs
   - 24GB+ VRAM: 8192+ envs
3. **Use multiple GPUs** (edit config for multi-GPU)
4. **Reduce logging frequency** for slight speedup

## Support

- **Documentation**: See `README.md`
- **Issues**: Check common errors in README Troubleshooting
- **Examples**: Explore `config/` for configuration options

**Happy Training! 🕷️🤖**
