# nanobot - Autonomous on Playing and interacting with game from screenshot only

```
git clone https://github.com/diopthe20/nanobot
git submodule update --init --recursive

```

### TODO

- Develop a deployment strategy

### Vision model
We will use some vision model to get the information about what we will see in the screen

| name                                                                          | status | description                                                                                                                                                       |
| ----------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [llama-3-vision-alpha](https://huggingface.co/qresearch/llama-3-vision-alpha) |        | projection module trained to add vision capabilties to Llama 3 using SigLIP. built by [@yeswondwerr](https://x.com/yeswondwerr) and [@qtnx_](https://x.com/qtnx_) |

### OCR 

I used EasyOCR for recognize some text in the screen during gathering. 
You can go to https://huggingface.co/spaces/tomofi/EasyOCR to test with EasyOCR
### Event Handling

We take the environment state as an event and send it to event handler


## Object Detection



Label with Label Studio, Export to YOLO Format and then Upload to ROBOFLOW to export to the right format for YOLO

Train with YOLO

=> Predict from the screen stream

Currently this project in development. The current phase is try out new probilities

## Dependency management with uv

This project now uses `uv` with `pyproject.toml` as the single source of truth.

### 1) Install uv (if needed)

```bash
pip install uv
```

### 2) Create/sync environment

CPU-only build:

```bash
uv sync --extra cpu
```

CUDA 11.8 build (recommended starting point for GTX 1060):

```bash
uv sync --extra cu118
```

For live simulator rendering window:

```bash
uv sync --extra cu118 --extra render
```

### 3) Run code with uv

```bash
uv run python main.py
uv run python state_sim/demo.py
```

Run state sim with render dependencies:

```bash
uv run --extra cu118 --extra render python state_sim/demo.py
```

Train navigation policy (map-to-map):

```bash
uv run --extra cu118 python state_sim/train_map_to_map.py
```

Train with Weights & Biases logging (offline mode):

```bash
uv sync --extra cu118 --extra train
uv run --extra cu118 --extra train python state_sim/train_map_to_map.py --episodes 500 --wandb --wandb-mode offline
```

Train with W&B online:

```bash
wandb login
uv run --extra cu118 --extra train python state_sim/train_map_to_map.py --episodes 500 --wandb --wandb-mode online --wandb-project nano-agent-state-sim
```

Train with PPO (recommended) + W&B online:

```bash
wandb login
uv run --extra cu118 --extra train python state_sim/train_map_to_map_ppo.py --episodes 1000 --wandb --wandb-mode online --wandb-project nano-agent-state-sim
```

Evaluate PPO checkpoint:

```bash
uv run --extra cu118 --extra train python state_sim/evaluate_ppo.py --checkpoint runs/state_sim/map_to_map_ppo_best.pt --episodes 200 --max-steps 120
```

Evaluate + export rollout video:

```bash
uv run --extra cu118 --extra render --extra train python state_sim/evaluate_ppo.py --checkpoint runs/state_sim/map_to_map_ppo_best.pt --episodes 60 --save-video --video-path runs/state_sim/eval_episode.mp4
```

### Notes

- PyTorch packages are routed via explicit PyTorch indexes configured in `pyproject.toml` (`cpu` and `cu118`).
- `cpu` and `cu118` extras are mutually exclusive.
- If CUDA build fails on your machine, switch to `uv sync --extra cpu`.
- `state_sim` render will try OpenCV window first; if GUI backend is unavailable (`GUI: NONE`), it will fallback to matplotlib live window.
