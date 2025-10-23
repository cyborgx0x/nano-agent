# Nano-Agent - Autonomous Game Bot with Deep Reinforcement Learning

Autonomous AI agent for playing Albion Online using computer vision and deep reinforcement learning.

## Features

🤖 **Deep RL Agent** - Learns to play using DQN, PPO, or A2C algorithms
👁️ **Multi-Sensor Fusion** - Combines YOLO object detection + OCR text recognition
🐳 **Docker Infrastructure** - Complete training environment with GPU support
📊 **Experiment Tracking** - TensorBoard, MLflow, and Jupyter integration
🎮 **Game Automation** - Autonomous resource gathering in Albion Online

## Quick Start

```bash
# Clone repository
git clone https://github.com/diopthe20/nanobot
git submodule update --init --recursive

# Train RL agent (Simple method)
python train_rl.py --algorithm dqn --timesteps 50000

# Or use Docker
docker-compose run --rm training python train_rl.py --algorithm dqn --timesteps 50000

# Monitor training
tensorboard --logdir=logs/rl
```

## Documentation

- **[RL_GUIDE.md](RL_GUIDE.md)** - Complete guide for training RL agents
- **[TRAINING.md](TRAINING.md)** - Guide for finetuning YOLO detection model
- **[examples/rl_example.py](examples/rl_example.py)** - Code examples

### TODO

- Develop a deployment strategy
- Integrate RL agent with production bot

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
