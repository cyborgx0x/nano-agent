# YOLOv8 Finetuning Infrastructure

This guide explains how to use the Docker-based infrastructure for finetuning the YOLOv8 model for Albion Online resource detection.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Services](#services)
- [Training Workflow](#training-workflow)
- [Data Preparation](#data-preparation)
- [Monitoring](#monitoring)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: GTX 1060 (6GB VRAM)
  - Recommended: RTX 3060+ (12GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Disk**: 50GB+ free space

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA drivers (compatible with CUDA 11.8)

### Installing NVIDIA Docker Support

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build training
```

### 2. Prepare Your Dataset

```bash
# Create dataset structure
mkdir -p datasets/images/{train,val}
mkdir -p datasets/labels/{train,val}

# Add your images and labels (see Data Preparation section)
```

### 3. Start Training

```bash
# Start training container
docker-compose run --rm training python train.py \
    --model model.pt \
    --data config/dataset.yaml \
    --epochs 100 \
    --batch 16
```

### 4. Monitor Training

```bash
# Start TensorBoard
docker-compose up -d tensorboard

# Open browser to http://localhost:6006
```

## Services

The infrastructure includes several services:

### 1. Training Service

Main service for model training.

```bash
# Interactive shell
docker-compose run --rm training bash

# Direct training
docker-compose run --rm training python train.py --epochs 100
```

### 2. Jupyter Lab

For experimentation and data exploration.

```bash
# Start Jupyter
docker-compose up -d jupyter

# Access at http://localhost:8888
```

### 3. TensorBoard

Real-time training monitoring.

```bash
# Start TensorBoard
docker-compose up -d tensorboard

# Access at http://localhost:6006
```

### 4. MLflow

Experiment tracking and model versioning.

```bash
# Start MLflow
docker-compose up -d mlflow

# Access at http://localhost:5000
```

### 5. Label Studio

Data annotation interface.

```bash
# Start Label Studio
docker-compose up -d label-studio

# Access at http://localhost:8080
```

## Training Workflow

### Basic Training

```bash
# Train from pretrained model
docker-compose run --rm training python train.py \
    --model model.pt \
    --data config/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

### Finetuning Existing Model

```bash
# Finetune with lower learning rate
docker-compose run --rm training python train.py \
    --model model.pt \
    --data config/dataset.yaml \
    --epochs 50 \
    --batch 16 \
    --lr0 0.001 \
    --augment
```

### Training from Scratch

```bash
# Train YOLOv8n from scratch
docker-compose run --rm training python train.py \
    --model yolov8n.pt \
    --data config/dataset.yaml \
    --epochs 300 \
    --batch 32 \
    --lr0 0.01
```

### Resume Training

```bash
# Resume from last checkpoint
docker-compose run --rm training python train.py \
    --resume \
    --project runs/train \
    --name exp
```

### Multi-GPU Training

```bash
# Use multiple GPUs
docker-compose run --rm training python train.py \
    --device 0,1,2,3 \
    --batch 64
```

## Data Preparation

### Directory Structure

```
datasets/
├── images/
│   ├── train/
│   │   ├── cotton_001.jpg
│   │   ├── flax_002.jpg
│   │   └── hemp_003.jpg
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── cotton_001.txt
    │   ├── flax_002.txt
    │   └── hemp_003.txt
    └── val/
        └── ...
```

### Label Format (YOLO)

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]:

```
0 0.5 0.5 0.2 0.3
1 0.7 0.3 0.15 0.25
```

### Annotation Options

#### Option 1: Label Studio (Included)

```bash
# Start Label Studio
docker-compose up -d label-studio

# Access at http://localhost:8080
# Create object detection project
# Import images from datasets/images/train/
# Export in YOLO format
```

#### Option 2: ROBOFLOW

1. Upload images to https://roboflow.com
2. Annotate online
3. Export in YOLOv8 format
4. Download and extract to `datasets/`

#### Option 3: Bot Trainer Submodule

```bash
# Use existing bot_trainer tools
cd bot_trainer
# Follow bot_trainer documentation
```

## Monitoring

### TensorBoard

Real-time training metrics:

```bash
docker-compose up -d tensorboard
# http://localhost:6006
```

Metrics available:
- Training/validation loss
- mAP (mean Average Precision)
- Precision/Recall curves
- Learning rate
- GPU utilization

### MLflow

Experiment tracking:

```bash
# Train with MLflow
docker-compose run --rm training python train.py --mlflow

# Start MLflow UI
docker-compose up -d mlflow
# http://localhost:5000
```

### Jupyter Notebooks

Explore data and results:

```bash
docker-compose up -d jupyter
# http://localhost:8888
```

## Advanced Usage

### Custom Hyperparameters

Edit `config/training_config.yaml`:

```yaml
epochs: 100
batch: 16
lr0: 0.01
augment: true
# ... more parameters
```

Then train:

```bash
docker-compose run --rm training python train.py
```

### Data Augmentation

Enable specific augmentations:

```bash
docker-compose run --rm training python train.py \
    --augment \
    --hsv-s 0.7 \
    --hsv-v 0.4 \
    --degrees 10 \
    --translate 0.2 \
    --scale 0.5 \
    --fliplr 0.5 \
    --mosaic 1.0
```

### Transfer Learning

Use different base models:

```bash
# YOLOv8 small (more accurate, slower)
docker-compose run --rm training python train.py --model yolov8s.pt

# YOLOv8 medium
docker-compose run --rm training python train.py --model yolov8m.pt

# YOLOv8 large
docker-compose run --rm training python train.py --model yolov8l.pt
```

### Export Trained Model

```bash
# Enter container
docker-compose run --rm training bash

# Export to different formats
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
model.export(format='onnx')  # ONNX
model.export(format='engine')  # TensorRT
model.export(format='torchscript')  # TorchScript
"
```

### Validation Only

```bash
# Validate a trained model
docker-compose run --rm training python -c "
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
metrics = model.val(data='config/dataset.yaml')
print(metrics)
"
```

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train.py --batch 8

# Reduce image size
python train.py --imgsz 416

# Use mixed precision (enabled by default)
python train.py --amp
```

### GPU Not Detected

```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify NVIDIA Docker runtime
docker info | grep nvidia

# Check CUDA in training container
docker-compose run --rm training python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Training

```bash
# Increase workers
python train.py --workers 16

# Use smaller validation set
python train.py --val-period 5  # Validate every 5 epochs

# Disable plots
python train.py --plots false
```

### Data Loading Issues

```bash
# Verify dataset
docker-compose run --rm training python -c "
import yaml
with open('config/dataset.yaml') as f:
    data = yaml.safe_load(f)
print(data)
"

# Check dataset integrity
docker-compose run --rm training python -c "
from ultralytics.utils import check_dataset
check_dataset('config/dataset.yaml')
"
```

## Best Practices

### Dataset Quality

- Minimum 100 images per class
- 70/20/10 train/val/test split
- Diverse conditions (lighting, angles, zoom)
- Include difficult cases

### Hyperparameter Tuning

1. Start with defaults
2. Adjust learning rate (0.001-0.01)
3. Tune batch size based on GPU memory
4. Enable augmentation for small datasets
5. Use early stopping (patience=50)

### Model Selection

- **YOLOv8n**: Fastest, good for real-time (current model.pt)
- **YOLOv8s**: Balanced speed/accuracy
- **YOLOv8m**: Better accuracy, slower
- **YOLOv8l**: Best accuracy, slowest

### Experiment Tracking

- Use MLflow or TensorBoard
- Tag experiments with descriptive names
- Save training configs with results
- Version your datasets

## File Structure

```
nano-agent/
├── Dockerfile                  # Training container
├── docker-compose.yml          # Service orchestration
├── train.py                    # Training script
├── config/
│   ├── dataset.yaml           # Dataset config
│   └── training_config.yaml   # Hyperparameters
├── datasets/                   # Training data (gitignored)
├── runs/                       # Training outputs (gitignored)
├── models/                     # Saved models (gitignored)
├── logs/                       # Logs (gitignored)
├── notebooks/                  # Jupyter notebooks
└── mlruns/                     # MLflow tracking
```

## Next Steps

1. Prepare your dataset in `datasets/`
2. Verify dataset with Label Studio
3. Start with small training run to test setup
4. Monitor with TensorBoard
5. Tune hyperparameters based on results
6. Export best model for production use

## Support

For issues or questions:
- Check the troubleshooting section
- Review YOLOv8 documentation: https://docs.ultralytics.com
- Check existing GitHub issues
- Consult bot_trainer submodule documentation
