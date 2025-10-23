# Dockerfile for YOLOv8 Finetuning Infrastructure
# Supports GPU acceleration with CUDA 11.8

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt
COPY requirements-training.txt /workspace/requirements-training.txt

# Install PyTorch with CUDA 11.8 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
RUN pip3 install -r requirements.txt

# Install training-specific requirements
RUN pip3 install -r requirements-training.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/datasets \
    /workspace/runs \
    /workspace/models \
    /workspace/logs

# Set permissions
RUN chmod -R 777 /workspace

# Expose ports for Tensorboard and MLflow
EXPOSE 6006 5000

# Default command
CMD ["/bin/bash"]
