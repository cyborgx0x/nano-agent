# Makefile for YOLOv8 Finetuning Infrastructure

.PHONY: help build up down train jupyter tensorboard mlflow label-studio clean logs shell test

help:
	@echo "YOLOv8 Finetuning Infrastructure"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make train          - Start interactive training container"
	@echo "  make jupyter        - Start Jupyter Lab"
	@echo "  make tensorboard    - Start TensorBoard"
	@echo "  make mlflow         - Start MLflow"
	@echo "  make label-studio   - Start Label Studio"
	@echo "  make shell          - Open shell in training container"
	@echo "  make logs           - Show service logs"
	@echo "  make clean          - Clean up training outputs"
	@echo "  make test           - Test GPU setup"
	@echo ""
	@echo "Training examples:"
	@echo "  make train-quick    - Quick training test (10 epochs)"
	@echo "  make train-full     - Full training (100 epochs)"
	@echo "  make train-finetune - Finetune existing model"

build:
	@echo "Building Docker images..."
	docker-compose build training

up:
	@echo "Starting all services..."
	docker-compose up -d

down:
	@echo "Stopping all services..."
	docker-compose down

train:
	@echo "Starting interactive training container..."
	docker-compose run --rm training bash

jupyter:
	@echo "Starting Jupyter Lab..."
	@echo "Access at: http://localhost:8888"
	docker-compose up -d jupyter

tensorboard:
	@echo "Starting TensorBoard..."
	@echo "Access at: http://localhost:6006"
	docker-compose up -d tensorboard

mlflow:
	@echo "Starting MLflow..."
	@echo "Access at: http://localhost:5000"
	docker-compose up -d mlflow

label-studio:
	@echo "Starting Label Studio..."
	@echo "Access at: http://localhost:8080"
	docker-compose up -d label-studio

shell:
	@echo "Opening shell in training container..."
	docker-compose run --rm training bash

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up training outputs..."
	@read -p "Are you sure? This will delete runs/, models/, and logs/ [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf runs/ models/ logs/ mlruns/; \
		echo "Cleaned up successfully"; \
	fi

test:
	@echo "Testing GPU setup..."
	docker-compose run --rm training python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Training shortcuts
train-quick:
	@echo "Running quick training test (10 epochs)..."
	docker-compose run --rm training python train.py \
		--model model.pt \
		--data config/dataset.yaml \
		--epochs 10 \
		--batch 16 \
		--name quick_test

train-full:
	@echo "Running full training (100 epochs)..."
	docker-compose run --rm training python train.py \
		--model model.pt \
		--data config/dataset.yaml \
		--epochs 100 \
		--batch 16 \
		--name full_training

train-finetune:
	@echo "Finetuning existing model..."
	docker-compose run --rm training python train.py \
		--model model.pt \
		--data config/dataset.yaml \
		--epochs 50 \
		--batch 16 \
		--lr0 0.001 \
		--augment \
		--name finetuned

train-scratch:
	@echo "Training from scratch..."
	docker-compose run --rm training python train.py \
		--model yolov8n.pt \
		--data config/dataset.yaml \
		--epochs 300 \
		--batch 32 \
		--name from_scratch

# Dataset validation
check-dataset:
	@echo "Checking dataset..."
	docker-compose run --rm training python -c "from ultralytics.utils import check_dataset; check_dataset('config/dataset.yaml')"

count-dataset:
	@echo "Counting dataset images..."
	@echo "Train images: $$(find datasets/images/train -type f 2>/dev/null | wc -l)"
	@echo "Val images: $$(find datasets/images/val -type f 2>/dev/null | wc -l)"
	@echo "Train labels: $$(find datasets/labels/train -type f 2>/dev/null | wc -l)"
	@echo "Val labels: $$(find datasets/labels/val -type f 2>/dev/null | wc -l)"
