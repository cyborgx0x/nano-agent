"""
Test Training Script for Albion Online Fiber Detection
Downloads dataset from HuggingFace and trains YOLOv8 model
"""

from ultralytics import YOLO
from huggingface_hub import snapshot_download
import os
import yaml

print("=" * 70)
print("Albion Online Fiber Detection - Training Test")
print("=" * 70)

# Download dataset from HuggingFace
print("\n[1/5] Downloading dataset from HuggingFace...")
dataset_path = snapshot_download(
    repo_id="leeboykt/albion-online-fiber",
    repo_type="dataset"
)
print(f"✓ Dataset downloaded to: {dataset_path}")

# Create a modified data.yaml with absolute paths
print(f"\n[2/5] Creating corrected data.yaml with absolute paths...")
original_yaml = os.path.join(dataset_path, "data/data.yaml")
corrected_yaml = os.path.join(dataset_path, "data/data_absolute.yaml")

with open(original_yaml, 'r') as f:
    data_config = yaml.safe_load(f)

# Update paths to be absolute
data_config['path'] = os.path.join(dataset_path, "data")
data_config['train'] = 'train/images'  # Relative to 'path'
data_config['val'] = 'val/images'      # Relative to 'path'

with open(corrected_yaml, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✓ Corrected config saved to: {corrected_yaml}")
print(f"  - Dataset root: {data_config['path']}")
print(f"  - Train images: {os.path.join(data_config['path'], data_config['train'])}")
print(f"  - Val images: {os.path.join(data_config['path'], data_config['val'])}")

# Verify paths exist
train_path = os.path.join(data_config['path'], data_config['train'])
val_path = os.path.join(data_config['path'], data_config['val'])
print(f"\n[3/5] Verifying dataset structure...")
print(f"✓ Train path exists: {os.path.exists(train_path)}")
print(f"✓ Val path exists: {os.path.exists(val_path)}")

# Count images
train_images = len([f for f in os.listdir(train_path) if f.endswith('.png')])
val_images = len([f for f in os.listdir(val_path) if f.endswith('.png')])
print(f"✓ Found {train_images} training images")
print(f"✓ Found {val_images} validation images")

# Initialize YOLOv8 model
print(f"\n[4/5] Initializing YOLOv8 nano model...")
model = YOLO('yolov8n.pt')
print("✓ Model initialized")

# Train the model (quick test with 5 epochs on CPU due to CUDA incompatibility)
print(f"\n[5/5] Starting training (5 epochs, CPU mode)...")
print("Note: Using CPU due to CUDA compatibility warning")
print("-" * 70)

results = model.train(
    data=corrected_yaml,
    epochs=5,       # Quick test
    imgsz=640,
    batch=2,        # Small batch for CPU
    name='albion_fiber_test',
    project='runs/detect',
    device='cpu',   # Use CPU instead of GPU due to compatibility
    verbose=True,
    workers=2
)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"Results saved to: runs/detect/albion_fiber_test")
print(f"\nTo view metrics:")
print(f"  - Check: runs/detect/albion_fiber_test/results.png")
print(f"  - Model: runs/detect/albion_fiber_test/weights/best.pt")
