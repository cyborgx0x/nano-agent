# Dataset Directory

This directory contains training, validation, and test datasets for YOLOv8 finetuning.

## Directory Structure

```
datasets/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images (optional)
└── labels/
    ├── train/          # Training labels (YOLO format)
    ├── val/            # Validation labels
    └── test/           # Test labels (optional)
```

## Data Format

### Images
- Format: JPEG or PNG
- Recommended size: 640x640 (will be resized automatically)
- Naming: Use descriptive names (e.g., `cotton_001.jpg`, `hemp_screenshot_01.png`)

### Labels
- Format: YOLO format (`.txt` files)
- One `.txt` file per image with the same base name
- Each line represents one object: `<class_id> <x_center> <y_center> <width> <height>`
- All coordinates normalized to [0, 1]

Example label file (`cotton_001.txt`):
```
0 0.5 0.5 0.2 0.3
1 0.7 0.3 0.15 0.25
```

### Class IDs
- 0: cotton (T2 fiber)
- 1: flax (T3 fiber)
- 2: hemp (T4 fiber)

## Data Collection

### Option 1: Manual Annotation with Label Studio

1. Start Label Studio:
   ```bash
   docker-compose up label-studio
   ```

2. Access at http://localhost:8080

3. Create a new project with object detection template

4. Import images from `datasets/images/train/`

5. Annotate objects with bounding boxes

6. Export in YOLO format

### Option 2: Using Existing Bot Trainer

The `bot_trainer` submodule contains tools for:
- Screenshot capture from Albion Online
- Label Studio integration
- ROBOFLOW export to YOLO format

### Option 3: Using ROBOFLOW

1. Create account at https://roboflow.com
2. Upload screenshots
3. Annotate online
4. Export in YOLOv8 format
5. Download and extract to this directory

## Data Preparation Tips

### Recommended Dataset Size
- Minimum: 100 images per class
- Good: 500+ images per class
- Excellent: 1000+ images per class

### Data Quality
- Capture screenshots in various:
  - Lighting conditions (day/night)
  - Weather conditions
  - Camera angles
  - Zoom levels
  - Different zones/biomes

- Include difficult cases:
  - Partially occluded resources
  - Resources at edge of screen
  - Multiple resources in frame
  - Resources at different distances

### Train/Val/Test Split
- Training: 70-80% of data
- Validation: 15-20% of data
- Test: 5-10% of data (optional)

Example split for 1000 images:
- train: 800 images
- val: 150 images
- test: 50 images

## Verification

Check your dataset before training:

```python
import os

# Count images and labels
for split in ['train', 'val', 'test']:
    img_dir = f'images/{split}'
    lbl_dir = f'labels/{split}'

    if os.path.exists(img_dir):
        n_images = len(os.listdir(img_dir))
        n_labels = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        print(f'{split}: {n_images} images, {n_labels} labels')
```

## Starting Training

Once you have prepared your dataset:

```bash
# Start training container
docker-compose up -d training

# Enter container
docker-compose exec training bash

# Start training
python train.py --data config/dataset.yaml --epochs 100 --batch 16

# Or use config file
python train.py --config config/training_config.yaml
```

## Monitoring Training

- TensorBoard: http://localhost:6006
- MLflow: http://localhost:5000
- Jupyter: http://localhost:8888

## Data Augmentation

The training pipeline automatically applies augmentation:
- Horizontal flips
- HSV color jittering
- Translation
- Scaling
- Mosaic augmentation

This helps create a more robust model even with limited data.

## Notes

- Keep original screenshots separate from this directory
- Use version control for dataset changes
- Document any preprocessing steps
- Track annotation changes and versions
