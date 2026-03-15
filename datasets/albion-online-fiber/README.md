---
license: cc-by-nc-nd-4.0
task_categories:
- object-detection
language:
- en
tags:
- computer-vision
- yolov8
- albion-online
- game-automation
- fiber-detection
pretty_name: Albion Online Fiber Detection Dataset
size_categories:
- n<1K
---

# Albion Online Fiber Detection Dataset

A custom dataset for training object detection models to identify fiber resources in Albion Online. This dataset contains annotated screenshots with various types of fiber nodes including cotton, flax, hemp, and their rarity variants.

## Dataset Details

### Dataset Description

This dataset contains labeled screenshots from Albion Online focusing on fiber gathering resources. The images are annotated with bounding boxes for 10 different classes of fiber resources.

- **Created by:** leeboykt
- **Language:** English (UI elements)
- **License:** CC-BY-NC-ND-4.0
- **Model trained on this dataset:** [albion-online-fiber-detection](https://huggingface.co/leeboykt/albion-online-fiber-detection)

### Dataset Structure

```
albion-online-fiber/
├── data/
│   ├── train/
│   │   ├── images/    # 12 training images
│   │   └── labels/    # 12 YOLO format annotations
│   ├── val/
│   │   ├── images/    # 3 validation images
│   │   └── labels/    # 3 YOLO format annotations
│   ├── class_names.json  # Class definitions
│   └── data.yaml         # YOLOv8 configuration
└── README.md
```

### Dataset Statistics

- **Total Images:** 15 (12 train + 3 validation)
- **Image Format:** PNG
- **Annotation Format:** YOLO format (normalized coordinates)
- **Image Size:** Variable (game screenshots)
- **Split Ratio:** 80% train / 20% validation

### Classes

The dataset includes 10 classes of fiber resources:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Exceptional Hemp | Highest rarity hemp node |
| 1 | Exceptional Skyflower | Highest rarity skyflower |
| 2 | Rare Skyflower | Rare quality skyflower |
| 3 | Rare Hemp | Rare quality hemp |
| 4 | Skyflower | Common skyflower |
| 5 | Uncommon Hemp | Uncommon quality hemp |
| 6 | Uncommon Skyflower | Uncommon quality skyflower |
| 7 | cotton | Basic cotton fiber |
| 8 | flax | Basic flax fiber |
| 9 | hemp | Basic hemp fiber |

### Label Format

Annotations are in YOLO format (one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1] range.

**Example:**
```
9 0.4099522059303691 0.40104634329588973 0.06255228443004895 0.09406705637304306
7 0.5123456789012345 0.6234567890123456 0.08123456789012345 0.10234567890123456
```

## Usage

### Loading with Ultralytics YOLOv8

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

# Download dataset configuration
config_path = hf_hub_download(
    repo_id="leeboykt/albion-online-fiber",
    filename="data/data.yaml",
    repo_type="dataset"
)

# Train a model
model = YOLO('yolov8n.pt')  # Load pretrained model
results = model.train(
    data=config_path,
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Loading with Python

```python
from huggingface_hub import snapshot_download
import os

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="leeboykt/albion-online-fiber",
    repo_type="dataset"
)

# Access files
train_images = os.path.join(dataset_path, "data/train/images")
train_labels = os.path.join(dataset_path, "data/train/labels")
val_images = os.path.join(dataset_path, "data/val/images")
val_labels = os.path.join(dataset_path, "data/val/labels")
```

### Using with Datasets Library

```python
from datasets import load_dataset

dataset = load_dataset("leeboykt/albion-online-fiber")
```

## Training Example

Complete training example with YOLOv8:

```python
from ultralytics import YOLO
from huggingface_hub import snapshot_download

# Download dataset
dataset_path = snapshot_download(
    repo_id="leeboykt/albion-online-fiber",
    repo_type="dataset"
)

# Initialize model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data=f"{dataset_path}/data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name='albion_fiber_detection',
    project='runs/detect'
)

# Validate
metrics = model.val()

# Export
model.export(format='onnx')
```

## Dataset Creation

### Annotation Process

- Images were captured from Albion Online gameplay
- Annotations created using Label Studio
- Bounding boxes manually drawn around fiber resources
- Quality control performed to ensure accuracy

### Data Collection

- **Source:** Albion Online game screenshots
- **Collection Period:** 2023
- **Collection Method:** Manual screenshot capture during gameplay
- **Resolution:** Native game resolution

## Limitations

- Small dataset size (15 images) - suitable for fine-tuning only
- Images captured from specific game settings and zones
- May not generalize to different:
  - Screen resolutions
  - Graphics settings
  - Game zones/biomes
  - UI configurations
- Imbalanced class distribution (some resource types more common)

## Ethical Considerations

- This dataset is for educational and research purposes
- Use of automation in Albion Online may violate the game's Terms of Service
- Users should be aware of and comply with game policies

## Citation

```bibtex
@misc{albion-fiber-dataset,
  author = {leeboykt},
  title = {Albion Online Fiber Detection Dataset},
  year = {2023},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/leeboykt/albion-online-fiber}}
}
```

## Related Resources

- **Trained Model:** [albion-online-fiber-detection](https://huggingface.co/leeboykt/albion-online-fiber-detection)
- **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Game:** [Albion Online](https://albiononline.com/)

## License

This dataset is released under the CC-BY-NC-ND-4.0 license, meaning:
- **BY (Attribution):** You must give appropriate credit
- **NC (NonCommercial):** Not for commercial use
- **ND (NoDerivatives):** No modifications or derivatives

## Disclaimer

This dataset is for educational and research purposes only. The use of automation tools in Albion Online may violate the game's Terms of Service. Use at your own risk.
