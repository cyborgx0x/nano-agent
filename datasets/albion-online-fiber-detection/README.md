---
license: cc-by-nc-nd-4.0
language:
- en
tags:
- yolov8
- object-detection
- game-automation
- albion-online
- computer-vision
- pytorch
- ultralytics
library_name: ultralytics
pipeline_tag: object-detection
base_model: yolov8n
datasets:
- leeboykt/albion-online-fiber
metrics:
- precision
- recall
- mAP
---

# Albion Online Fiber Detection Model

A custom-trained YOLOv8 model for detecting fiber resources (cotton, flax, hemp) in Albion Online. This model is designed for game automation and resource gathering assistance.

## Model Details

- **Model Type**: YOLOv8 (Ultralytics)
- **Task**: Object Detection
- **Training Data**: [Albion Online Fiber Detection Dataset](https://huggingface.co/datasets/leeboykt/albion-online-fiber)
- **Classes**: 10 fiber resource types (cotton, flax, hemp, skyflower, and rarity variants)
- **Model Size**: 22 MB
- **Framework**: PyTorch
- **Training Images**: 12 images
- **Validation Images**: 3 images

## Installation

```bash
pip install ultralytics huggingface-hub
```

## Usage

### Quick Start - Load from HuggingFace Hub

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download model from HuggingFace
model_path = hf_hub_download(
    repo_id="leeboykt/albion-online-fiber-detection",
    filename="model.pt"
)

# Load the model
model = YOLO(model_path)

# Run inference on an image
results = model.predict('screenshot.jpg')

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xyxy}")
```

### Advanced Usage - Screen Capture Detection

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import pyautogui
from PIL import Image

# Load model
model_path = hf_hub_download(
    repo_id="leeboykt/albion-online-fiber-detection",
    filename="model.pt"
)
model = YOLO(model_path)

# Capture screen and detect resources
screenshot = pyautogui.screenshot()
results = model.predict(screenshot)

# Get detections
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())

        print(f"Detected resource at ({x1}, {y1}) with confidence {confidence:.2f}")
```

### Using with Real-time Detection

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

# Load model
model_path = hf_hub_download(
    repo_id="leeboykt/albion-online-fiber-detection",
    filename="model.pt"
)
model = YOLO(model_path)

# Real-time detection with streaming
model.predict(source=0, show=True)  # Use webcam
# or
model.predict(source='video.mp4', show=True)  # Use video file
```

## Model Output

The model returns detection results with:
- **Bounding boxes**: `(x1, y1, x2, y2)` coordinates
- **Confidence scores**: Detection confidence (0-1)
- **Class IDs**: Resource type identifiers
- **Class names**: Human-readable resource names

## Use Cases

- Automated resource gathering in Albion Online
- Game bot development
- Computer vision research for game automation
- Educational purposes for object detection

## Limitations

- Trained specifically for Albion Online graphics
- Performance may vary with different screen resolutions
- May not work well with game UI updates or visual changes
- Intended for educational and research purposes

## Citation

If you use this model, please cite:

```bibtex
@misc{albion-fiber-detection,
  author = {leeboykt},
  title = {Albion Online Fiber Detection Model},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/leeboykt/albion-online-fiber-detection}}
}
```

## License

This model is released under CC-BY-NC-ND-4.0 license.

## Disclaimer

This model is for educational and research purposes only. Use of automation tools may violate the Terms of Service of Albion Online. Use at your own risk.
