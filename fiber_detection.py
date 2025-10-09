"""
FIBER DETECTION MODULE - YOLO Object Detection for Resource Identification

This module uses YOLOv8 (You Only Look Once) to detect fiber resources
in Albion Online screenshots.

YOLO is a real-time object detection neural network that:
- Processes images in a single forward pass (fast!)
- Returns bounding boxes and confidence scores for detected objects
- Has been custom-trained on Albion resource images (model.pt)

The model detects:
- Cotton (tier 2 fiber)
- Flax (tier 3 fiber)
- Hemp (tier 4 fiber)
"""

import json

from ultralytics import YOLO

# Load the custom-trained YOLO model
# model.pt is a 22MB file trained on labeled Albion resource screenshots
model = YOLO("model.pt")


def get_detection_fiber(image):
    """
    Detect fiber resources in a game screenshot using YOLO.

    Process:
    1. Pass image to YOLO model
    2. Model returns detected objects with:
       - Class (cotton/flax/hemp)
       - Confidence score (0-1)
       - Bounding box coordinates (x1, y1, x2, y2)
    3. Convert results to JSON format

    Args:
        image: PIL Image object (screenshot from pyautogui)

    Returns:
        list: Array of detection results in JSON format
              [
                {
                  "name": "cotton",
                  "class": 1,
                  "confidence": 0.91,
                  "box": {"x1": 744, "y1": 404, "x2": 827, "y2": 477}
                },
                ...
              ]
    """
    # Run YOLO inference on the image
    results = model.predict(image)

    # Convert YOLO results to JSON format for easy parsing
    return [json.loads(r.tojson()) for r in results]
