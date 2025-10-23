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
import os
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available")


class FiberDetection:
    """
    Wrapper class for YOLO-based fiber resource detection
    Supports both real YOLO model and simulated detection
    """

    def __init__(self, model_path: str = "model.pt", confidence_threshold: float = 0.7):
        """
        Initialize fiber detection

        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

        # Try to load real model if available
        if YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"✓ Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load YOLO model: {e}")
                print("   Falling back to simulated detection")
        else:
            if not YOLO_AVAILABLE:
                print("⚠️  YOLO not available, using simulated detection")
            elif not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                print("   Using simulated detection")

    def detect(self, image) -> list:
        """
        Detect fiber resources in image

        Args:
            image: PIL Image or numpy array

        Returns:
            list: Detection results in standardized format
        """
        if self.model is not None:
            return self._detect_real(image)
        else:
            return self._detect_simulated(image)

    def _detect_real(self, image) -> list:
        """Use real YOLO model for detection"""
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                detections.append({
                    'class': cls,
                    'confidence': conf,
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                })

        return detections

    def _detect_simulated(self, image) -> list:
        """Generate simulated detections for testing"""
        # Convert image to numpy if needed
        if hasattr(image, 'size'):  # PIL Image
            width, height = image.size
        else:  # numpy array
            height, width = image.shape[:2]

        # Generate random detections (2-6 resources)
        num_detections = np.random.randint(2, 7)
        detections = []

        for _ in range(num_detections):
            # Random class (0=cotton, 1=flax, 2=hemp)
            cls = np.random.randint(0, 3)

            # Random position (avoid edges)
            x = np.random.randint(int(width * 0.2), int(width * 0.8))
            y = np.random.randint(int(height * 0.2), int(height * 0.8))

            # Random size (50-150 pixels)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)

            # Random confidence (0.7-0.95)
            conf = np.random.uniform(0.7, 0.95)

            detections.append({
                'class': cls,
                'confidence': conf,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })

        return detections


# Load the custom-trained YOLO model (legacy global variable)
# model.pt is a 22MB file trained on labeled Albion resource screenshots
if YOLO_AVAILABLE and os.path.exists("model.pt"):
    try:
        model = YOLO("model.pt")
    except:
        model = None
else:
    model = None


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
    # Check if model is available
    if model is None:
        raise RuntimeError("YOLO model not loaded. Check that model.pt exists and ultralytics is installed.")

    # Run YOLO inference on the image
    results = model.predict(image)

    # Convert YOLO results to JSON format for easy parsing
    return [json.loads(r.tojson()) for r in results]
