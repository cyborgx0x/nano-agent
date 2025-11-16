"""
PREDICT MODULE - YOLO Model Testing Script

This is a simple test script to verify the custom YOLO model works.
It demonstrates:
1. Loading the official YOLOv8 nano model (yolov8n.pt)
2. Loading the custom-trained model (model.pt)
3. Running prediction on screen captures

This file is mainly for debugging and model validation.
"""

from ultralytics import YOLO

# Load models
model = YOLO('yolov8n.pt')  # Official YOLOv8 nano model (general object detection)
model = YOLO('model.pt')     # Custom model trained on Albion resources

# Predict with the model
# Note: The string 'screen 1920 0 1920 1080' looks like it's trying to specify
#       a screen region, but this might not be valid YOLO syntax.
#       Normally you'd pass an image file path or NumPy array.
#
# Parameters:
# - save=True: Save prediction results with bounding boxes drawn
# - vid_stride=True: This is for video processing (skip frames)
#
# TODO: Verify this syntax is correct. Might need:
#       model.predict(source='screen', save=True)
model.predict('screen 1920 0 1920 1080', save=True, vid_stride=True)
