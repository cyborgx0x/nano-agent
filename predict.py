from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('model.pt')  # load a custom model
# Predict with the model
model.predict('screen 1920 0 1920 1080', save=True, vid_stride=True)