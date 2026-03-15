from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('position.pt')  # load a custom model
# Predict with the model
model.track(source="screen 1920 0 1920 1080", show=True) 
# model.predict('screen 1920 0 1920 1080', save=True, vid_stride=True, show=True)