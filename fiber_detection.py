import json

from ultralytics import YOLO

model = YOLO("model.pt")


def get_detection_fiber(image):
    """
    Take an image and return the json object of fiber detection result
    """
    results = model.predict(image)

    return [json.loads(r.tojson()) for r in results]
