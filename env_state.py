import json
import easyocr
import time
import pyautogui
import numpy as np

# Initialize the easyocr reader
reader = easyocr.Reader(["en"], gpu=True)

while True:
    # Capture a screenshot
    img = pyautogui.screenshot()

    # Convert the PIL image to a numpy array
    img_np = np.array(img)

    # Perform OCR
    begin = time.time()
    result = reader.readtext(img_np)
    end = time.time()

    # Convert the result to a JSON-serializable format
    serializable_result = []
    for item in result:
        box = item[0]  # Bounding box
        text = item[1]  # Detected text
        confidence = item[2]  # Confidence score
        
        # Convert bounding box coordinates to a list of lists
        box_list = [[int(point[0]), int(point[1])] for point in box]
        
        serializable_result.append({
            "box": box_list,
            "text": text,
            "confidence": float(confidence)
        })

    # Save the result to a JSON file
    file_name = time.time_ns()
    with open(f"state/{file_name}.json", "w") as f:
        json.dump(serializable_result, f)

    # Add a sleep delay to avoid rapid looping (for demonstration purposes)
    time.sleep(5)
