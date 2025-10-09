"""
ENVIRONMENT STATE MODULE - Full OCR Data Logging

This script continuously captures and logs ALL text visible on screen.
Unlike gather_state.py (which only reads gathering UI), this captures
the entire game state for analysis and debugging.

Use cases:
- Debug OCR accuracy
- Collect training data for text detection
- Monitor game state changes over time
- Analyze UI elements for future bot features

Output: JSON files in state/ directory with all detected text and positions
"""

import json
import easyocr
import time
import pyautogui
import numpy as np

# Initialize the easyocr reader
# - Language: English
# - GPU: Enabled for faster OCR processing
reader = easyocr.Reader(["en"], gpu=True)

# ============================================================================
# CONTINUOUS SCREEN MONITORING LOOP
# ============================================================================
while True:
    # Capture full screenshot
    img = pyautogui.screenshot()

    # Convert PIL Image to NumPy array for EasyOCR
    img_np = np.array(img)

    # Perform OCR on entire screen
    begin = time.time()
    result = reader.readtext(img_np)
    end = time.time()
    print(f"OCR took {end - begin:.2f} seconds")

    # Convert the result to a JSON-serializable format
    serializable_result = []
    for item in result:
        box = item[0]  # Bounding box (4 corner points)
        text = item[1]  # Detected text string
        confidence = item[2]  # Confidence score (0-1)

        # Convert bounding box coordinates to list of lists
        # box is like: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        box_list = [[int(point[0]), int(point[1])] for point in box]

        serializable_result.append({
            "box": box_list,
            "text": text,
            "confidence": float(confidence)
        })

    # Save the result to a JSON file with timestamp
    # Filename uses nanoseconds to avoid collisions
    file_name = time.time_ns()
    with open(f"state/{file_name}.json", "w") as f:
        json.dump(serializable_result, f, indent=2)

    print(f"Saved {len(serializable_result)} text detections to state/{file_name}.json")

    # Wait 5 seconds before next capture
    # Note: OCR is slow, so frequent captures may cause performance issues
    time.sleep(5)
