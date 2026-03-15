"""
GATHER STATE MODULE - OCR for Reading Game UI Text

This module uses EasyOCR (Optical Character Recognition) to read text
from Albion Online's gathering UI.

OCR Process:
1. Takes a screenshot of the gathering progress UI
2. Detects and recognizes text in the image
3. Extracts gathering status like "3/9" or "0/9"

EasyOCR uses deep learning to recognize text in images, supporting
multiple languages and working well with game UIs.
"""

import easyocr

# Initialize EasyOCR reader
# - Language: English only (["en"])
# - GPU: Enabled for faster processing (requires CUDA)
reader = easyocr.Reader(["en"], gpu=True)


def get_gather_state(img_np):
    """
    Extract gathering status text from a UI screenshot using OCR.

    This function reads text from the gathering progress UI to determine
    if the bot should continue gathering or if inventory is full.

    Args:
        img_np: NumPy array of the screenshot (from np.array(screenshot))

    Returns:
        str: Concatenated text found in the image
             Example: "Cotton 3/9" or "Flax 0/6"

    EasyOCR returns:
        [
          ([[x1,y1], [x2,y2], ...], "text", confidence),
          ...
        ]
    We only need the text (item[1]), not bounding boxes or confidence.
    """
    # Run OCR on the image
    gather_state = reader.readtext(img_np)

    # Extract only the text portion (index 1) from each detection
    gather_state = [item[1] for item in gather_state]

    # Join all detected text into a single string
    return " ".join(gather_state)