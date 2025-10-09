"""
MAIN BOT LOOP - Albion Online Resource Gathering Automation

This is the main entry point for an autonomous game bot that:
1. Takes screenshots of the game screen
2. Uses computer vision (YOLO) to detect resources (fiber/cotton/flax/hemp)
3. Clicks on detected resources
4. Monitors gathering progress using OCR
5. Repeats the cycle

Technology Stack:
- PyAutoGUI: Screen capture and mouse control
- YOLO (YOLOv8): Object detection for identifying resources on screen
- EasyOCR: Optical Character Recognition to read gathering progress text
"""

import math
import random
import time

import numpy as np
import pyautogui

from fiber_detection import get_detection_fiber
from gather_state import get_gather_state
from value_sort import sort_resource_by_value

# Disable PyAutoGUI's safety feature (moving mouse to corner stops script)
# WARNING: In production, you might want this enabled for safety
pyautogui.FAILSAFE = False


def random_move():
    """
    Perform a random mouse movement and click on the screen.

    This function is useful when no resources are detected, allowing the bot to:
    - Explore different areas of the screen
    - Simulate human-like random behavior
    - Potentially discover new resource nodes

    Screen coordinates assume 1920x1080 resolution (Full HD).
    Uses easeInOutQuad for smooth, human-like cursor movement.
    """
    pyautogui.moveTo(
        random.choice(range(1, 1920)),  # Random X coordinate
        random.choice(range(1, 1080)),  # Random Y coordinate
        2,  # Duration in seconds (smooth movement)
        pyautogui.easeInOutQuad,  # Easing function for natural motion
    )
    pyautogui.click()
    time.sleep(random.choice(range(1, 3)))  # Random delay to avoid bot detection


def stream(image):
    """
    Process a screenshot and extract game state information.

    This is the PERCEPTION layer of the bot:
    - Takes a PIL Image object (screenshot)
    - Runs YOLO object detection to find resources (fiber, cotton, flax, hemp)
    - Returns structured data about detected objects

    Args:
        image: PIL Image object from pyautogui.screenshot()

    Returns:
        dict: {"fiber_inference": [list of detected objects with bounding boxes and confidence]}
    """
    return {"fiber_inference": get_detection_fiber(image)}


def get_nearest_resource(result):
    """
    Find the resource closest to the center of the screen.

    This implements a simple decision strategy: prioritize resources that are
    nearest to the player (assumed to be at screen center).

    Algorithm:
    1. Center of screen (960, 540) is the reference point (player position)
    2. Calculate Euclidean distance from each detected resource to center
    3. Return the bounding box of the nearest resource

    Args:
        result: List of detected objects from YOLO, each with:
                - confidence: Detection confidence score (0-1)
                - box: {x1, y1, x2, y2} bounding box coordinates

    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the nearest resource's bounding box
    """
    k0, j0 = (1920 / 2, 1080 / 2)  # Screen center (player position)
    x0, y0 = (1920, 1080)  # Initialize with far corner
    nearest = 0  # Index of nearest resource

    for index, item in enumerate(result):
        # Skip low-confidence detections (likely false positives)
        if item["confidence"] < 0.7:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = (
            item["box"]["x1"],
            item["box"]["y1"],
            item["box"]["x2"],
            item["box"]["y2"],
        )

        # Calculate center point of the resource
        k1, j1 = math.ceil((x1 + x2) / 2), math.ceil((y1 + y2) / 2)

        # Compare Euclidean distance: sqrt((k1-k0)² + (j1-j0)²)
        # We skip sqrt since we only need comparison
        if (k1 - k0) ** 2 + (j1 - j0) ** 2 < (x0 - k0) ** 2 + (y0 - k0) ** 2:
            x0, y0 = k1, j1
            nearest = index

    # Return bounding box of the nearest resource
    bbox = result[nearest]["box"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    return x1, y1, x2, y2


def mount_up():
    """
    Press the 'A' key to mount/dismount in Albion Online.

    In the game, mounting increases movement speed between resource nodes,
    improving gathering efficiency.
    """
    pyautogui.press("a")


def act(env_state):
    """
    ACTION layer: Decide what to do based on the environment state.

    This is the bot's decision-making function:
    1. If resources detected → Move cursor to resource center and click
    2. Wait and monitor gathering progress using callback
    3. Timeout after 5 seconds if gathering doesn't complete

    Args:
        env_state: dict containing {"fiber_inference": [[detected_objects]]}

    Note: Currently uses first detected object (fiber_inference[0]).
          Commented code shows alternative strategies:
          - sort_resource_by_value: Prioritize valuable resources (hemp > flax > cotton)
          - get_nearest_resource: Click nearest resource to player
    """
    fiber_inference = env_state["fiber_inference"][0]

    if len(fiber_inference) != 0:
        # STRATEGY: Currently just takes the first detected resource
        # Alternative strategies (commented out):
        # 1. Sort by value: array = sort_resource_by_value(fiber_inference)
        # 2. Get nearest: nearest = get_nearest_resource(fiber_inference)

        # Extract bounding box of first detected resource
        x1, y1, x2, y2 = (
            fiber_inference[0]["box"]["x1"],
            fiber_inference[0]["box"]["y1"],
            fiber_inference[0]["box"]["x2"],
            fiber_inference[0]["box"]["y2"],
        )

        # Move cursor to center of resource and click
        pyautogui.moveTo(
            math.ceil((x1 + x2) / 2),  # Center X
            math.ceil((y1 + y2) / 2),  # Center Y
            0.5,  # Movement duration (seconds)
            pyautogui.easeInOutQuad,  # Smooth easing
        )
        pyautogui.click()

    # Wait for gathering to complete (with timeout)
    start_time = time.time()
    while True:
        result = call_back()  # Check if gathering finished
        if result:
            break
        if time.time() - start_time > 5:  # 5 second timeout
            break


def call_back():
    """
    Monitor gathering progress using OCR to detect completion.

    This function:
    1. Takes 3 screenshots of the gathering UI region (right side of screen)
    2. Uses EasyOCR to read text from screenshots
    3. Looks for "0/9" or "0/6" which indicates inventory is full

    In Albion Online, the gathering UI shows progress like:
    - "3/9" = 3 items gathered out of 9 capacity
    - "0/9" = No more space, gathering complete!

    Region (800, 200, 300, 600):
    - X=800, Y=200: Top-left corner
    - Width=300, Height=600: Captures the right UI area

    Returns:
        bool: True if gathering is complete (inventory full), False otherwise
    """
    region = (800, 200, 300, 600)  # UI region showing gathering progress

    # Take 3 screenshots to ensure we catch the text
    array = []
    while True:
        screenshot = pyautogui.screenshot(region=region)
        img_np = np.array(screenshot)
        array.append(img_np)

        if len(array) == 3:
            break

    # Check each screenshot for completion text
    for img in array:
        gather_state = get_gather_state(img)  # Run OCR
        print(gather_state)
        # "0/9" or "0/6" means inventory full (different resource types have different stack sizes)
        if "0/9" in gather_state or "0/6" in gather_state:
            return True


def main_loop():
    """
    The MAIN BOT LOOP - Runs continuously to automate resource gathering.

    This implements the classic sense-think-act cycle:
    1. SENSE: Capture screenshot of game
    2. THINK: Process image with YOLO to detect resources
    3. ACT: Click on resources and monitor gathering

    Loop runs infinitely until manually stopped (Ctrl+C).
    """
    while True:
        image = pyautogui.screenshot()  # SENSE: Capture current game state
        env = stream(image)              # THINK: Process with computer vision
        act(env)                         # ACT: Execute actions in game


# ============================================================================
# ENTRY POINT: Start the bot
# ============================================================================
if __name__ == "__main__":
    print("🤖 Starting Albion Online Resource Gathering Bot...")
    print("📸 YOLO model loaded for fiber detection")
    print("🔤 EasyOCR ready for gathering state monitoring")
    print("⚠️  Press Ctrl+C to stop the bot")
    print("-" * 60)
    main_loop()
