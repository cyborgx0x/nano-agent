"""
WORKER MODULE - Client for Remote YOLO Inference Server

This is an alternative architecture where:
1. Worker (this file) runs on the gaming PC
2. Captures screenshots and sends to remote server
3. Server runs YOLO inference (offloading GPU compute)
4. Worker receives detection results and clicks resources

Benefits:
- Separates bot logic from ML inference
- Can run YOLO on a more powerful GPU server
- Multiple workers can share one inference server
- Easier to update model without touching bot clients

Architecture:
  Gaming PC (Worker) <--HTTP--> Inference Server (YOLO model)
"""

import pyautogui

pyautogui.FAILSAFE = False  # Disable safety feature
import time
import random
import math
import requests
import json


def get_server():
    """
    Query the server registry to get the current inference server URL.

    This allows dynamic server discovery - the URL can change without
    updating worker code. Useful for load balancing or failover.

    Returns:
        str: Base URL of the inference server (e.g., "https://server.com/")
    """
    url = "https://sv.diopthe20.com/"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)

    return response.json()


def random_move():
    """
    Perform random movement when no resources are found.

    Same as main.py random_move() - helps explore the map.
    """
    pyautogui.moveTo(
        random.choice(range(1, 1920)),
        random.choice(range(1, 1080)),
        2,
        pyautogui.easeInOutQuad,
    )
    pyautogui.click()
    time.sleep(random.choice(range(1, 3)))


def mount_up():
    """Press 'A' to mount/dismount (increase movement speed)."""
    pyautogui.press("a")


def get_predict(server, image):
    """
    Send screenshot to remote server for YOLO inference.

    This is the KEY function for distributed architecture:
    1. Compress image to JPEG (quality=10 for fast upload)
    2. Send as multipart/form-data to server
    3. Server runs YOLO and returns detection results
    4. Worker receives bounding box coordinates

    Args:
        server: Base URL of inference server
        image: PIL Image from pyautogui.screenshot()

    Returns:
        dict: {"data": [x1, y1, x2, y2]} or {"data": None} if no resources
              Note: Server returns only the first/best detection

    Network optimization: Low JPEG quality (10%) reduces upload time
    """
    from io import BytesIO

    # Convert PIL Image to JPEG bytes (compressed for fast transfer)
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG", quality=10)  # Heavy compression
    image_bytes.seek(0)

    # Prepare multipart form data
    files = {"file": ("screen.jpg", image_bytes, "image/jpg")}
    headers = {}
    url = f"{server}/fiber_detection/"
    print(url)

    try:
        response = requests.post(url, files=files, headers=headers)
        return response.json()
    except:
        pass  # Network error - will retry next loop


def job_request():
    """
    Placeholder for job assignment system.

    Future feature: Server could assign specific zones/resources
    to different workers for coordinated farming.
    """
    pass


def log_work(server, content):
    """
    Report bot activity to server for monitoring/analytics.

    Sends status updates like:
    - "Found fiber at (x, y)"
    - "No fiber found"
    - "Gathering complete"

    This enables centralized bot monitoring and statistics.

    Args:
        server: Base URL of server
        content: Status message string
    """
    url = f"{server}/status/"

    payload = json.dumps({"id": 1, "status": content})
    headers = {"Content-Type": "application/json"}

    requests.request("POST", url, headers=headers, data=payload)

# ============================================================================
# MAIN WORKER LOOP - Remote Inference Client
# ============================================================================

# Get inference server URL from registry
server = get_server()
a = 0  # Counter for "no resource found" iterations

while True:
    # 1. CAPTURE: Screenshot of game
    image = pyautogui.screenshot()

    from timeit import default_timer as timer

    # 2. INFER: Send to remote server for YOLO detection
    t1 = timer()
    result = get_predict(server, image=image)
    t2 = timer()
    print("Time taken: ", t2 - t1)
    print(result)

    try:
        result["data"]
        if result["data"] != None:
            # Resource found! Extract bounding box
            x1, y1, x2, y2 = result["data"]

            # 3. ACT: Move cursor to resource center and click
            pyautogui.moveTo(
                math.ceil((x1 + x2) / 2),  # Center X
                math.ceil((y1 + y2) / 2),  # Center Y
                0.5,
                pyautogui.easeInOutQuad,
            )
            pyautogui.click()

            # Log success to server
            log_work(server, f"{x1}, {y1}, {x2}, {y2}")
            time.sleep(2)  # Wait for gathering to start
            a = 0  # Reset no-resource counter
        else:
            # No resource found
            if a == 5:
                # After 5 failures, could do random_move() to explore
                a = 0
                # random_move()  # Currently disabled
            a += 1
            log_work("No Fiber Found")
            time.sleep(1)
    except:
        # Network error or server unavailable - refresh server URL
        server = get_server()
            