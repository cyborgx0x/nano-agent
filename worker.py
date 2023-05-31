import pyautogui

pyautogui.FAILSAFE = False
import time
import random
import math
import requests
import json


def get_server():
    url = "https://sv.diopthe20.com/"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.text

def random_move():
    pyautogui.moveTo(
        random.choice(range(1, 1920)),
        random.choice(range(1, 1080)),
        2,
        pyautogui.easeInOutQuad,
    )
    pyautogui.click()
    time.sleep(random.choice(range(1, 3)))


def mount_up():
    pyautogui.press("a")


def get_predict(server, image):
    from io import BytesIO

    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG", quality=10)
    image_bytes.seek(0)
    files = {"file": ("screen.jpg", image_bytes, "image/jpg")}
    headers = {}
    url = f"{server}/fiber_detection/"

    try:
        response = requests.post(url, files=files, headers=headers)
        return response.json()
    except:
        pass


def job_request():
    pass


def log_work(server, content):
    url = f"{server}/status/"

    payload = json.dumps({"id": 1, "status": content})
    headers = {"Content-Type": "application/json"}

    requests.request("POST", url, headers=headers, data=payload)

server = get_server()
a  = 0 
while True:
    image = pyautogui.screenshot()

    from timeit import default_timer as timer

    t1 = timer()
    result = get_predict(server, image=image)
    t2 = timer()
    print("Time taken: ", t2 - t1)
    print(result)
    try:
        result["data"]
    except:
        server = get_server()
    if result["data"] != None:
        x1, y1, x2, y2 = result["data"]
        pyautogui.moveTo(
            math.ceil((x1 + x2) / 2),
            math.ceil((y1 + y2) / 2),
            0.5,
            pyautogui.easeInOutQuad,
        )
        pyautogui.click()
        log_work(server, f"{x1}, {y1}, {x2}, {y2}")
        time.sleep(2)
        a = 0
    else:
        if a == 5:
            a = 0
            # random_move()
        a += 1
        log_work("No Fiber Found")
        time.sleep(1)
            