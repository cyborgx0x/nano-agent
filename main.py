import math
import random
import time

import numpy as np
import pyautogui

from fiber_detection import get_detection_fiber
from gather_state import get_gather_state
from value_sort import sort_resource_by_value

pyautogui.FAILSAFE = False


def random_move():
    pyautogui.moveTo(
        random.choice(range(1, 1920)),
        random.choice(range(1, 1080)),
        2,
        pyautogui.easeInOutQuad,
    )
    pyautogui.click()
    time.sleep(random.choice(range(1, 3)))


def stream(image):
    """
    based on stream of image, create the best decision
    """
    return {"fiber_inference": get_detection_fiber(image)}


def get_nearest_resource(result):

    k0, j0 = (1920 / 2, 1080 / 2)
    x0, y0 = (1920, 1080)
    nearest = 0

    for index, item in enumerate(result):
        if item["confidence"] < 0.7:
            continue

        x1, y1, x2, y2 = (
            item["box"]["x1"],
            item["box"]["y1"],
            item["box"]["x2"],
            item["box"]["y2"],
        )

        k1, j1 = math.ceil((x1 + x2) / 2), math.ceil((y1 + y2) / 2)

        if (k1 - k0) ** 2 + (j1 - j0) ** 2 < (x0 - k0) ** 2 + (y0 - k0) ** 2:
            x0, y0 = k1, j1
            nearest = index

    bbox = result[nearest]["box"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    return x1, y1, x2, y2


def mount_up():
    pyautogui.press("a")


def act(env_state):
    """
    based on env state, agent will decide what to do
    """
    fiber_inference = env_state["fiber_inference"][0]
    if len(fiber_inference) != 0:
        # array = sort_resource_by_value(fiber_inference)
        # box = array[0]["box"]
        # pyautogui.moveTo(
        #     math.ceil((box["x1"] + box["x2"]) / 2),
        #     math.ceil((box["y1"] + box["y2"]) / 2),
        #     0.5,
        #     pyautogui.easeInOutQuad,
        # )
        # nearest = get_nearest_resource(fiber_inference)
        x1, y1, x2, y2 = (
            fiber_inference[0]["box"]["x1"],
            fiber_inference[0]["box"]["y1"],
            fiber_inference[0]["box"]["x2"],
            fiber_inference[0]["box"]["y2"],
        )
        pyautogui.moveTo(
            math.ceil((x1 + x2) / 2),
            math.ceil((y1 + y2) / 2),
            0.5,
            pyautogui.easeInOutQuad,
        )
        pyautogui.click()
    start_time = time.time()
    while True:
        result = call_back()
        if result:
            break
        if time.time() - start_time > 5:
            break


def call_back():
    region = (800, 200, 300, 600)

    array = []
    while True:
        screenshot = pyautogui.screenshot(region=region)
        img_np = np.array(screenshot)
        array.append(img_np)

        if len(array) == 3:
            break

    for img in array:
        gather_state = get_gather_state(img)
        print(gather_state)
        if "0/9" in gather_state or "0/6" in gather_state:
            return True


def main_loop():
    while True:
        image = pyautogui.screenshot()
        env = stream(image)
        act(env)


# Start the main loop
main_loop()
