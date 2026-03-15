from ultralytics import YOLO
import keyboard
import pyautogui
import math
import numpy as np
import time
import random
import json


def compute_vector(current_position, target_position):
    x1, y1 = current_position
    x2, y2 = target_position
    vector = x1 - x2, y1 - y2
    return vector


def add_vector(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    return x1 + x2, y1 + y2


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def move(map, results, model=None):
    node_queue = json.load(open(map, "r"))
    # pyautogui.moveTo(522,300)
    # pyautogui.rightClick()
    pyautogui.typewrite(["tab"])
    # random_move()
    for node in node_queue:
        previous_point = (None, None)
        move_to_node(node, results, previous_point, model)
    time.sleep(15)

def main():
    model = YOLO("yolov8n.pt")
    model = YOLO("location_model.pt")

    results = model.predict(
        "screen 1920 0 1920 1080", stream=True, save=True, vid_stride=True
    )
    
    move(map="t1.json", results=results, model=model)
    # move(map="map2.json", results=results)
    # move(map="map3.json", results=results)
    # move(map="map4.json", results=results)
    
        
def get_current_location(results):
    pyautogui.typewrite(["tab"])
    capture = next(results)
    result = capture.boxes.xyxy

    if len(result) != 0:
        bbox = result[0]
        x1, y1, x2, y2 = (
            bbox[0].item(),
            bbox[1].item(),
            bbox[2].item(),
            bbox[3].item(),
        )
        return math.ceil((x1+x2)/2), math.ceil((y1+y2)/2)
    

def random_move():
    keyboard.press("~")
    pyautogui.moveTo(
        random.choice(range(1, 1920)),
        random.choice(range(1, 1080)),
        2,
        pyautogui.easeInOutQuad,
    )
    time.sleep(random.choice(range(1, 3)))


def detect_location(results):
    keyboard.press("tab")
    capture = next(results)
    result = capture.boxes.xyxy
    if len(result) == 0:
        keyboard.press("tab")
        random_move()
        time.sleep(3)


def move_to_node(node, results, previous_point, model):
    while True:
        target_position = node
        print(f"heading to {target_position}")
        # pyautogui.click(1860,883)
        # time.sleep(2)

        capture = next(results)
        result = capture.boxes.xyxy
        
        if len(result) != 0:
            bbox = result[0]
            x1, y1, x2, y2 = (
                bbox[0].item(),
                bbox[1].item(),
                bbox[2].item(),
                bbox[3].item(),
            )
            current_position = math.ceil((x1 + x2) / 2), math.ceil((y1 + y2) / 2)
            center_location = 962, 480
            vector = compute_vector(current_position, target_position)
            if vector[0] < 10 and vector[1] < 10:
                return 0
            k = normalize(vector)
            move_range = np.ceil(500 * k) * (-1)
            print("calculate move range")
            print(move_range)
            move_range = move_range[0].item(), move_range[1].item()
            new_target = add_vector(center_location, move_range)
            previous_point = new_target
            print(new_target)
            # time.sleep(2)         
            # pyautogui.click(1860,883)
            keyboard.press("~")
            pyautogui.moveTo(*new_target, 0.4, pyautogui.easeInOutQuad)
            # pyautogui.rightClick()
        else:
            # pyautogui.click(1860,883)
            print(previous_point)
            if previous_point == (None, None):
                keyboard.press("~")

                pyautogui.moveTo(random.choice(range(1, 1920)), random.choice(range(1, 1080)), 0.5, pyautogui.easeInOutQuad)
                continue
            pyautogui.moveTo(*previous_point, 0.4, pyautogui.easeInOutQuad)
            keyboard.press("~")
            # pyautogui.rightClick()
            continue


if __name__ == "__main__":
    main()
