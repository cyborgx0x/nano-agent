from ultralytics import YOLO
# import keyboard
# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('model.pt')  # load a custom model
import pyautogui
pyautogui.FAILSAFE = False
import math
import time
import random
# Predict with the model

#results = model.predict('screen', stream=True)

def random_move():
    pyautogui.moveTo(random.choice(range(1, 1920)), random.choice(range(1, 1080)), 2, pyautogui.easeInOutQuad)
    pyautogui.click()
    time.sleep(random.choice(range(1,3)))


def run(result):
    
    if len(result.boxes.xyxy) != 0:
        print(result.boxes.xyxy)
        k0,j0 = (1920/2, 1080/2)
        x0,y0 = (1920, 1080)
        nearest = 0
        boxes_data = result.boxes.xyxy

        for index, bbox in enumerate(boxes_data):
            
            if result.boxes.conf[index].item() < 0.7:
                continue
            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

            k1,j1 = math.ceil((x1+x2)/2), math.ceil((y1+y2)/2)
            
            if (k1-k0)**2 + (j1-j0)**2 < (x0-k0)**2 + (y0-k0)**2:
                x0,y0=k1,j1
                nearest = index
        '''
        after get the nearest point, move the mouse and click :)
        '''
        print(nearest, x0,y0)
        bbox = result.boxes.xyxy[nearest]
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    
        pyautogui.moveTo(math.ceil((x1+x2)/2), math.ceil((y1+y2)/2), 0.5, pyautogui.easeInOutQuad)
        pyautogui.click()
        time.sleep(2)
        # pyautogui.moveTo(math.ceil((1920)/2)+random.choice(range(1, 100)), math.ceil((1080/2)+random.choice(range(1, 100))), 0.9, pyautogui.easeInOutQuad)

        
    # else:
    #     keyboard.press('a')
    #     time.sleep(3)
    #     random_move()

        
while True:
    image = pyautogui.screenshot()
    result = model.predict(image)
    run(result[0])
    
