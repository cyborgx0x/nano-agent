from ultralytics import YOLO
import keyboard
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('model.pt')  # load a custom model
import pyautogui
import math
import time
import random
# Predict with the model
results = model.predict('screen 1920 0 1920 1080', stream=True, save=True, vid_stride=True)
# result = model.predict('capture.png', show=True, stream=True)
def random_move():
    pyautogui.moveTo(random.choice(range(1, 1920)), random.choice(range(1, 1080)), 2, pyautogui.easeInOutQuad)
    pyautogui.click()
    time.sleep(random.choice(range(1,3)))

for i in results:
    
    if len(i.boxes.xyxy) != 0:
        # for bbox in i.boxes.xyxy: 
        #     x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        #     print(x1,y1,x2,y2)
        bbox = i.boxes.xyxy[0]
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    
        pyautogui.moveTo(math.ceil((x1+x2)/2), math.ceil((y1+y2)/2), 1, pyautogui.easeInOutQuad)
        pyautogui.click()
        time.sleep(3)
        pyautogui.moveTo(math.ceil((1920)/2)+random.choice(range(1, 100)), math.ceil((1080/2)+random.choice(range(1, 100))), 0.9, pyautogui.easeInOutQuad)

        
    # else:
    #     keyboard.press('a')
    #     time.sleep(3)
    #     random_move()

        