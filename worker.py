import pyautogui
pyautogui.FAILSAFE = False
import time
import random
import math

def random_move():
    pyautogui.moveTo(random.choice(range(1, 1920)), random.choice(range(1, 1080)), 2, pyautogui.easeInOutQuad)
    pyautogui.click()
    time.sleep(random.choice(range(1,3)))


def mount_up():
    pyautogui.press("a")

def get_predict(image):
    import requests
    from io import BytesIO
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    files = {'file': ('example.png', image_bytes, 'image/png')}
    headers = {}
    url = "http://detection.diopthe20.com/fiber_detection/"
    response = requests.post(url, files=files, headers=headers)

    return response.json()

while True:
    image = pyautogui.screenshot()
    
    from timeit import default_timer as timer
    t1 = timer()
    result = get_predict(image=image)
    t2 = timer()
    print("Time taken: ", t2-t1 )
    print(result)
    if result["data"] != None:
        x1, y1, x2, y2 = result["data"]
        pyautogui.moveTo(math.ceil((x1+x2)/2), math.ceil((y1+y2)/2), 0.5, pyautogui.easeInOutQuad)
        pyautogui.click()
        time.sleep(2)
    else:
        time.sleep(1)
    
