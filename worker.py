import pyautogui
pyautogui.FAILSAFE = False
import time
import random
import math
import requests




def random_move():
    pyautogui.moveTo(random.choice(range(1, 1920)), random.choice(range(1, 1080)), 2, pyautogui.easeInOutQuad)
    pyautogui.click()
    time.sleep(random.choice(range(1,3)))


def mount_up():
    pyautogui.press("a")

def get_predict(image):
    from io import BytesIO
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG', quality=10)
    image_bytes.seek(0)
    files = {'file': ('screen.jpg', image_bytes, 'image/jpg')}
    headers = {}
    url = "http://detection.diopthe20.com/fiber_detection/"
    
    try:
        response = requests.post(url, files=files, headers=headers)
        return response.json()
    except requests.exceptions.ConnectionError:
        pass

def job_request():
    pass

class Worker():
    master_url = 'http://sv.diopthe20.com/'

    def logging(self):
        requests.post(self.master_url)
        
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
    

