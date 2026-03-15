import pyautogui
import time
from pynput import mouse

from pynput import mouse
file_name = "t15.json"
with open(file_name, "a") as f:
            f.write(f"[\n")
def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def on_click(x, y, button, pressed):
    try:
        if pressed:
            with open(file_name, "a") as f:
                f.write(f"[{x}, {y}],\n")
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
    except KeyboardInterrupt:
        return False

def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))

# Collect events until released
with mouse.Listener(
        on_click=on_click) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
# listener = mouse.Listener(
#     on_click=on_click,)
# listener.start()

# time.sleep(5)
# f  = open("mouse_rc", "a")
# try:
#     while True:
#         time.sleep(0.1)
#         p = f"{pyautogui.position().x}, {pyautogui.position().y}\n"
#         print(p)
#         f.write(p)
# except KeyboardInterrupt:
#     f.close()

def to_center():
    pyautogui.moveTo(962, 480, 1, pyautogui.easeInOutQuad)
    print(pyautogui.position())
    pyautogui.rightClick()

def move_left():
    pyautogui.moveTo(0, 480, 1, pyautogui.easeInOutQuad)
    print(pyautogui.position())
    pyautogui.rightClick()

def move_down():
    pyautogui.moveTo(962, 600, 1, pyautogui.easeInOutQuad)
    print(pyautogui.position())
    pyautogui.rightClick()

with open(file_name, "a") as f:
            f.write(f"]\n")