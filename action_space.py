import pyautogui

def mount_up():
    pyautogui.press("a")

def click(x, y):
    pyautogui.click(x, y)

def click_and_drag(x1, y1, x2, y2):
    pyautogui.dragTo(x1, y1, x2, y2)

def toggle_inventory():
    pyautogui.press("3")


def toggle_mini_map():
    pyautogui.press("tab")

