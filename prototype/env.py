import time

import cv2
import numpy as np
from PIL import Image

# Minimal Gym-like wrapper that captures the screen and issues keyboard/mouse via pyautogui.
# This is intentionally minimal: it exposes reset(), step(action), render(), close().


class GameEnv:
    """Prototype environment for interacting with a game client via screenshots and pyautogui.

    Observation: RGB image (H,W,3) as uint8 numpy array
    Action: discrete integers mapped to simple key presses
    """

    ACTIONS = {
        0: None,  # noop
        1: ("key", "w"),  # move forward
        2: ("key", "s"),  # move back
        3: ("key", "a"),  # left
        4: ("key", "d"),  # right
        5: ("key", "space"),  # jump / interact
        # Mouse actions
        6: ("mouse_move", 0, -80),  # look/move cursor up
        7: ("mouse_move", 0, 80),  # down
        8: ("mouse_move", -80, 0),  # left
        9: ("mouse_move", 80, 0),  # right
        10: ("mouse_click", "left"),  # left click
        11: ("mouse_click", "right"),  # right click
        12: ("mouse_move_to_center",),  # move cursor to capture center
    }

    def __init__(self, capture_region=None, fps=8):
        # capture_region: (left, top, width, height) or None for full screen
        self.capture_region = capture_region
        self.fps = fps
        self._last_obs = None
        self._running = True

    def _capture(self):
        try:
            import mss
        except Exception:
            raise RuntimeError("mss is required for screen capture (pip install mss)")

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            if self.capture_region:
                left, top, w, h = self.capture_region
                monitor = {"left": left, "top": top, "width": w, "height": h}
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
            arr = np.array(img)
            return arr

    def reset(self):
        # No explicit env reset for the real game; return current screen
        obs = self._capture()
        self._last_obs = obs
        return obs

    def step(self, action, duration=0.1):
        # Execute simple mapped action using pyautogui
        a = self.ACTIONS.get(int(action), None)
        if a is not None:
            kind = a[0]
            try:
                import pyautogui
            except Exception:
                pyautogui = None

            if kind == "key":
                key = a[1]
                if pyautogui:
                    try:
                        pyautogui.keyDown(key)
                        time.sleep(duration)
                        pyautogui.keyUp(key)
                    except Exception:
                        pass

            elif kind == "mouse_move":
                dx, dy = a[1], a[2]
                if pyautogui:
                    try:
                        pyautogui.moveRel(dx, dy, duration=max(0.01, duration))
                    except Exception:
                        pass

            elif kind == "mouse_click":
                button = a[1]
                if pyautogui:
                    try:
                        pyautogui.click(button=button)
                    except Exception:
                        pass

            elif kind == "mouse_move_to_center":
                if pyautogui:
                    try:
                        # move to center of capture region or screen
                        if self.capture_region:
                            left, top, w, h = self.capture_region
                            cx = left + w // 2
                            cy = top + h // 2
                        else:
                            import mss

                            with mss.mss() as sct:
                                mon = sct.monitors[1]
                                cx = mon["left"] + mon["width"] // 2
                                cy = mon["top"] + mon["height"] // 2
                        pyautogui.moveTo(cx, cy, duration=max(0.01, duration))
                    except Exception:
                        pass
        # Wait a bit to match fps
        time.sleep(max(0, 1.0 / self.fps - 0.01))
        obs = self._capture()
        self._last_obs = obs
        # No reward signal in prototype; return 0 and done=False
        return obs, 0.0, False, {}

    def render(self, window_name="game"):
        if self._last_obs is None:
            return
        cv2.imshow(window_name, cv2.cvtColor(self._last_obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        self._running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
