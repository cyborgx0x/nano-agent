"""
ACTION SPACE MODULE - Available Game Actions

This module defines the "action space" - all possible actions the bot
can perform in the game. This is similar to how reinforcement learning
frameworks define discrete action spaces.

In RL terms:
- State Space: What the agent can observe (screenshots, UI data)
- Action Space: What the agent can do (defined here)
- Reward: Game objectives (resources gathered, silver earned)

Each function maps to a specific in-game action via keyboard/mouse controls.
"""

import pyautogui


def mount_up():
    """
    Press 'A' to mount/dismount.

    Mounting increases movement speed, useful for traveling between resource nodes.
    """
    pyautogui.press("a")


def click(x, y):
    """
    Click at specific screen coordinates.

    Args:
        x: X coordinate (0-1920 for Full HD)
        y: Y coordinate (0-1080 for Full HD)

    Used for clicking on resources, NPCs, items, etc.
    """
    pyautogui.click(x, y)


def click_and_drag(x1, y1, x2, y2):
    """
    Drag from (x1, y1) to (x2, y2).

    This could be used for:
    - Moving items in inventory
    - Drawing paths on minimap
    - Camera rotation

    Note: Current implementation might have a bug - dragTo() takes one coordinate pair,
          not two. Should probably be: dragTo(x2, y2, duration=0.5)
    """
    pyautogui.dragTo(x1, y1, x2, y2)


def toggle_inventory():
    """
    Press '3' to open/close inventory.

    Inventory management is crucial for:
    - Checking if bag is full
    - Organizing items
    - Selling to vendors
    """
    pyautogui.press("3")


def toggle_mini_map():
    """
    Press 'Tab' to toggle minimap.

    The minimap is useful for:
    - Navigation
    - Seeing nearby resources
    - Avoiding danger zones/PvP
    """
    pyautogui.press("tab")

