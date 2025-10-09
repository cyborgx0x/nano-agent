"""
AGENT MODULE - Object-Oriented Bot Architecture (SKELETON)

This is an INCOMPLETE design for a more sophisticated agent architecture.
Currently, this is just a design pattern/template that's not being used.

The idea is to create a stateful agent with:
- position: Player location on the map
- weight: Current inventory weight/capacity
- action: Current action being performed
- next_action: AI/CNN decides next optimal action based on state

This represents a more advanced RL (Reinforcement Learning) approach where
the agent maintains state and makes decisions based on:
  State (position, inventory, etc.) → Neural Network → Action

Compare this to main.py which uses a simpler reactive approach:
  Screenshot → YOLO Detection → Click Resource

TODO: This module needs implementation to be functional.
"""

import pyautogui


class Agent():
    """
    Stateful game agent with properties and action planning.

    This class represents an object-oriented approach to bot design,
    where the agent tracks its state and plans actions accordingly.
    """

    @property
    def position(self):
        """Get player's current position on the map."""
        return self.calculate_position()

    @property
    def weight(self):
        """Get current inventory weight/capacity."""
        return self.calculate_weight()

    @staticmethod
    def capture(*args, **kwargs):
        """
        Static method to capture screenshots.

        This is a wrapper around pyautogui.screenshot() for easier testing
        and potential mock injection.
        """
        return pyautogui.screenshot(*args, **kwargs)

    def calculate_position(self):
        """
        Extract player position from minimap using computer vision.

        TODO: Implement using OCR or template matching on minimap
        Could use color detection to find player icon position
        """
        image = self.__class__.capture()
        # TODO: Implement position detection
        pass

    def calculate_weight(self):
        """
        Extract inventory weight from UI using OCR.

        In Albion, UI shows something like "Weight: 45/100"
        This is useful for deciding when to return to town.

        TODO: Implement using EasyOCR on inventory UI region
        """
        image = self.__class__.capture(region=())
        # TODO: Implement weight detection
        pass

    @property
    def action(self):
        """Get the current action being performed."""
        return self.get_current_action()

    def get_current_acction(self):  # Note: Typo in original code (acction vs action)
        """
        Detect what the agent is currently doing (gathering, moving, etc.).

        TODO: Implement action state detection
        Could use screen analysis to detect gathering UI, movement, combat, etc.
        """
        return None

    @property
    def next_action(self):
        """
        Decide the next action using a neural network.

        This is where reinforcement learning would come in:
        - Input: (position, weight, current_action)
        - Output: Best next action (gather, move, return to town, etc.)
        """
        action_set = (
            self.position,
            self.weight,
            self.action
        )
        return self.send_to_cnn(action_set)

    def send_to_cnn(self):
        """
        Send state to CNN/neural network for action prediction.

        TODO: Implement neural network inference
        This would be the "brain" of the agent using deep learning
        to make optimal decisions based on current state.
        """
        return None