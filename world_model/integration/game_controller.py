"""
Game controller integrating world model with existing agent.
Bridges the world model predictions with actual game control.
"""

import time
import numpy as np
import pyautogui
import torch
from mss import mss


class WorldModelGameController:
    """
    Controls the game using world model predictions.
    Integrates with existing agent.py and action_space.py
    """

    def __init__(
        self,
        world_model,
        screen_region=None,
        action_delay=0.1,  # Delay between actions (seconds)
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            world_model: GameWorldModel instance
            screen_region: Screen region to capture {'top': y, 'left': x, 'width': w, 'height': h}
            action_delay: Time to wait between actions
            device: torch device
        """
        self.world_model = world_model
        self.screen_region = screen_region
        self.action_delay = action_delay
        self.device = device

        # Screen capture
        self.sct = mss()

        # State tracking
        self.current_mouse_pos = pyautogui.position()
        self.last_action_time = 0

    def capture_screen(self):
        """
        Capture current game screenshot.

        Returns:
            frame: numpy array [H, W, 3] RGB image
        """
        if self.screen_region:
            screenshot = self.sct.grab(self.screen_region)
        else:
            screenshot = self.sct.grab(self.sct.monitors[1])

        # Convert to RGB numpy array
        frame = np.array(screenshot)[:, :, :3]
        return frame

    def execute_action(self, action):
        """
        Execute action in the game.

        Args:
            action: dict with keys 'mouse_delta', 'click', 'key'
        """
        # Rate limiting
        elapsed = time.time() - self.last_action_time
        if elapsed < self.action_delay:
            time.sleep(self.action_delay - elapsed)

        # Mouse movement
        if 'mouse_delta' in action and action['mouse_delta'] is not None:
            dx, dy = action['mouse_delta']
            current_pos = pyautogui.position()
            new_x = current_pos[0] + int(dx)
            new_y = current_pos[1] + int(dy)

            # Clamp to screen bounds
            screen_width, screen_height = pyautogui.size()
            new_x = max(0, min(screen_width - 1, new_x))
            new_y = max(0, min(screen_height - 1, new_y))

            # Move mouse
            pyautogui.moveTo(new_x, new_y, duration=0.1)
            self.current_mouse_pos = (new_x, new_y)

        # Click
        if action.get('click', False):
            pyautogui.click()

        # Key press
        if action.get('key') is not None:
            pyautogui.press(action['key'])

        self.last_action_time = time.time()

    def run_episode(self, goal_image=None, max_steps=100, verbose=True):
        """
        Run one episode using world model for control.

        Args:
            goal_image: Target image (numpy array) or None
            max_steps: Maximum steps per episode
            verbose: Print debug info

        Returns:
            success: Whether goal was reached
            steps: Number of steps taken
        """
        if goal_image is None:
            print("Warning: No goal image provided. Running in exploration mode.")

        # Encode goal
        if goal_image is not None:
            goal_rep = self.world_model.encode(goal_image)
        else:
            goal_rep = None

        steps = 0
        for step in range(max_steps):
            # Capture current state
            current_frame = self.capture_screen()
            current_rep = self.world_model.encode(current_frame)

            # Plan action
            if goal_rep is not None:
                action = self.world_model.plan_action(
                    current_rep=current_rep,
                    goal_rep=goal_rep,
                    current_mouse_pos=self.current_mouse_pos
                )
            else:
                # Exploration mode: random actions
                action = self._random_action()

            # Execute action
            self.execute_action(action)
            steps += 1

            if verbose and step % 10 == 0:
                print(f"Step {step}/{max_steps}: Action = {action}")

            # Check if goal reached (simple threshold)
            if goal_rep is not None:
                similarity = self._compute_similarity(current_rep, goal_rep)
                if similarity < 0.1:  # Threshold for "close enough"
                    if verbose:
                        print(f"Goal reached in {steps} steps!")
                    return True, steps

        if verbose:
            print(f"Episode finished after {steps} steps (goal not reached)")

        return False, steps

    def _compute_similarity(self, rep1, rep2):
        """
        Compute similarity between two representations.

        Args:
            rep1, rep2: [1, N, D] torch tensors

        Returns:
            similarity: float (lower = more similar)
        """
        dist = torch.mean(torch.abs(rep1 - rep2)).item()
        return dist

    def _random_action(self):
        """Generate random action for exploration."""
        return {
            'mouse_delta': np.random.randn(2) * 50,  # Random movement
            'click': np.random.rand() < 0.1,  # 10% chance of click
            'key': None,
        }


class HybridController(WorldModelGameController):
    """
    Hybrid controller combining world model with heuristic rules.
    Falls back to rules when world model is uncertain.
    """

    def __init__(self, world_model, heuristic_agent, **kwargs):
        """
        Args:
            world_model: GameWorldModel instance
            heuristic_agent: Your existing agent (from agent.py or main.py)
            **kwargs: Additional args for WorldModelGameController
        """
        super().__init__(world_model, **kwargs)
        self.heuristic_agent = heuristic_agent
        self.use_world_model = True  # Can toggle between modes

    def run_episode(self, goal_image=None, max_steps=100, verbose=True):
        """
        Run episode with hybrid control.
        Uses world model when confident, falls back to heuristics otherwise.
        """
        if not self.use_world_model:
            # Use pure heuristic mode
            return self._run_heuristic_episode(max_steps, verbose)

        # Try world model first
        try:
            return super().run_episode(goal_image, max_steps, verbose)
        except Exception as e:
            if verbose:
                print(f"World model failed: {e}. Falling back to heuristics.")
            return self._run_heuristic_episode(max_steps, verbose)

    def _run_heuristic_episode(self, max_steps, verbose):
        """
        Fallback to heuristic control.
        Integrate with your existing agent logic.
        """
        # TODO: Integrate with your existing agent from agent.py or main.py
        print("Running heuristic control (not implemented yet)")
        return False, 0


# Example integration with existing agent
class WorldModelAgent:
    """
    Enhanced agent using world model.
    Replaces the skeleton Agent class in agent.py
    """

    def __init__(self, world_model, model=None):
        """
        Args:
            world_model: GameWorldModel instance
            model: Optional YOLO or other model for hybrid approach
        """
        self.world_model = world_model
        self.model = model  # Legacy model (YOLO, etc.)
        self.controller = WorldModelGameController(world_model)

    @property
    def position(self):
        """
        Get player position using world model representations.
        World model's latent space implicitly encodes position.
        """
        frame = self.controller.capture_screen()
        latent = self.world_model.encode(frame)
        # TODO: Decode position from latent (requires training)
        return None

    @property
    def weight(self):
        """
        Get inventory weight.
        Could be decoded from world model or use OCR.
        """
        # TODO: Implement using world model or OCR
        return None

    def goto(self, goal_image):
        """
        Navigate to a goal state represented by an image.

        Args:
            goal_image: numpy array [H, W, 3] target screenshot

        Returns:
            success: bool
        """
        success, steps = self.controller.run_episode(
            goal_image=goal_image,
            max_steps=100,
            verbose=True
        )
        return success

    def gather_resource(self, resource_image):
        """
        Gather a resource by navigating to it.

        Args:
            resource_image: Screenshot showing the resource

        Returns:
            success: bool
        """
        return self.goto(resource_image)


# Example usage
if __name__ == "__main__":
    # This is a demo - requires trained world model
    print("World Model Game Controller")
    print("This requires a trained world model to work.")
    print("First, collect training data using gameplay_dataset.py")
    print("Then train the world model.")
    print("Finally, use this controller to play the game.")
