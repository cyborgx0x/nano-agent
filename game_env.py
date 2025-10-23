"""
Gym Environment Wrapper for Albion Online Resource Gathering
Integrates OCR and YOLO object detection as sensors
Supports RL training with stable-baselines3
"""

import gymnasium as gym
import numpy as np
import pyautogui
import time
from typing import Dict, Tuple, Optional, List
import cv2

from fiber_detection import FiberDetection
from gather_state import GatherState


class AlbionGatherEnv(gym.Env):
    """
    Gym environment for Albion Online resource gathering.

    Observation Space:
        - YOLO detections (bounding boxes, classes, confidences)
        - OCR state (inventory count)
        - Player state (last action, time step)

    Action Space:
        - 0: Click on nearest cotton
        - 1: Click on nearest flax
        - 2: Click on nearest hemp
        - 3: Wait (do nothing)
        - 4: Mount up

    Reward:
        - +10 for successful gather
        - -0.1 per time step (encourage efficiency)
        - +5 for gathering higher tier resources
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        model_path: str = 'model.pt',
        max_steps: int = 1000,
        confidence_threshold: float = 0.7,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        # Initialize sensors
        self.detector = FiberDetection(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.ocr_reader = GatherState()

        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # State tracking
        self.last_inventory_count = 0
        self.total_gathered = 0
        self.last_action = 0

        # Action space: 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)

        # Observation space: flattened state vector
        # - Max 10 detections x 6 features (x, y, w, h, class, conf) = 60
        # - Inventory count = 1
        # - Last action = 1
        # - Current step normalized = 1
        # Total = 63
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(63,),
            dtype=np.float32
        )

        # Class names and values
        self.class_names = {0: 'cotton', 1: 'flax', 2: 'hemp'}
        self.class_rewards = {0: 1.0, 1: 3.0, 2: 5.0}  # Higher tier = more reward

        print("AlbionGatherEnv initialized")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space.shape}")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0
        self.last_inventory_count = self._get_inventory_count()
        self.total_gathered = 0
        self.last_action = 0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return (observation, reward, terminated, truncated, info)

        Args:
            action: Integer action (0-4)

        Returns:
            observation: State vector
            reward: Scalar reward
            terminated: Whether episode is done (success)
            truncated: Whether episode is truncated (time limit)
            info: Additional information
        """
        self.current_step += 1
        self.last_action = action

        # Take screenshot before action
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)

        # Execute action
        action_success = self._execute_action(action, screenshot_np)

        # Wait for action to complete
        time.sleep(0.5)

        # Get reward
        reward = self._compute_reward(action, action_success, screenshot_np)

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int, screenshot: np.ndarray) -> bool:
        """
        Execute the selected action

        Args:
            action: Action index
            screenshot: Current screenshot

        Returns:
            success: Whether action was successfully executed
        """
        if action == 3:  # Wait
            return True

        if action == 4:  # Mount up
            pyautogui.press('a')  # Assuming 'a' is mount key
            return True

        # Actions 0-2: Click on resource
        target_class = action  # 0=cotton, 1=flax, 2=hemp

        # Get detections
        detections = self.detector.detect(screenshot)

        # Filter by target class
        target_detections = [
            d for d in detections
            if d['class'] == target_class
        ]

        if not target_detections:
            return False  # No target found

        # Click on nearest detection
        nearest = self._get_nearest_detection(target_detections, screenshot)
        if nearest:
            center_x = int(nearest['x'] + nearest['width'] / 2)
            center_y = int(nearest['y'] + nearest['height'] / 2)

            # Smooth mouse movement
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            pyautogui.click()

            return True

        return False

    def _compute_reward(
        self,
        action: int,
        action_success: bool,
        screenshot: np.ndarray
    ) -> float:
        """
        Compute reward for current step

        Reward structure:
        - Successfully gathering a resource: +1 to +5 (based on tier)
        - Time penalty: -0.1 per step
        - Failed action: -0.5
        - Inventory full: +20 (episode success)
        """
        reward = -0.1  # Time penalty

        # Check if inventory increased
        current_inventory = self._get_inventory_count()
        items_gathered = current_inventory - self.last_inventory_count

        if items_gathered > 0:
            # Successfully gathered!
            self.total_gathered += items_gathered

            # Reward based on resource tier
            if action < 3:  # Was a click action
                tier_reward = self.class_rewards.get(action, 1.0)
                reward += tier_reward * items_gathered
            else:
                reward += 1.0 * items_gathered  # Default reward

            print(f"Gathered {items_gathered} items! Total: {self.total_gathered}")

        elif action < 3 and not action_success:
            # Tried to click but no target found
            reward -= 0.5

        # Update last inventory
        self.last_inventory_count = current_inventory

        # Check if inventory full (9/9)
        if current_inventory >= 9:
            reward += 20.0  # Big bonus for filling inventory

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation

        Returns:
            Flattened state vector of shape (63,)
        """
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)

        # Get detections
        detections = self.detector.detect(screenshot_np)

        # Convert detections to fixed-size array (max 10 detections)
        detection_features = np.zeros((10, 6), dtype=np.float32)

        for i, det in enumerate(detections[:10]):  # Take first 10
            # Normalize coordinates to [0, 1]
            h, w = screenshot_np.shape[:2]
            detection_features[i] = [
                det['x'] / w,                    # Normalized x
                det['y'] / h,                    # Normalized y
                det['width'] / w,                # Normalized width
                det['height'] / h,               # Normalized height
                det['class'] / 2.0,              # Normalized class (0, 1, 2 -> 0, 0.5, 1)
                det['confidence']                # Confidence already [0, 1]
            ]

        # Flatten detections
        detection_flat = detection_features.flatten()  # Shape: (60,)

        # Get inventory count (normalized)
        inventory_norm = self.last_inventory_count / 9.0  # Max is 9

        # Last action (normalized)
        action_norm = self.last_action / 4.0  # Max action is 4

        # Current step (normalized)
        step_norm = self.current_step / self.max_steps

        # Concatenate all features
        obs = np.concatenate([
            detection_flat,           # 60 features
            [inventory_norm],         # 1 feature
            [action_norm],            # 1 feature
            [step_norm]               # 1 feature
        ]).astype(np.float32)

        return obs

    def _get_inventory_count(self) -> int:
        """Get current inventory count using OCR"""
        try:
            count_str = self.ocr_reader.get_gather_count()
            if '/' in count_str:
                current = int(count_str.split('/')[0])
                return current
        except Exception as e:
            pass

        return self.last_inventory_count  # Return last known value

    def _get_nearest_detection(
        self,
        detections: List[Dict],
        screenshot: np.ndarray
    ) -> Optional[Dict]:
        """Get detection nearest to screen center"""
        if not detections:
            return None

        h, w = screenshot.shape[:2]
        center_x, center_y = w / 2, h / 2

        min_dist = float('inf')
        nearest = None

        for det in detections:
            det_x = det['x'] + det['width'] / 2
            det_y = det['y'] + det['height'] / 2

            dist = np.sqrt((det_x - center_x)**2 + (det_y - center_y)**2)

            if dist < min_dist:
                min_dist = dist
                nearest = det

        return nearest

    def _is_terminated(self) -> bool:
        """Check if episode is successfully completed"""
        # Episode ends when inventory is full
        return self.last_inventory_count >= 9

    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'inventory': self.last_inventory_count,
            'total_gathered': self.total_gathered,
            'last_action': self.last_action,
        }

    def render(self):
        """Render environment (optional)"""
        if self.render_mode == 'human':
            # Just show the game screen
            pass

    def close(self):
        """Clean up resources"""
        pass


class SimplifiedAlbionEnv(gym.Env):
    """
    Simplified version with simpler observation space for faster training.

    Observation:
        - Number of each resource type visible (3 values)
        - Inventory count (1 value)
        - Distance to nearest resource (1 value)

    Total: 5 features
    """

    def __init__(self, model_path: str = 'model.pt', max_steps: int = 500):
        super().__init__()

        self.detector = FiberDetection(model_path=model_path)
        self.ocr_reader = GatherState()

        self.max_steps = max_steps
        self.current_step = 0
        self.last_inventory_count = 0
        self.total_gathered = 0

        # Simplified action space
        self.action_space = gym.spaces.Discrete(4)  # Click cotton/flax/hemp, or wait

        # Simplified observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=10.0,  # Max 10 of each type
            shape=(5,),
            dtype=np.float32
        )

        print("SimplifiedAlbionEnv initialized")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.last_inventory_count = self._get_inventory_count()
        self.total_gathered = 0

        obs = self._get_observation()
        info = {'step': 0, 'inventory': self.last_inventory_count}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1

        # Take screenshot
        screenshot = np.array(pyautogui.screenshot())

        # Execute action
        action_success = self._execute_action(action, screenshot)
        time.sleep(0.5)

        # Compute reward
        reward = -0.1  # Time penalty

        current_inventory = self._get_inventory_count()
        items_gathered = current_inventory - self.last_inventory_count

        if items_gathered > 0:
            reward += 5.0 * items_gathered  # Good reward for gathering
            self.total_gathered += items_gathered

        self.last_inventory_count = current_inventory

        # Episode ends if inventory full
        terminated = current_inventory >= 9
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += 50.0  # Bonus for completing episode

        obs = self._get_observation()
        info = {
            'step': self.current_step,
            'inventory': current_inventory,
            'total_gathered': self.total_gathered
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int, screenshot: np.ndarray) -> bool:
        if action == 3:  # Wait
            return True

        # Get detections
        detections = self.detector.detect(screenshot)

        # Filter by class
        target_detections = [d for d in detections if d['class'] == action]

        if not target_detections:
            return False

        # Click nearest
        h, w = screenshot.shape[:2]
        center_x, center_y = w / 2, h / 2

        nearest = min(
            target_detections,
            key=lambda d: np.sqrt(
                (d['x'] + d['width']/2 - center_x)**2 +
                (d['y'] + d['height']/2 - center_y)**2
            )
        )

        click_x = int(nearest['x'] + nearest['width'] / 2)
        click_y = int(nearest['y'] + nearest['height'] / 2)

        pyautogui.moveTo(click_x, click_y, duration=0.2)
        pyautogui.click()

        return True

    def _get_observation(self) -> np.ndarray:
        """
        Simplified observation:
        [cotton_count, flax_count, hemp_count, inventory_count, nearest_distance]
        """
        screenshot = np.array(pyautogui.screenshot())
        detections = self.detector.detect(screenshot)

        # Count each type
        counts = np.zeros(3, dtype=np.float32)
        for det in detections:
            counts[det['class']] += 1

        # Inventory
        inventory = float(self.last_inventory_count)

        # Distance to nearest resource (normalized)
        if detections:
            h, w = screenshot.shape[:2]
            center_x, center_y = w / 2, h / 2

            distances = [
                np.sqrt((d['x'] + d['width']/2 - center_x)**2 +
                       (d['y'] + d['height']/2 - center_y)**2)
                for d in detections
            ]
            nearest_dist = min(distances) / w  # Normalize by screen width
        else:
            nearest_dist = 1.0  # Max distance if nothing detected

        obs = np.array([
            counts[0],  # Cotton count
            counts[1],  # Flax count
            counts[2],  # Hemp count
            inventory,  # Inventory count
            nearest_dist  # Nearest resource distance
        ], dtype=np.float32)

        return obs

    def _get_inventory_count(self) -> int:
        try:
            count_str = self.ocr_reader.get_gather_count()
            if '/' in count_str:
                return int(count_str.split('/')[0])
        except:
            pass
        return self.last_inventory_count


if __name__ == '__main__':
    # Test environment
    print("Testing AlbionGatherEnv...")

    env = SimplifiedAlbionEnv()

    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Test random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Info: {info}")

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\nEnvironment test completed!")
