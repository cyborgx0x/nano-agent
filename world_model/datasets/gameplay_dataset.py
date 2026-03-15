"""
Gameplay data collection and dataset for training world model.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from mss import mss
from pynput import mouse, keyboard


class GameplayCollector:
    """
    Collects gameplay data: screenshots + actions.
    Records at specified FPS for world model training.
    """

    def __init__(
        self,
        save_dir,
        fps=10,
        region=None,  # {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        max_duration=3600,  # Max 1 hour per session
    ):
        """
        Args:
            save_dir: Directory to save collected data
            fps: Frames per second to capture
            region: Screen region to capture (None = full screen)
            max_duration: Maximum recording duration in seconds
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.region = region
        self.max_duration = max_duration

        # Data storage
        self.frames = []
        self.actions = []
        self.timestamps = []
        self.mouse_positions = []

        # Current state
        self.current_mouse_pos = (0, 0)
        self.current_keys = set()
        self.last_click = None
        self.is_recording = False

        # Screen capture
        self.sct = mss()

        # Input listeners
        self.mouse_listener = None
        self.keyboard_listener = None

    def _on_mouse_move(self, x, y):
        """Mouse move callback."""
        self.current_mouse_pos = (x, y)

    def _on_mouse_click(self, x, y, button, pressed):
        """Mouse click callback."""
        if pressed:
            self.last_click = {
                'button': button.name,
                'pos': (x, y),
                'time': time.time()
            }

    def _on_key_press(self, key):
        """Key press callback."""
        try:
            key_name = key.char
        except AttributeError:
            key_name = key.name
        self.current_keys.add(key_name)

    def _on_key_release(self, key):
        """Key release callback."""
        try:
            key_name = key.char
        except AttributeError:
            key_name = key.name
        self.current_keys.discard(key_name)

    def start_recording(self, session_name=None):
        """Start recording gameplay."""
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_name = session_name
        self.session_dir = self.save_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Reset data
        self.frames = []
        self.actions = []
        self.timestamps = []
        self.mouse_positions = []
        self.is_recording = True

        # Start input listeners
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

        print(f"Started recording session: {session_name}")
        print(f"FPS: {self.fps}, Max duration: {self.max_duration}s")

        # Recording loop
        start_time = time.time()
        frame_count = 0

        try:
            while self.is_recording:
                loop_start = time.time()

                # Capture screenshot
                if self.region:
                    screenshot = self.sct.grab(self.region)
                else:
                    screenshot = self.sct.grab(self.sct.monitors[1])

                # Convert to numpy array
                frame = np.array(screenshot)[:, :, :3]  # RGB

                # Record current state
                action = self._get_current_action()
                timestamp = time.time() - start_time

                self.frames.append(frame)
                self.actions.append(action)
                self.timestamps.append(timestamp)
                self.mouse_positions.append(self.current_mouse_pos)

                frame_count += 1

                # Check duration limit
                if timestamp >= self.max_duration:
                    print(f"Reached max duration ({self.max_duration}s)")
                    break

                # Maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")

        finally:
            self.stop_recording()

    def _get_current_action(self):
        """Get current action state."""
        action = {
            'mouse_pos': self.current_mouse_pos,
            'click': self.last_click,
            'keys': list(self.current_keys),
        }

        # Reset click (one-time event)
        if self.last_click:
            # Check if click happened recently (within frame interval)
            if time.time() - self.last_click['time'] < self.frame_interval:
                pass  # Keep it
            else:
                self.last_click = None
                action['click'] = None

        return action

    def stop_recording(self):
        """Stop recording and save data."""
        self.is_recording = False

        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        print(f"\nRecorded {len(self.frames)} frames")

        # Save data
        self._save_session()

    def _save_session(self):
        """Save recorded session to disk."""
        print(f"Saving session to {self.session_dir}...")

        # Save video
        video_path = self.session_dir / "gameplay.mp4"
        self._save_video(video_path)

        # Save actions as JSON
        actions_path = self.session_dir / "actions.json"
        actions_data = {
            'actions': self.actions,
            'timestamps': self.timestamps,
            'mouse_positions': self.mouse_positions,
            'fps': self.fps,
            'num_frames': len(self.frames),
        }
        with open(actions_path, 'w') as f:
            json.dump(actions_data, f, indent=2, default=str)

        # Save metadata
        metadata_path = self.session_dir / "metadata.json"
        metadata = {
            'session_name': self.session_name,
            'num_frames': len(self.frames),
            'duration': self.timestamps[-1] if self.timestamps else 0,
            'fps': self.fps,
            'resolution': self.frames[0].shape[:2] if self.frames else None,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(self.frames)} frames to {video_path}")
        print(f"Saved actions to {actions_path}")

    def _save_video(self, video_path):
        """Save frames as video file."""
        if not self.frames:
            return

        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()


class GameplayDataset(Dataset):
    """
    PyTorch dataset for gameplay sequences.
    Returns (frames, actions) sequences for world model training.
    """

    def __init__(
        self,
        data_dir,
        sequence_length=8,
        transform=None,
        action_transform=None,
    ):
        """
        Args:
            data_dir: Directory containing gameplay sessions
            sequence_length: Number of frames per sequence
            transform: Transform for frames
            action_transform: Transform for actions
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.action_transform = action_transform

        # Load all sessions
        self.sessions = []
        self._load_sessions()

    def _load_sessions(self):
        """Load all gameplay sessions."""
        for session_dir in self.data_dir.iterdir():
            if not session_dir.is_dir():
                continue

            metadata_path = session_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Only load sessions with enough frames
            if metadata['num_frames'] >= self.sequence_length:
                self.sessions.append({
                    'dir': session_dir,
                    'metadata': metadata,
                })

        print(f"Loaded {len(self.sessions)} gameplay sessions")

    def __len__(self):
        """Total number of sequences."""
        total = 0
        for session in self.sessions:
            num_frames = session['metadata']['num_frames']
            total += max(0, num_frames - self.sequence_length + 1)
        return total

    def __getitem__(self, idx):
        """Get a sequence of (frames, actions)."""
        # Find which session and offset
        session_idx = 0
        offset = idx

        for session_idx, session in enumerate(self.sessions):
            num_frames = session['metadata']['num_frames']
            num_sequences = max(0, num_frames - self.sequence_length + 1)

            if offset < num_sequences:
                break
            offset -= num_sequences

        # Load video frames
        video_path = session['dir'] / "gameplay.mp4"
        frames = self._load_video_sequence(video_path, offset, self.sequence_length)

        # Load actions
        actions_path = session['dir'] / "actions.json"
        with open(actions_path) as f:
            actions_data = json.load(f)

        actions = actions_data['actions'][offset:offset + self.sequence_length]

        # Apply transforms
        if self.transform:
            frames = self.transform(frames)

        if self.action_transform:
            actions = self.action_transform(actions)

        return frames, actions

    def _load_video_sequence(self, video_path, start_frame, num_frames):
        """Load a sequence of frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        return np.array(frames)


# Example usage
if __name__ == "__main__":
    # Collect gameplay
    collector = GameplayCollector(
        save_dir="./gameplay_data",
        fps=10,
        max_duration=60,  # 1 minute test
    )

    print("Press Ctrl+C to stop recording...")
    collector.start_recording()
