"""
SLAM Navigator for Albion Online - Multi-Zone Navigation
Builds persistent map across multiple zones with metadata tagging

Features:
- Visual odometry for position tracking
- Landmark detection and mapping
- Multi-zone graph construction
- Metadata tagging (resources, danger zones, NPCs)
- Persistent map storage (save/load)
"""

import cv2
import numpy as np
import json
import pickle
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import pyautogui


@dataclass
class MapObject:
    """Metadata for detected objects on the map."""
    object_type: str  # 'fiber', 'portal', 'npc', 'danger_zone', etc.
    position: Tuple[int, int]  # (x, y) on global map
    zone_id: str  # Which zone it's in
    confidence: float
    timestamp: str
    properties: Dict  # Extra metadata (e.g., {'resource_tier': 5, 'occupied': False})


@dataclass
class Zone:
    """Represents a game zone/map."""
    zone_id: str
    name: str
    map_image: np.ndarray  # Visual map of the zone
    objects: List[MapObject]  # All detected objects in this zone
    connections: Dict[str, Tuple[int, int]]  # {neighbor_zone_id: (portal_x, portal_y)}
    bounds: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y) in global coords


class SLAMNavigator:
    """
    Multi-zone SLAM navigation system.
    Builds and maintains a persistent world map across game sessions.
    """

    def __init__(self,
                 fiber_detector=None,  # Your YOLO fiber detector
                 landmark_detector=None,  # YOLO for landmarks
                 map_size=(10000, 10000),  # Global map size
                 save_path='./slam_maps'):

        self.fiber_detector = fiber_detector
        self.landmark_detector = landmark_detector
        self.save_path = save_path

        # Global map (stitched from all zones)
        self.global_map = np.zeros((*map_size, 3), dtype=np.uint8)

        # Current state
        self.current_position = (map_size[0] // 2, map_size[1] // 2)  # Global coords
        self.current_zone = "unknown_zone_0"
        self.heading = 0.0  # Orientation in radians

        # Zone management
        self.zones: Dict[str, Zone] = {}
        self.zone_graph = {}  # {zone_id: [neighbor_zone_ids]}

        # Object database (all detected objects across all zones)
        self.object_database: Dict[str, List[MapObject]] = defaultdict(list)

        # Visual odometry
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_screenshot = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_features = None  # For optical flow
        self.prev_gray = None

        # State tracking
        self.in_loading_screen = False  # Debounce loading screen detection
        self.popup_active = False  # Track if popup (skill tree, inventory) is open

        # Calibration (adjust based on your game)
        self.pixels_per_meter = 50  # Game units to map pixels
        self.movement_scale = 2.0  # Visual displacement to actual movement

    def create_world_mask(self, screenshot):
        """
        Create mask that ONLY tracks world features (no UI, no character)

        Returns:
            mask: 255 = track, 0 = ignore
        """
        height, width = screenshot.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255

        # Mask out UI regions (based on 1280x720 resolution)
        # Top bar: Health/mana/buffs
        mask[0:100, :] = 0

        # Top-left: Minimap + stats
        mask[0:200, 0:250] = 0

        # Bottom-left: Chat panel
        mask[400:720, 0:400] = 0

        # Bottom strip: Skill bar, inventory
        mask[600:720, :] = 0

        # Right side: All panels
        mask[:, int(width * 0.75):width] = 0  # Right 25%

        # Mask out CHARACTER in center (dead zone)
        center_x, center_y = width // 2, height // 2
        character_radius = 60
        dead_zone_radius = 120

        y, x = np.ogrid[:height, :width]
        dead_zone_mask = (x - center_x)**2 + (y - center_y)**2 <= dead_zone_radius**2
        mask[dead_zone_mask] = 0

        return mask

    def detect_zone_transition(self, screenshot):
        """
        Detect if player crossed into a new zone (portal, gate, etc.).

        Returns:
            (is_transition, new_zone_name) or (False, None)
        """
        # Method 1: Look for loading screen (black screen) with debouncing
        is_black = np.mean(screenshot) < 10

        if is_black and not self.in_loading_screen:
            # Just entered loading screen
            self.in_loading_screen = True
            return (False, None)  # Don't create "loading" zone
        elif not is_black and self.in_loading_screen:
            # Just exited loading screen - NEW ZONE!
            self.in_loading_screen = False
            new_zone_id = f"zone_{len(self.zones)}"
            return (True, new_zone_id)
        elif self.in_loading_screen:
            # Still in loading screen, don't spam
            return (False, None)

        # Method 2: Template matching for zone name UI
        # (You'd need to OCR the zone name that appears on screen)

        # Method 3: Detect portals/gates and check if player passed through
        if self.landmark_detector:
            results = self.landmark_detector(screenshot)
            portals = [r for r in results if 'portal' in r.names[int(r.cls)]]

            if len(portals) > 0 and self._near_portal(portals[0]):
                # Player is near portal, likely transitioning
                return (True, f"zone_from_portal_{len(self.zones)}")

        return (False, None)

    def _near_portal(self, portal_detection, threshold=50):
        """Check if player (screen center) is near a portal."""
        portal_center = (
            (portal_detection.bbox[0] + portal_detection.bbox[2]) / 2,
            (portal_detection.bbox[1] + portal_detection.bbox[3]) / 2
        )
        screen_center = (1920 / 2, 1080 / 2)
        distance = np.sqrt(
            (portal_center[0] - screen_center[0])**2 +
            (portal_center[1] - screen_center[1])**2
        )
        return distance < threshold

    def detect_popup(self, screenshot):
        """
        Detect if a popup (skill tree, inventory, etc.) is open.
        Popup = sudden massive change in screen content.

        Returns:
            bool: True if popup detected
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        mask = self.create_world_mask(screenshot)

        # Detect features in masked region
        keypoints, _ = self.orb.detectAndCompute(gray, mask)

        if self.prev_keypoints is None:
            return False

        # If we suddenly lose >70% of features, it's probably a popup
        feature_loss = 1 - (len(keypoints) / max(len(self.prev_keypoints), 1))

        if feature_loss > 0.7:
            return True

        return False

    def update_position(self, screenshot):
        """
        Update position using optical flow with UI masking.

        Returns:
            (dx, dy, dtheta): Change in position and rotation
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        # Check for popups (skill tree, inventory, etc.)
        if self.detect_popup(screenshot):
            if not self.popup_active:
                print("Popup detected - pausing tracking")
                self.popup_active = True
            return (0, 0, 0)  # Don't track movement during popups
        else:
            if self.popup_active:
                print("Popup closed - resuming tracking")
                self.popup_active = False

        # Create mask to ignore UI and character
        mask = self.create_world_mask(screenshot)

        # Detect features ONLY in world region
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

        if self.prev_gray is None:
            # First frame - initialize
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors

            # Extract good features for optical flow
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                              minDistance=10, mask=mask)
            self.prev_features = corners
            return (0, 0, 0)

        # Use Lucas-Kanade optical flow (better for camera-follow games)
        if self.prev_features is not None and len(self.prev_features) > 0:
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_features, None
            )

            if new_features is not None and np.sum(status) > 10:
                # Filter good matches
                good_old = self.prev_features[status == 1]
                good_new = new_features[status == 1]

                # Calculate displacement (INVERSE - world moves opposite to character)
                displacement = np.median(good_new - good_old, axis=0)
                dx, dy = -displacement * self.movement_scale  # NEGATIVE!

                # Update features for next frame
                self.prev_features = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                                              qualityLevel=0.01,
                                                              minDistance=10, mask=mask)
            else:
                # Lost tracking, reinitialize
                self.prev_features = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                                              qualityLevel=0.01,
                                                              minDistance=10, mask=mask)
                dx, dy = 0, 0
        else:
            dx, dy = 0, 0

        # Update stored state for next frame
        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return (int(dx), int(dy), 0)

    def detect_and_tag_objects(self, screenshot):
        """
        Detect objects in current view and add to database with metadata.

        Returns:
            List[MapObject]: Detected objects with positions
        """
        detected_objects = []

        # Detect fiber resources
        if self.fiber_detector:
            results = self.fiber_detector(screenshot, verbose=False)

            # YOLOv8 returns a list of Results objects
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # Iterate through detected boxes
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Convert screen coords to global map coords
                    x1, y1, x2, y2 = box
                    screen_x = (x1 + x2) / 2
                    screen_y = (y1 + y2) / 2

                    # Get screen dimensions (current frame)
                    h, w = screenshot.shape[:2]

                    # Offset from screen center to world position
                    offset_x = screen_x - w / 2
                    offset_y = screen_y - h / 2

                    global_x = self.current_position[0] + int(offset_x)
                    global_y = self.current_position[1] + int(offset_y)

                    obj = MapObject(
                        object_type=result.names[int(cls_id)],
                        position=(global_x, global_y),
                        zone_id=self.current_zone,
                        confidence=float(conf),
                        timestamp=datetime.now().isoformat(),
                        properties={
                            'resource_tier': self._infer_tier(result.names[int(cls_id)]),
                            'occupied': False  # Could detect if another player is gathering
                        }
                    )
                    detected_objects.append(obj)

        # Detect landmarks
        if self.landmark_detector:
            landmark_results = self.landmark_detector(screenshot)
            for result in landmark_results:
                screen_x = (result.bbox[0] + result.bbox[2]) / 2
                screen_y = (result.bbox[1] + result.bbox[3]) / 2

                offset_x = screen_x - 1920 / 2
                offset_y = screen_y - 1080 / 2

                global_x = self.current_position[0] + int(offset_x)
                global_y = self.current_position[1] + int(offset_y)

                obj = MapObject(
                    object_type=result.names[int(result.cls)],
                    position=(global_x, global_y),
                    zone_id=self.current_zone,
                    confidence=float(result.conf),
                    timestamp=datetime.now().isoformat(),
                    properties={
                        'landmark_type': result.names[int(result.cls)]
                    }
                )
                detected_objects.append(obj)

        # Add to database
        for obj in detected_objects:
            self.object_database[obj.object_type].append(obj)

        return detected_objects

    def _infer_tier(self, resource_name):
        """Infer resource tier from name."""
        if 'exceptional' in resource_name.lower():
            return 8
        elif 'rare' in resource_name.lower():
            return 6
        elif 'uncommon' in resource_name.lower():
            return 4
        else:
            return 2

    def update_map(self, screenshot):
        """
        Main SLAM update loop.
        Call this every frame while playing.
        """
        # 1. Check for zone transition
        is_transition, new_zone = self.detect_zone_transition(screenshot)

        if is_transition:
            print(f"Zone transition detected: {self.current_zone} -> {new_zone}")
            self._handle_zone_transition(new_zone)
            return

        # 2. Update position via visual odometry
        dx, dy, dtheta = self.update_position(screenshot)

        self.current_position = (
            self.current_position[0] + dx,
            self.current_position[1] - dy  # Y is inverted
        )
        self.heading += dtheta

        # 3. Detect and tag objects in current view
        detected_objects = self.detect_and_tag_objects(screenshot)

        # 4. Update zone map
        self._update_zone_map(screenshot)

        # 5. Update global map
        self._stitch_to_global_map(screenshot)

        return {
            'position': self.current_position,
            'zone': self.current_zone,
            'heading': self.heading,
            'detected_objects': detected_objects
        }

    def _handle_zone_transition(self, new_zone):
        """Handle entering a new zone."""
        # Save previous zone
        if self.current_zone in self.zones:
            old_zone = self.zones[self.current_zone]
            # Record connection
            if new_zone not in old_zone.connections:
                old_zone.connections[new_zone] = self.current_position

        # Create new zone if needed
        if new_zone not in self.zones:
            self.zones[new_zone] = Zone(
                zone_id=new_zone,
                name=new_zone,
                map_image=np.zeros((2000, 2000, 3), dtype=np.uint8),
                objects=[],
                connections={self.current_zone: self.current_position},
                bounds=(0, 0, 2000, 2000)
            )
            print(f"Created new zone: {new_zone}")

        # Update zone graph
        if self.current_zone not in self.zone_graph:
            self.zone_graph[self.current_zone] = []
        if new_zone not in self.zone_graph[self.current_zone]:
            self.zone_graph[self.current_zone].append(new_zone)

        # Switch to new zone
        self.current_zone = new_zone

    def _update_zone_map(self, screenshot):
        """Update the current zone's local map."""
        if self.current_zone not in self.zones:
            return

        zone = self.zones[self.current_zone]

        # Resize screenshot to map scale
        view_size = 200
        view_resized = cv2.resize(screenshot, (view_size, view_size))

        # Local position within zone
        local_x = self.current_position[0] % 2000
        local_y = self.current_position[1] % 2000

        # Paste onto zone map
        x1, y1 = max(0, local_x - view_size // 2), max(0, local_y - view_size // 2)
        x2, y2 = min(2000, x1 + view_size), min(2000, y1 + view_size)

        # Blend with existing map
        alpha = 0.6
        zone.map_image[y1:y2, x1:x2] = (
            alpha * view_resized[:y2-y1, :x2-x1] +
            (1 - alpha) * zone.map_image[y1:y2, x1:x2]
        ).astype(np.uint8)

    def _stitch_to_global_map(self, screenshot):
        """Stitch current view into global map."""
        view_size = 200
        view_resized = cv2.resize(screenshot, (view_size, view_size))

        x, y = self.current_position
        x1, y1 = x - view_size // 2, y - view_size // 2
        x2, y2 = x1 + view_size, y1 + view_size

        # Bounds check
        if x1 < 0 or y1 < 0 or x2 >= self.global_map.shape[1] or y2 >= self.global_map.shape[0]:
            return

        # Blend
        alpha = 0.7
        self.global_map[y1:y2, x1:x2] = (
            alpha * view_resized +
            (1 - alpha) * self.global_map[y1:y2, x1:x2]
        ).astype(np.uint8)

    def find_path_multi_zone(self, target_zone, target_position=None):
        """
        Find path from current zone to target zone.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: [(zone_id, waypoint), ...]
        """
        # BFS to find zone path
        queue = deque([(self.current_zone, [self.current_zone])])
        visited = {self.current_zone}

        while queue:
            current, path = queue.popleft()

            if current == target_zone:
                # Found zone path!
                waypoints = []
                for i, zone_id in enumerate(path):
                    if i < len(path) - 1:
                        next_zone = path[i + 1]
                        # Get portal position
                        portal_pos = self.zones[zone_id].connections.get(next_zone)
                        waypoints.append((zone_id, portal_pos))
                    else:
                        # Final zone
                        waypoints.append((zone_id, target_position or self.current_position))

                return waypoints

            # Explore neighbors
            for neighbor in self.zone_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    def query_objects(self, object_type, zone_id=None, radius=None):
        """
        Query object database.

        Args:
            object_type: Type of object ('fiber', 'portal', etc.)
            zone_id: Filter by zone (None = all zones)
            radius: Search radius from current position

        Returns:
            List[MapObject]: Matching objects
        """
        objects = self.object_database.get(object_type, [])

        if zone_id:
            objects = [obj for obj in objects if obj.zone_id == zone_id]

        if radius:
            objects = [
                obj for obj in objects
                if np.sqrt(
                    (obj.position[0] - self.current_position[0])**2 +
                    (obj.position[1] - self.current_position[1])**2
                ) <= radius
            ]

        return objects

    def render_map_with_objects(self, trajectory=None):
        """
        Create a visual map showing trajectory and all detected objects.

        Args:
            trajectory: List of (frame_id, x, y, heading) tuples from test

        Returns:
            np.ndarray: Rendered map image
        """
        # Find bounds of traveled area + objects
        all_x = []
        all_y = []

        # Add trajectory points
        if trajectory:
            all_x.extend([p[1] for p in trajectory])
            all_y.extend([p[2] for p in trajectory])

        # Add object positions
        for obj_type, objects in self.object_database.items():
            for obj in objects:
                all_x.append(obj.position[0])
                all_y.append(obj.position[1])

        if not all_x:
            return None

        # Calculate bounds with padding
        min_x, max_x = int(min(all_x)) - 200, int(max(all_x)) + 200
        min_y, max_y = int(min(all_y)) - 200, int(max(all_y)) + 200

        width = max_x - min_x
        height = max_y - min_y

        # Create map canvas (white background)
        map_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw trajectory if available
        if trajectory and len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                x1 = int(trajectory[i][1] - min_x)
                y1 = int(trajectory[i][2] - min_y)
                x2 = int(trajectory[i+1][1] - min_x)
                y2 = int(trajectory[i+1][2] - min_y)

                # Draw path line (light blue)
                cv2.line(map_img, (x1, y1), (x2, y2), (200, 200, 100), 2)

            # Mark start (green) and end (red)
            start_x = int(trajectory[0][1] - min_x)
            start_y = int(trajectory[0][2] - min_y)
            end_x = int(trajectory[-1][1] - min_x)
            end_y = int(trajectory[-1][2] - min_y)

            cv2.circle(map_img, (start_x, start_y), 10, (0, 255, 0), -1)
            cv2.circle(map_img, (end_x, end_y), 10, (0, 0, 255), -1)

        # Draw detected objects
        object_colors = {
            'fiber': (0, 165, 255),  # Orange for fiber
            'portal': (255, 0, 255),  # Magenta for portals
            'landmark': (255, 255, 0)  # Cyan for landmarks
        }

        for obj_type, objects in self.object_database.items():
            color = object_colors.get(obj_type, (128, 128, 128))

            for obj in objects:
                x = int(obj.position[0] - min_x)
                y = int(obj.position[1] - min_y)

                # Draw object marker
                cv2.circle(map_img, (x, y), 5, color, -1)
                cv2.circle(map_img, (x, y), 7, (0, 0, 0), 1)

        # Add legend
        legend_y = 30
        cv2.putText(map_img, "Legend:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        legend_y += 30

        cv2.circle(map_img, (20, legend_y), 5, (0, 255, 0), -1)
        cv2.putText(map_img, "Start", (35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        legend_y += 25

        cv2.circle(map_img, (20, legend_y), 5, (0, 0, 255), -1)
        cv2.putText(map_img, "End", (35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        legend_y += 25

        for obj_type, color in object_colors.items():
            count = len(self.object_database.get(obj_type, []))
            if count > 0:
                cv2.circle(map_img, (20, legend_y), 5, color, -1)
                cv2.putText(map_img, f"{obj_type.title()} ({count})", (35, legend_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                legend_y += 25

        return map_img

    def save_map(self, filename=None):
        """Save map to disk for persistence."""
        if filename is None:
            filename = f"{self.save_path}/slam_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        data = {
            'global_map': self.global_map,
            'zones': {k: asdict(v) for k, v in self.zones.items()},
            'zone_graph': self.zone_graph,
            'object_database': {k: [asdict(obj) for obj in v] for k, v in self.object_database.items()},
            'current_position': self.current_position,
            'current_zone': self.current_zone
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Map saved to {filename}")

        # Also save visual map
        cv2.imwrite(filename.replace('.pkl', '_global.png'), self.global_map)

    def load_map(self, filename):
        """Load previously saved map."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.global_map = data['global_map']
        self.zones = {k: Zone(**v) for k, v in data['zones'].items()}
        self.zone_graph = data['zone_graph']
        self.object_database = defaultdict(list)
        for k, v in data['object_database'].items():
            self.object_database[k] = [MapObject(**obj) for obj in v]
        self.current_position = data['current_position']
        self.current_zone = data['current_zone']

        print(f"Map loaded from {filename}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main_slam_loop():
    """Example main loop using SLAM navigator."""
    from ultralytics import YOLO

    # Load your detectors
    fiber_detector = YOLO('/home/lucas/Documents/ai-research/nano-agent/model.pt')
    landmark_detector = None  # Train this later

    # Initialize SLAM
    slam = SLAMNavigator(
        fiber_detector=fiber_detector,
        landmark_detector=landmark_detector
    )

    # Try to load previous map
    try:
        slam.load_map('slam_maps/latest.pkl')
        print("Loaded existing map!")
    except:
        print("Starting with fresh map")

    frame_count = 0
    save_interval = 100  # Save every 100 frames

    while True:
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Update SLAM
        status = slam.update_map(screenshot)

        # Print status
        if frame_count % 10 == 0:
            print(f"Position: {status['position']}, Zone: {status['zone']}")
            print(f"Objects detected: {len(status['detected_objects'])}")

        # Save periodically
        if frame_count % save_interval == 0:
            slam.save_map('slam_maps/latest.pkl')

        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final save
    slam.save_map('slam_maps/final.pkl')

    # Show discovered map
    cv2.imshow('Discovered World Map', slam.global_map)
    cv2.waitKey(0)


if __name__ == "__main__":
    print("SLAM Navigator for Albion Online")
    print("=" * 60)
    print("This will build a persistent map as you play")
    print("Press 'q' to quit and save")
    print("=" * 60)

    main_slam_loop()
