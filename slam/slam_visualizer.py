"""
SLAM Map Visualizer
Interactive tool to view and query your discovered map
"""

import json

import cv2
import numpy as np

from .slam_navigator import MapObject, SLAMNavigator


class SLAMVisualizer:
    """
    Interactive visualizer for SLAM maps.
    Shows discovered areas, objects, and paths.
    """

    def __init__(self, slam_navigator):
        self.slam = slam_navigator
        self.window_name = "SLAM Map Viewer"
        self.zoom = 1.0
        self.offset = [0, 0]
        self.show_objects = True
        self.selected_object_type = None

    def draw_map(self):
        """Render the map with overlays."""
        # Start with global map
        display = self.slam.global_map.copy()

        # Draw current position
        cv2.circle(
            display, self.slam.current_position, 10, (0, 255, 0), -1  # Green for player
        )

        # Draw heading indicator
        heading_length = 30
        end_x = int(
            self.slam.current_position[0] + heading_length * np.cos(self.slam.heading)
        )
        end_y = int(
            self.slam.current_position[1] + heading_length * np.sin(self.slam.heading)
        )
        cv2.arrowedLine(
            display, self.slam.current_position, (end_x, end_y), (0, 255, 0), 2
        )

        # Draw zone boundaries
        for zone_id, zone in self.slam.zones.items():
            color = (
                (100, 100, 255)
                if zone_id == self.slam.current_zone
                else (100, 100, 100)
            )
            # Draw zone name
            # (positions would need to be stored in zone data)

        # Draw objects
        if self.show_objects:
            self._draw_objects(display)

        # Draw zone connections (portals)
        self._draw_zone_graph(display)

        # Add minimap
        minimap = self._create_minimap()
        display[10 : 10 + minimap.shape[0], 10 : 10 + minimap.shape[1]] = minimap

        # Add legend
        self._draw_legend(display)

        return display

    def _draw_objects(self, display):
        """Draw all detected objects on the map."""
        colors = {
            "cotton": (200, 200, 100),
            "flax": (150, 200, 150),
            "hemp": (100, 200, 100),
            "exceptional_hemp": (50, 255, 50),
            "rare_hemp": (100, 255, 100),
            "portal": (255, 100, 255),
            "danger_zone": (0, 0, 255),
        }

        for object_type, objects in self.slam.object_database.items():
            if self.selected_object_type and object_type != self.selected_object_type:
                continue

            color = colors.get(object_type, (255, 255, 255))

            for obj in objects:
                # Only show objects in discovered areas
                cv2.circle(display, obj.position, 5, color, -1)

                # Draw metadata on hover (would need mouse callback)
                # For now, just show count
                cv2.putText(
                    display,
                    f"{obj.object_type[:3]}",
                    (obj.position[0] + 7, obj.position[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )

    def _draw_zone_graph(self, display):
        """Draw connections between zones (portals/gates)."""
        for zone_id, zone in self.slam.zones.items():
            for neighbor_id, portal_pos in zone.connections.items():
                if neighbor_id in self.slam.zones:
                    neighbor_portal = self.slam.zones[neighbor_id].connections.get(
                        zone_id
                    )
                    if neighbor_portal:
                        # Draw line between portals
                        cv2.line(
                            display,
                            portal_pos,
                            neighbor_portal,
                            (255, 200, 0),  # Orange for portals
                            2,
                        )

                        # Draw portal markers
                        cv2.circle(display, portal_pos, 8, (255, 200, 0), -1)
                        cv2.circle(display, neighbor_portal, 8, (255, 200, 0), -1)

    def _create_minimap(self):
        """Create a small overview minimap."""
        minimap_size = 200
        minimap = cv2.resize(self.slam.global_map, (minimap_size, minimap_size))

        # Add current position indicator
        scale_x = minimap_size / self.slam.global_map.shape[1]
        scale_y = minimap_size / self.slam.global_map.shape[0]

        mini_x = int(self.slam.current_position[0] * scale_x)
        mini_y = int(self.slam.current_position[1] * scale_y)

        cv2.circle(minimap, (mini_x, mini_y), 3, (0, 255, 0), -1)

        return minimap

    def _draw_legend(self, display):
        """Draw legend explaining colors."""
        legend_items = [
            ("Player", (0, 255, 0)),
            ("Cotton", (200, 200, 100)),
            ("Flax", (150, 200, 150)),
            ("Hemp", (100, 200, 100)),
            ("Portal", (255, 200, 0)),
            ("Danger", (0, 0, 255)),
        ]

        legend_x = display.shape[1] - 150
        legend_y = 20

        for i, (label, color) in enumerate(legend_items):
            y = legend_y + i * 25
            cv2.rectangle(display, (legend_x, y), (legend_x + 20, y + 15), color, -1)
            cv2.putText(
                display,
                label,
                (legend_x + 25, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    def show(self):
        """Display the map interactively."""
        cv2.namedWindow(self.window_name)

        while True:
            display = self.draw_map()

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(100) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("o"):
                # Toggle object display
                self.show_objects = not self.show_objects
            elif key == ord("s"):
                # Save current view
                cv2.imwrite("slam_view.png", display)
                print("View saved to slam_view.png")
            elif key == ord("+"):
                self.zoom *= 1.2
            elif key == ord("-"):
                self.zoom /= 1.2

        cv2.destroyAllWindows()


def query_nearby_resources():
    """Example: Query resources near current position."""
    from ultralytics import YOLO

    from .slam_navigator import SLAMNavigator

    # Load SLAM map
    slam = SLAMNavigator(
        fiber_detector=YOLO("/home/lucas/Documents/ai-research/nano-agent/model.pt")
    )
    slam.load_map("slam_maps/latest.pkl")

    # Find all hemp within 500 pixels
    hemp_nodes = slam.query_objects("hemp", radius=500)

    print(f"Found {len(hemp_nodes)} hemp nodes nearby:")
    for node in hemp_nodes:
        distance = np.sqrt(
            (node.position[0] - slam.current_position[0]) ** 2
            + (node.position[1] - slam.current_position[1]) ** 2
        )
        print(f"  - {node.object_type} at {node.position}, distance: {distance:.0f}px")
        print(f"    Tier: {node.properties.get('resource_tier', 'unknown')}")
        print(f"    Zone: {node.zone_id}")


def plan_multi_zone_route():
    """Example: Plan route across multiple zones."""
    from ultralytics import YOLO

    from .slam_navigator import SLAMNavigator

    slam = SLAMNavigator(
        fiber_detector=YOLO("/home/lucas/Documents/ai-research/nano-agent/model.pt")
    )
    slam.load_map("slam_maps/latest.pkl")

    # I want to go to a zone 4 maps away
    target_zone = "Highland_Zone_North"
    target_position = (7500, 4500)  # Specific location in that zone

    # Find path
    path = slam.find_path_multi_zone(target_zone, target_position)

    if path:
        print(f"Route from {slam.current_zone} to {target_zone}:")
        for i, (zone, waypoint) in enumerate(path):
            print(f"  {i+1}. Go to {zone} at position {waypoint}")
    else:
        print("No path found! Need to explore more.")


def find_best_resource_zone():
    """Example: Find zone with most resources."""
    from collections import Counter

    from ultralytics import YOLO

    from .slam_navigator import SLAMNavigator

    slam = SLAMNavigator(
        fiber_detector=YOLO("/home/lucas/Documents/ai-research/nano-agent/model.pt")
    )
    slam.load_map("slam_maps/latest.pkl")

    # Count resources per zone
    zone_resources = Counter()

    for resource_type in ["hemp", "flax", "cotton"]:
        for obj in slam.object_database.get(resource_type, []):
            # Only count high-tier resources
            if obj.properties.get("resource_tier", 0) >= 6:
                zone_resources[obj.zone_id] += 1

    # Find best zone
    best_zone = zone_resources.most_common(1)[0] if zone_resources else None

    if best_zone:
        print(
            f"Best farming zone: {best_zone[0]} with {best_zone[1]} high-tier resources"
        )

        # Plan route there
        path = slam.find_path_multi_zone(best_zone[0])
        if path:
            print(f"Route: {' -> '.join([z for z, _ in path])}")
    else:
        print("No resource data yet. Keep exploring!")


if __name__ == "__main__":
    print("SLAM Visualizer Examples")
    print("=" * 60)

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m slam.slam_visualizer view          # Interactive viewer")
        print("  python -m slam.slam_visualizer query         # Query nearby resources")
        print("  python -m slam.slam_visualizer route         # Plan multi-zone route")
        print("  python -m slam.slam_visualizer best          # Find best farming zone")
        sys.exit(0)

    command = sys.argv[1]

    if command == "view":
        from ultralytics import YOLO

        from .slam_navigator import SLAMNavigator

        slam = SLAMNavigator(
            fiber_detector=YOLO("/home/lucas/Documents/ai-research/nano-agent/model.pt")
        )
        try:
            slam.load_map("slam_maps/latest.pkl")
        except:
            print(
                "No map found. Run python -m slam.slam_navigator first to build a map."
            )
            sys.exit(1)

        viz = SLAMVisualizer(slam)
        viz.show()

    elif command == "query":
        query_nearby_resources()

    elif command == "route":
        plan_multi_zone_route()

    elif command == "best":
        find_best_resource_zone()
