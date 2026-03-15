"""
Test Path Planning with SLAM Map
Load saved SLAM data and plan optimal fiber gathering routes
"""

import cv2
import pickle
import numpy as np
from path_planner import PathPlanner, GatheringRouter, visualize_path


def load_slam_map(slam_path='slam_maps/video_test_final.pkl'):
    """Load saved SLAM map data."""
    with open(slam_path, 'rb') as f:
        data = pickle.load(f)
    return data


def test_pathfinding():
    """Test path planning on saved SLAM map."""
    print("=" * 70)
    print("🗺️  Path Planning Test")
    print("=" * 70)

    # Load SLAM data
    print("\n📂 Loading SLAM map...")
    try:
        slam_data = load_slam_map()
    except FileNotFoundError:
        print("❌ SLAM map not found. Run test_slam_video.py first!")
        return

    print(f"✅ Loaded SLAM data")
    print(f"   Current position: {slam_data['current_position']}")
    print(f"   Zones: {len(slam_data['zones'])}")

    # Extract fiber positions
    fiber_objects = slam_data['object_database'].get('fiber', [])
    print(f"   Detected fibers: {len(fiber_objects)}")

    if len(fiber_objects) == 0:
        print("\n⚠️  No fibers detected. Run test with fiber detection enabled!")
        return

    # Convert to position list
    fiber_positions = [obj['position'] for obj in fiber_objects]
    current_pos = tuple(slam_data['current_position'])

    print(f"\n🎯 Planning gathering route...")
    print(f"   Starting from: {current_pos}")
    print(f"   Total fibers available: {len(fiber_positions)}")

    # Initialize planner
    planner = PathPlanner()
    router = GatheringRouter(planner)

    # Plan route to collect up to 10 nearest fibers
    fiber_route = router.plan_gathering_route(
        current_pos=current_pos,
        fiber_positions=fiber_positions,
        max_fibers=10
    )

    print(f"\n📍 Planned route:")
    print(f"   Fibers to collect: {len(fiber_route)}")
    print(f"   Total route distance: {router.calculate_route_length([current_pos] + fiber_route):.0f} pixels")

    # Plan detailed path to first fiber
    if fiber_route:
        first_fiber = fiber_route[0]
        print(f"\n🛣️  Finding path to first fiber at {first_fiber}...")

        path = planner.find_path(current_pos, first_fiber)

        if path:
            print(f"✅ Path found!")
            print(f"   Waypoints: {len(path)}")
            print(f"   Distance: {router.calculate_route_length(path):.0f} pixels")

            # Simplify path
            simplified = planner.simplify_path(path, tolerance=10.0)
            print(f"   Simplified to {len(simplified)} waypoints (reduced {len(path) - len(simplified)})")

            # Visualize on map
            print(f"\n🎨 Creating visualization...")

            # Create blank map
            all_positions = [current_pos] + fiber_positions
            xs = [p[0] for p in all_positions]
            ys = [p[1] for p in all_positions]

            min_x, max_x = int(min(xs)) - 100, int(max(xs)) + 100
            min_y, max_y = int(min(ys)) - 100, int(max(ys)) + 100

            width = max_x - min_x
            height = max_y - min_y

            # White background
            map_img = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Draw all fibers (orange)
            for fiber_pos in fiber_positions:
                x = int(fiber_pos[0] - min_x)
                y = int(fiber_pos[1] - min_y)
                cv2.circle(map_img, (x, y), 5, (0, 165, 255), -1)

            # Draw planned route (yellow dashed line)
            route_with_start = [current_pos] + fiber_route
            for i in range(len(route_with_start) - 1):
                x1 = int(route_with_start[i][0] - min_x)
                y1 = int(route_with_start[i][1] - min_y)
                x2 = int(route_with_start[i+1][0] - min_x)
                y2 = int(route_with_start[i+1][1] - min_y)
                cv2.line(map_img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

            # Draw path to first fiber (magenta)
            path_offset = [(int(p[0] - min_x), int(p[1] - min_y)) for p in path]
            for i in range(len(path_offset) - 1):
                cv2.line(map_img, path_offset[i], path_offset[i+1], (255, 0, 255), 3)

            # Draw simplified path waypoints (cyan)
            simplified_offset = [(int(p[0] - min_x), int(p[1] - min_y)) for p in simplified]
            for point in simplified_offset:
                cv2.circle(map_img, point, 7, (255, 255, 0), -1)

            # Draw current position (green)
            curr_x = int(current_pos[0] - min_x)
            curr_y = int(current_pos[1] - min_y)
            cv2.circle(map_img, (curr_x, curr_y), 10, (0, 255, 0), -1)

            # Draw first fiber target (red)
            target_x = int(first_fiber[0] - min_x)
            target_y = int(first_fiber[1] - min_y)
            cv2.circle(map_img, (target_x, target_y), 10, (0, 0, 255), -1)

            # Add legend
            legend_y = 30
            cv2.putText(map_img, "Path Planning Visualization", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            legend_y += 40

            cv2.circle(map_img, (20, legend_y), 5, (0, 255, 0), -1)
            cv2.putText(map_img, "Current Position", (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 25

            cv2.circle(map_img, (20, legend_y), 5, (0, 0, 255), -1)
            cv2.putText(map_img, f"Target Fiber ({len(fiber_route)} in route)", (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 25

            cv2.circle(map_img, (20, legend_y), 5, (0, 165, 255), -1)
            cv2.putText(map_img, f"Other Fibers ({len(fiber_positions)})", (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 25

            cv2.line(map_img, (15, legend_y), (25, legend_y), (255, 0, 255), 3)
            cv2.putText(map_img, f"A* Path ({len(path)} waypoints)", (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 25

            cv2.circle(map_img, (20, legend_y), 5, (255, 255, 0), -1)
            cv2.putText(map_img, f"Simplified ({len(simplified)} waypoints)", (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Save
            cv2.imwrite('path_planning_demo.png', map_img)
            print(f"✅ Visualization saved to: path_planning_demo.png")

            # Print waypoints for bot execution
            print(f"\n🤖 Simplified waypoints for bot execution:")
            for i, waypoint in enumerate(simplified):
                print(f"   {i+1}. Move to {waypoint}")

        else:
            print(f"❌ No path found to first fiber!")

    print("\n" + "=" * 70)
    print("✅ Path Planning Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_pathfinding()
