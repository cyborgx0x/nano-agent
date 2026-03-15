"""
Test SLAM Navigator on YouTube Video
Download Albion Online gameplay and test visual odometry + mapping
"""

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .slam_navigator import SLAMNavigator

# from ultralytics import YOLO  # Temporarily disabled during PyTorch reinstall


class VideoSLAMTester:
    """
    Test SLAM system on pre-recorded video.
    """

    def __init__(self, video_path, fiber_detector=None):
        self.video_path = video_path
        self.slam = SLAMNavigator(
            fiber_detector=fiber_detector,
            map_size=(5000, 5000),  # Large map for full exploration
        )

        # Results tracking
        self.trajectory = []  # [(frame_id, x, y, heading)]
        self.detected_objects_count = []
        self.processing_times = []

    def run_test(
        self,
        max_frames=None,
        display=True,
        save_interval=50,
        frame_skip=3,
        start_frame=0,
    ):
        """
        Run SLAM on entire video.

        Args:
            max_frames: Limit number of frames (None = all)
            display: Show visualization
            save_interval: Save map every N frames
            frame_skip: Process every Nth frame (default: 3 = process 1 out of every 3 frames)
            start_frame: Frame number to start from
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"❌ Error: Could not open video {self.video_path}")
            return

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("=" * 70)
        print("🎥 Video SLAM Test")
        print("=" * 70)
        print(f"Video: {self.video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.1f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f} seconds")
        if start_frame > 0:
            print(f"Starting from frame: {start_frame}")
        print("=" * 70)

        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"Processing frames {start_frame} to {max_frames}")

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_id = start_frame

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret or (max_frames and frame_id >= max_frames):
                    break

                # Skip frames for performance
                if frame_id % frame_skip != 0:
                    frame_id += 1
                    continue

                # Convert BGR to RGB for SLAM
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                start_time = time.time()
                status = self.slam.update_map(frame_rgb)
                processing_time = time.time() - start_time

                # Record results
                self.trajectory.append(
                    (
                        frame_id,
                        self.slam.current_position[0],
                        self.slam.current_position[1],
                        self.slam.heading,
                    )
                )

                if status and "detected_objects" in status:
                    self.detected_objects_count.append(len(status["detected_objects"]))
                else:
                    self.detected_objects_count.append(0)

                self.processing_times.append(processing_time)

                # Display progress
                if frame_id % 10 == 0:
                    avg_time = (
                        np.mean(self.processing_times[-10:])
                        if self.processing_times
                        else 0
                    )
                    print(
                        f"Frame {frame_id}/{total_frames} | "
                        f"Pos: ({self.slam.current_position[0]}, {self.slam.current_position[1]}) | "
                        f"Objects: {self.detected_objects_count[-1]} | "
                        f"Time: {processing_time*1000:.1f}ms | "
                        f"Avg: {avg_time*1000:.1f}ms"
                    )

                # Visualize
                if display and frame_id % 5 == 0:
                    self._display_frame(frame, frame_id)

                # Save map periodically (DISABLED - only save final to avoid disk space issues)
                # if frame_id % save_interval == 0 and frame_id > 0:
                #     self.slam.save_map(f'slam_maps/video_test_frame_{frame_id}.pkl')

                frame_id += 1

                # Press 'q' to quit early
                if display and cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n⚠️  Stopped by user")
                    break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        # Final save
        self.slam.save_map("slam_maps/video_test_final.pkl")

        print("\n" + "=" * 70)
        print("✅ SLAM Test Complete!")
        print("=" * 70)
        print(f"Processed {frame_id} frames")
        print(f"Average processing time: {np.mean(self.processing_times)*1000:.1f}ms")
        print(f"Total distance traveled: {self._calculate_distance():.0f} pixels")
        print(f"Total objects detected: {sum(self.detected_objects_count)}")
        print(f"Zones discovered: {len(self.slam.zones)}")

        return self.generate_report()

    def _display_frame(self, frame, frame_id):
        """Display current frame with SLAM overlay."""
        display = frame.copy()

        # Draw position info
        info_text = [
            f"Frame: {frame_id}",
            f"Pos: ({self.slam.current_position[0]}, {self.slam.current_position[1]})",
            f"Zone: {self.slam.current_zone}",
            f"Objects: {len(self.slam.object_database)}",
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(
                display,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 25

        # Draw mini trajectory (last 50 positions)
        if len(self.trajectory) > 1:
            recent = self.trajectory[-50:]

            # Normalize to fit in corner
            xs = [p[1] for p in recent]
            ys = [p[2] for p in recent]

            if max(xs) - min(xs) > 0 and max(ys) - min(ys) > 0:
                scale_x = 150 / (max(xs) - min(xs))
                scale_y = 150 / (max(ys) - min(ys))
                offset_x = display.shape[1] - 170
                offset_y = 20

                for i in range(len(recent) - 1):
                    x1 = int((recent[i][1] - min(xs)) * scale_x) + offset_x
                    y1 = int((recent[i][2] - min(ys)) * scale_y) + offset_y
                    x2 = int((recent[i + 1][1] - min(xs)) * scale_x) + offset_x
                    y2 = int((recent[i + 1][2] - min(ys)) * scale_y) + offset_y

                    cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Current position
                x_curr = int((recent[-1][1] - min(xs)) * scale_x) + offset_x
                y_curr = int((recent[-1][2] - min(ys)) * scale_y) + offset_y
                cv2.circle(display, (x_curr, y_curr), 5, (0, 0, 255), -1)

        cv2.imshow("SLAM Video Test", display)

    def _calculate_distance(self):
        """Calculate total distance traveled."""
        if len(self.trajectory) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][1] - self.trajectory[i - 1][1]
            dy = self.trajectory[i][2] - self.trajectory[i - 1][2]
            total_distance += np.sqrt(dx**2 + dy**2)

        return total_distance

    def generate_report(self):
        """Generate detailed test report."""
        report = {
            "total_frames": len(self.trajectory),
            "total_distance": self._calculate_distance(),
            "avg_processing_time_ms": np.mean(self.processing_times) * 1000,
            "total_objects_detected": sum(self.detected_objects_count),
            "zones_discovered": len(self.slam.zones),
            "trajectory": self.trajectory,
        }

        return report

    def visualize_results(self, save_path="slam_test_results.png"):
        """Create visualization of test results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Trajectory plot
        ax = axes[0, 0]
        xs = [p[1] for p in self.trajectory]
        ys = [p[2] for p in self.trajectory]

        ax.plot(xs, ys, "b-", linewidth=1, alpha=0.6)
        ax.scatter(xs[0], ys[0], c="green", s=100, marker="o", label="Start", zorder=5)
        ax.scatter(xs[-1], ys[-1], c="red", s=100, marker="X", label="End", zorder=5)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        ax.set_title("SLAM Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # 2. Distance over time
        ax = axes[0, 1]
        distances = [0]
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][1] - self.trajectory[i - 1][1]
            dy = self.trajectory[i][2] - self.trajectory[i - 1][2]
            distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))

        ax.plot(distances, "g-", linewidth=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cumulative Distance (pixels)")
        ax.set_title("Distance Traveled Over Time")
        ax.grid(True, alpha=0.3)

        # 3. Processing time
        ax = axes[1, 0]
        ax.plot([t * 1000 for t in self.processing_times], "r-", alpha=0.6)
        ax.axhline(
            np.mean(self.processing_times) * 1000,
            color="blue",
            linestyle="--",
            label=f"Mean: {np.mean(self.processing_times)*1000:.1f}ms",
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Processing Time (ms)")
        ax.set_title("SLAM Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Objects detected
        ax = axes[1, 1]
        ax.plot(self.detected_objects_count, "purple", linewidth=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Objects Detected")
        ax.set_title("Object Detection Count")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n📊 Results saved to {save_path}")
        plt.show()


def download_youtube_video(url, output_path="test_video.mp4"):
    """
    Download YouTube video using yt-dlp.

    Install: pip install yt-dlp
    """
    try:
        import yt_dlp

        ydl_opts = {
            "format": "best[height<=1080]",  # Max 1080p
            "outtmpl": output_path,
        }

        print(f"📥 Downloading video from YouTube...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print(f"✅ Video downloaded to {output_path}")
        return output_path

    except ImportError:
        print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def main():
    """Main test function."""
    import sys

    print("=" * 70)
    print("🎥 SLAM Video Tester for Albion Online")
    print("=" * 70)

    # Check if video path provided
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m slam.test_slam_video <video_path>")
        print("  python -m slam.test_slam_video <youtube_url>")
        print("\nExamples:")
        print("  python -m slam.test_slam_video albion_gameplay.mp4")
        print("  python -m slam.test_slam_video https://youtube.com/watch?v=...")
        print("\n💡 Tip: Search YouTube for 'Albion Online gathering gameplay'")
        sys.exit(0)

    video_input = sys.argv[1]

    # Check if it's a YouTube URL
    if "youtube.com" in video_input or "youtu.be" in video_input:
        print("🔗 YouTube URL detected")
        video_path = download_youtube_video(video_input, "albion_test_video.mp4")
        if video_path is None:
            sys.exit(1)
    else:
        video_path = video_input
        if not Path(video_path).exists():
            print(f"❌ Error: Video file not found: {video_path}")
            sys.exit(1)

    # Load fiber detector
    try:
        from ultralytics import YOLO

        fiber_detector = YOLO("model.pt")
        print("✅ Fiber detection enabled (model.pt loaded)")
    except Exception as e:
        print(f"⚠️  Fiber detection disabled: {e}")
        fiber_detector = None

    # Create tester
    tester = VideoSLAMTester(video_path, fiber_detector)

    # Run test
    print("\n▶️  Starting SLAM test...")
    print("Press 'q' to stop early\n")

    # For testing: start at frame 3000, process 500 frames
    report = tester.run_test(
        max_frames=3500,  # Start at 3000, process 500 frames
        display=False,  # Disable display to save CPU
        save_interval=300,
        frame_skip=5,  # Process every 5th frame
    )

    # Visualize results
    print("\n📊 Generating visualization...")
    tester.visualize_results("slam_video_results.png")

    # Render map with trajectory and detected objects
    print("\n🗺️  Rendering resource map...")
    resource_map = tester.slam.render_map_with_objects(trajectory=tester.trajectory)

    if resource_map is not None:
        cv2.imwrite("resource_map.png", resource_map)
        print(
            f"✅ Resource map saved to resource_map.png ({resource_map.shape[1]}x{resource_map.shape[0]})"
        )
        print(f"   - Trajectory: {len(tester.trajectory)} waypoints")
        print(
            f"   - Detected objects: {sum(len(objs) for objs in tester.slam.object_database.values())}"
        )
    else:
        print("⚠️  No trajectory or objects to render")

    print("\n" + "=" * 70)
    print("✅ SLAM Test Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - slam_maps/video_test_final.pkl (SLAM map)")
    print("  - slam_video_results.png (performance graphs)")
    print("  - discovered_map.png (visual map)")


if __name__ == "__main__":
    main()
