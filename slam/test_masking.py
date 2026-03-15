"""
Test UI and Character Masking for SLAM
Show preview of what regions are tracked vs masked
"""

import cv2
import numpy as np
from pathlib import Path


def create_world_mask(screenshot):
    """
    Create mask that ONLY shows the world (no UI, no character)

    Returns:
        mask: 255 = track these pixels, 0 = ignore these pixels
    """
    height, width = screenshot.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Resolution: 1280x720 (from your video)

    # 1. Mask out UI regions (simplified approach)

    # Top bar: Health/mana, buffs, etc (entire top strip)
    mask[0:100, :] = 0  # Top 100 pixels across entire width

    # Top-left corner: Minimap + stats (additional)
    mask[0:200, 0:250] = 0

    # Bottom-left: Chat panel
    mask[400:720, 0:400] = 0  # Bottom-left corner for chat

    # Bottom strip: Skill bar, health/mana, inventory
    mask[600:720, :] = 0

    # Right side: Just mask the right half to catch all UI elements
    mask[:, 960:1280] = 0  # Right 320 pixels (~25% of width)

    # 2. Mask out CHARACTER in center (critical!)
    center_x, center_y = width // 2, height // 2

    # Character + immediate surroundings
    character_radius = 60  # Character sprite size
    y, x = np.ogrid[:height, :width]
    char_mask = (x - center_x)**2 + (y - center_y)**2 <= character_radius**2
    mask[char_mask] = 0

    # 3. Dead zone around character (other players, NPCs, visual effects)
    dead_zone_radius = 120
    dead_zone_mask = (x - center_x)**2 + (y - center_y)**2 <= dead_zone_radius**2
    # Make this semi-transparent in preview
    mask[dead_zone_mask & ~char_mask] = 128  # Gray zone

    return mask


def visualize_mask(video_path, frame_number=1000):
    """
    Show which regions will be tracked (green) vs ignored (red)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    # Seek to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Cannot read frame {frame_number}")
        cap.release()
        return

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create mask
    mask = create_world_mask(frame_rgb)

    # Create visualization overlay
    overlay = frame_rgb.copy()

    # Green = tracked region (mask = 255)
    overlay[mask == 255] = (0, 255, 0)  # Green tint

    # Red = ignored region (mask = 0)
    overlay[mask == 0] = (255, 0, 0)  # Red tint

    # Yellow = dead zone (mask = 128)
    overlay[mask == 128] = (255, 255, 0)  # Yellow tint

    # Blend original frame with overlay
    alpha = 0.4
    preview = cv2.addWeighted(frame_rgb, 1-alpha, overlay, alpha, 0)

    # Add text labels
    height, width = frame_rgb.shape[:2]
    cv2.putText(preview, "GREEN = Tracked (World Features)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, "RED = Ignored (UI)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(preview, "YELLOW = Dead Zone (Character Area)",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(preview, f"Frame: {frame_number}",
                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save preview
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.imwrite('mask_preview.png', preview_bgr)
    print("✅ Saved mask preview to: mask_preview.png")

    # Also save just the mask for debugging
    mask_colored = np.zeros_like(frame_rgb)
    mask_colored[mask == 255] = (0, 255, 0)  # Green = track
    mask_colored[mask == 0] = (255, 0, 0)    # Red = ignore
    mask_colored[mask == 128] = (255, 255, 0)  # Yellow = dead zone

    mask_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite('mask_only.png', mask_bgr)
    print("✅ Saved mask-only view to: mask_only.png")

    # Show in window (optional)
    cv2.imshow('Mask Preview - Press any key to close', preview_bgr)
    cv2.imshow('Mask Only', mask_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

    # Return stats
    total_pixels = mask.size
    tracked_pixels = np.sum(mask == 255)
    ignored_pixels = np.sum(mask == 0)
    dead_zone_pixels = np.sum(mask == 128)

    print(f"\n📊 Mask Statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Tracked (green): {tracked_pixels:,} ({tracked_pixels/total_pixels*100:.1f}%)")
    print(f"  Ignored (red): {ignored_pixels:,} ({ignored_pixels/total_pixels*100:.1f}%)")
    print(f"  Dead zone (yellow): {dead_zone_pixels:,} ({dead_zone_pixels/total_pixels*100:.1f}%)")


if __name__ == "__main__":
    video_path = "Albion Online Client 2024 08 10 - Fiber Gathering - Pest Lizard - T6 Avalonian Sickle - 100K 15min.mp4"

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Please provide the correct path.")
    else:
        print("🎨 Creating mask preview...")
        print("This will show GREEN=tracked, RED=ignored, YELLOW=dead zone\n")

        # Test at different frames to see different game states
        visualize_mask(video_path, frame_number=1000)  # Normal gameplay

        print("\n💡 Check mask_preview.png and mask_only.png")
        print("   If regions look wrong, we can adjust the coordinates!")
