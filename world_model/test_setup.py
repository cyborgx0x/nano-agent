"""
Quick test to verify world_model setup.
Run this to check if all dependencies are working.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed. Run: pip install torch torchvision")
        return False

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed. Run: pip install opencv-python")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed. Run: pip install numpy")
        return False

    try:
        from mss import mss
        print("✓ mss (screen capture)")
    except ImportError:
        print("✗ mss not installed. Run: pip install mss")
        return False

    try:
        import pyautogui
        print("✓ pyautogui")
    except ImportError:
        print("✗ pyautogui not installed. Run: pip install pyautogui")
        return False

    try:
        from pynput import mouse, keyboard
        print("✓ pynput")
    except ImportError:
        print("✗ pynput not installed. Run: pip install pynput")
        return False

    return True


def test_file_structure():
    """Test if all required files are present."""
    print("\nTesting file structure...")

    required_files = [
        "models/vision_transformer.py",
        "models/predictor.py",
        "models/utils/modules.py",
        "datasets/gameplay_dataset.py",
        "integration/game_controller.py",
        "game_world_model.py",
        "utils/checkpoint_loader.py",
        "masks/utils.py",
    ]

    world_model_dir = Path(__file__).parent
    all_present = True

    for file_path in required_files:
        full_path = world_model_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_present = False

    return all_present


def test_model_import():
    """Test if models can be imported."""
    print("\nTesting model imports...")

    try:
        # Add world_model to path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from world_model.models.vision_transformer import VisionTransformer
        print("✓ Can import VisionTransformer")

        from world_model.models.predictor import VisionTransformerPredictor
        print("✓ Can import VisionTransformerPredictor")

        from world_model.game_world_model import GameWorldModel
        print("✓ Can import GameWorldModel")

        from world_model.datasets.gameplay_dataset import GameplayCollector, GameplayDataset
        print("✓ Can import GameplayCollector and GameplayDataset")

        from world_model.integration.game_controller import WorldModelGameController
        print("✓ Can import WorldModelGameController")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_screen_capture():
    """Test screen capture."""
    print("\nTesting screen capture...")

    try:
        from mss import mss
        import numpy as np

        with mss() as sct:
            # Capture full screen
            screenshot = sct.grab(sct.monitors[1])
            frame = np.array(screenshot)[:, :, :3]

            print(f"✓ Screen capture working")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            print(f"  Channels: {frame.shape[2]}")

            return True

    except Exception as e:
        print(f"✗ Screen capture failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("World Model Setup Test")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test file structure
    results.append(("File Structure", test_file_structure()))

    # Test model imports
    results.append(("Model Imports", test_model_import()))

    # Test screen capture
    results.append(("Screen Capture", test_screen_capture()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Collect training data:")
        print("   python world_model/datasets/gameplay_dataset.py")
        print("\n2. Train world model (script coming soon)")
        print("\n3. Use agent:")
        print("   python world_model/integration/game_controller.py")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies.")
        print("\nQuick install:")
        print("pip install torch torchvision opencv-python numpy mss pyautogui pynput")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
