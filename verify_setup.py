#!/usr/bin/env python3
"""
Setup Verification Script for RL Training
Checks all prerequisites and provides diagnostic information
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]

    if major >= 3 and minor >= 10:
        print(f"  ✓ Python {major}.{minor} (OK)")
        return True
    else:
        print(f"  ✗ Python {major}.{minor} (Need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable-Baselines3',
        'numpy': 'NumPy',
        'pyautogui': 'PyAutoGUI',
    }

    optional = {
        'ultralytics': 'YOLOv8 (required for real environment)',
        'easyocr': 'EasyOCR (required for real environment)',
    }

    all_ok = True

    # Check required
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING (required)")
            all_ok = False

    # Check optional
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - missing (optional)")

    return all_ok


def check_cuda():
    """Check if CUDA is available"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA available: {device_name}")
            print(f"    Memory: {memory:.2f} GB")
            return True
        else:
            print("  ⚠️  CUDA not available (CPU training will be slower)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_model_file():
    """Check if YOLO model exists"""
    print("\nChecking YOLO model...")
    model_path = "model.pt"

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ model.pt found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠️  model.pt not found")
        print("     - Required for REAL environment")
        print("     - Not needed for SIMULATED environment")
        print("     - Train with: python train.py --data config/dataset.yaml")
        return False


def test_fiber_detection():
    """Test YOLO detection"""
    print("\nTesting fiber detection...")
    try:
        from fiber_detection import FiberDetection
        import numpy as np

        detector = FiberDetection()

        # Create fake image
        fake_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Try detection
        detections = detector.detect(fake_image)

        if detector.model is not None:
            print(f"  ✓ Real YOLO detection working")
            print(f"    Detected {len(detections)} objects")
        else:
            print(f"  ✓ Simulated detection working")
            print(f"    Generated {len(detections)} fake detections")

        return True
    except Exception as e:
        print(f"  ✗ Detection failed: {e}")
        return False


def test_ocr():
    """Test OCR"""
    print("\nTesting OCR...")
    try:
        from gather_state import GatherState
        print("  ✓ OCR module loaded")
        print("    Note: OCR requires game running to test fully")
        return True
    except Exception as e:
        print(f"  ⚠️  OCR loading failed: {e}")
        print("     Not critical for simulated environment")
        return False


def test_environment():
    """Test RL environment"""
    print("\nTesting RL environment...")
    try:
        from game_env import SimplifiedAlbionEnv

        env = SimplifiedAlbionEnv()
        print("  ✓ Environment created")

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset (observation shape: {obs.shape})")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Environment step works")

        return True
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_agent():
    """Test RL agent creation"""
    print("\nTesting RL agent...")
    try:
        from rl_agent import create_agent

        agent = create_agent(algorithm='dqn', env_type='simplified', verbose=0)
        print("  ✓ DQN agent created")

        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def check_screen_setup():
    """Check screen setup"""
    print("\nChecking screen setup...")
    try:
        import pyautogui

        width, height = pyautogui.size()
        print(f"  ✓ Screen resolution: {width}x{height}")

        if width >= 1920 and height >= 1080:
            print("    Good resolution for training")
        else:
            print("    ⚠️  Low resolution (recommended: 1920x1080+)")

        return True
    except Exception as e:
        print(f"  ✗ Screen check failed: {e}")
        return False


def print_summary(results):
    """Print summary and recommendations"""
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)

    critical_checks = [
        'python', 'dependencies', 'environment', 'agent'
    ]

    critical_passed = sum(1 for k in critical_checks if results.get(k, False))
    total_critical = len(critical_checks)

    if critical_passed == total_critical:
        print("✓ All critical checks passed!")
        print("\nYou can now:")
        print("  1. Train with simulated environment:")
        print("     python train_rl.py --algorithm dqn --timesteps 10000")
        print()

        if results.get('model', False):
            print("  2. Train with real environment (if game running):")
            print("     python train_rl.py --algorithm dqn --timesteps 50000")
        else:
            print("  2. For real environment:")
            print("     - Train YOLO model first: see TRAINING.md")
            print("     - Or use simulated environment for testing")
    else:
        print(f"✗ {total_critical - critical_passed}/{total_critical} critical checks failed")
        print("\nPlease fix the issues above before training")
        print("\nFor installation help:")
        print("  pip install -r requirements.txt")
        print("  pip install -r requirements-training.txt")

    print()
    print("Environment Options:")
    print("  - Simulated: No game required, fast training")
    print("  - Real: Requires game running, realistic training")
    print()
    print("For detailed setup instructions, see SETUP.md")
    print("=" * 70)


def main():
    """Run all checks"""
    print("=" * 70)
    print("RL TRAINING SETUP VERIFICATION")
    print("=" * 70)
    print()

    results = {}

    # Run all checks
    results['python'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['cuda'] = check_cuda()
    results['model'] = check_model_file()
    results['detection'] = test_fiber_detection()
    results['ocr'] = test_ocr()
    results['screen'] = check_screen_setup()
    results['environment'] = test_environment()
    results['agent'] = test_rl_agent()

    # Print summary
    print_summary(results)

    # Exit code
    critical_passed = all([
        results['python'],
        results['dependencies'],
        results['environment'],
        results['agent']
    ])

    sys.exit(0 if critical_passed else 1)


if __name__ == '__main__':
    main()
