"""
YOLOv8 Finetuning Script for Albion Online Resource Detection
Supports training from scratch or finetuning existing model.pt
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train or finetune YOLOv8 model')

    # Model configuration
    parser.add_argument('--model', type=str, default='model.pt',
                       help='Path to model file (model.pt for finetuning, yolov8n.pt for scratch)')
    parser.add_argument('--data', type=str, default='config/dataset.yaml',
                       help='Path to dataset configuration YAML')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (reduce if OOM errors)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (0 or cpu)')

    # Advanced training options
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Warmup epochs')

    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--hsv-h', type=float, default=0.015,
                       help='Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                       help='Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                       help='Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Rotation augmentation (degrees)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Scaling augmentation')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Shear augmentation')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Vertical flip probability')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Horizontal flip probability')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability')

    # Output and saving
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')

    # Validation
    parser.add_argument('--val', action='store_true', default=True,
                       help='Validate during training')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')

    # Multi-GPU
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')

    # Experiment tracking
    parser.add_argument('--mlflow', action='store_true',
                       help='Enable MLflow tracking')

    return parser.parse_args()

def check_dataset(data_yaml):
    """Verify dataset configuration and paths"""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in dataset config: {key}")

    # Check if paths exist
    for split in ['train', 'val']:
        path = data.get(split)
        if path and not os.path.exists(path):
            print(f"⚠️  Warning: {split} path not found: {path}")

    print(f"✓ Dataset config loaded: {data['nc']} classes")
    print(f"  Classes: {data['names']}")

    return data

def setup_mlflow(args):
    """Setup MLflow experiment tracking"""
    try:
        import mlflow
        mlflow.set_experiment("yolo-finetuning")
        mlflow.start_run(run_name=args.name)

        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "img_size": args.imgsz,
            "lr0": args.lr0,
            "device": args.device,
        })

        print("✓ MLflow tracking enabled")
        return True
    except ImportError:
        print("⚠️  MLflow not installed, skipping experiment tracking")
        return False

def main():
    """Main training function"""
    args = parse_args()

    print("=" * 60)
    print("YOLOv8 Finetuning - Albion Online Resource Detection")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Verify dataset
    dataset_config = check_dataset(args.data)

    # Setup MLflow if requested
    if args.mlflow:
        setup_mlflow(args)

    # Load model
    print(f"\n📦 Loading model: {args.model}")
    if not os.path.exists(args.model):
        print(f"⚠️  Model not found, downloading {args.model}...")

    model = YOLO(args.model)

    # Print model info
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Training configuration
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.workers}")

    # Train the model
    results = model.train(
        # Data
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        # Device
        device=args.device,
        workers=args.workers,

        # Learning rate
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,

        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,

        # Output
        project=args.project,
        name=args.name,
        exist_ok=True,
        save_period=args.save_period,

        # Validation
        val=args.val,
        patience=args.patience,

        # Resume
        resume=args.resume,

        # Other
        verbose=True,
        plots=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Save final model
    output_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f"✓ Best model saved: {output_path}")

    # Validation metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📊 Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    # Log to MLflow
    if args.mlflow:
        try:
            import mlflow
            mlflow.log_artifact(str(output_path))
            mlflow.end_run()
            print("✓ Results logged to MLflow")
        except:
            pass

    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
