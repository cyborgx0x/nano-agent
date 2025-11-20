#!/usr/bin/env python3
"""
Plot training metrics from Tensorboard logs

Usage:
    python scripts/plot_training.py --log_dir logs/tensorboard
"""

import argparse
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_tensorboard_logs(log_dir: str, output_dir: str = None):
    """Plot training metrics from Tensorboard logs"""

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return

    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))

    if not event_files:
        print(f"No Tensorboard event files found in {log_dir}")
        return

    print(f"Found {len(event_files)} event file(s)")

    # Load events
    ea = event_accumulator.EventAccumulator(
        event_files[0],
        size_guidance={
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"\nAvailable metrics: {tags}")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Spider Standing Training Metrics', fontsize=16)

    # Plot 1: Losses
    ax = axes[0, 0]
    if 'Loss/value_loss' in tags:
        events = ea.Scalars('Loss/value_loss')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, label='Value Loss')

    if 'Loss/surrogate_loss' in tags:
        events = ea.Scalars('Loss/surrogate_loss')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, label='Surrogate Loss')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Standing Success Rate
    ax = axes[0, 1]
    if 'Metrics/standing_success_rate' in tags:
        events = ea.Scalars('Metrics/standing_success_rate')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Success Rate')
        ax.set_title('Standing Success Rate')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    # Plot 3: Torso Height
    ax = axes[1, 0]
    if 'Metrics/avg_torso_height' in tags:
        events = ea.Scalars('Metrics/avg_torso_height')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, color='blue')
        ax.axhline(y=0.18, color='r', linestyle='--', label='Target Height')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Height (m)')
        ax.set_title('Average Torso Height')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Tilt
    ax = axes[1, 1]
    if 'Metrics/avg_tilt_deg' in tags:
        events = ea.Scalars('Metrics/avg_tilt_deg')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, color='orange')
        ax.axhline(y=10, color='g', linestyle='--', label='Success Threshold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Tilt (degrees)')
        ax.set_title('Average Tilt')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(log_dir)

    output_file = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard",
                        help="Path to Tensorboard log directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plot (default: same as log_dir parent)")

    args = parser.parse_args()

    plot_tensorboard_logs(args.log_dir, args.output_dir)
