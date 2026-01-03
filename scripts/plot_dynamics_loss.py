#!/usr/bin/env python3
"""Plot loss curve from dynamics training log file."""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def extract_losses(log_file: str) -> tuple[list[int], list[float]]:
    """Extract step and loss values from training log."""
    steps = []
    losses = []
    
    # Pattern to match "Step X/10000: loss=Y" lines
    step_pattern = re.compile(r'Step (\d+)/\d+: loss=([0-9.]+)')
    
    # Pattern to match progress bar lines with loss values (capturing last loss on line)
    progress_pattern = re.compile(r'\| (\d+)/\d+ \[.*loss=([0-9.]+)')
    
    # Alternative pattern for tqdm output
    tqdm_pattern = re.compile(r'(\d+)/10000.*loss=([0-9.e+-]+)\]?\s*$')
    
    seen_steps = set()
    
    with open(log_file, 'r') as f:
        for line in f:
            # Skip NaN losses
            if 'loss=nan' in line.lower():
                continue
            
            # Try step pattern first (more reliable)
            match = step_pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                if step not in seen_steps and loss > 0:
                    steps.append(step)
                    losses.append(loss)
                    seen_steps.add(step)
                continue
            
            # Try tqdm pattern
            match = tqdm_pattern.search(line)
            if match:
                step = int(match.group(1))
                try:
                    loss = float(match.group(2))
                    if step not in seen_steps and loss > 0 and not np.isnan(loss):
                        steps.append(step)
                        losses.append(loss)
                        seen_steps.add(step)
                except ValueError:
                    continue
    
    # Sort by step
    sorted_data = sorted(zip(steps, losses), key=lambda x: x[0])
    if sorted_data:
        steps, losses = zip(*sorted_data)
        return list(steps), list(losses)
    return [], []


def smooth_curve(values: list[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_loss_curve(log_file: str, output_file: str = None, show: bool = True):
    """Plot the loss curve from a training log."""
    steps, losses = extract_losses(log_file)
    
    if not steps:
        print("No loss data found in log file!")
        return
    
    print(f"Found {len(steps)} data points")
    print(f"Steps range: {min(steps)} to {max(steps)}")
    print(f"Loss range: {min(losses):.6f} to {max(losses):.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Full loss curve
    ax1 = axes[0]
    ax1.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw loss')
    
    # Add smoothed curve
    if len(losses) > 100:
        window = min(100, len(losses) // 10)
        smoothed = smooth_curve(losses, window)
        smooth_steps = steps[window-1:]
        ax1.plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Dynamics Model Training Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(steps) * 1.02)
    
    # Right plot: Log scale
    ax2 = axes[1]
    ax2.semilogy(steps, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw loss')
    
    if len(losses) > 100:
        ax2.semilogy(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Dynamics Model Training Loss (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(steps) * 1.02)
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        log_path = Path(log_file)
        output_file = log_path.parent / f"{log_path.stem}_loss_curve.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss curve from training log')
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output image file')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    
    args = parser.parse_args()
    plot_loss_curve(args.log_file, args.output, show=not args.no_show)
