#!/usr/bin/env python3
"""Plot loss curves from training log files"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def parse_log_file(log_path):
    """Parse training log file and extract step and loss values"""
    steps = []
    losses = []
    
    # Pattern to match tqdm progress bar with loss value
    # Example: Training:   1%|          | 100/15000 [00:50<2:02:01,  2.04it/s, loss=0.128]
    pattern = r'Training:.*?\|.*?(\d+)/\d+.*?loss=([\d\.]+|nan)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss_str = match.group(2).lower()
                
                # Skip NaN values
                if loss_str == 'nan':
                    continue
                
                try:
                    loss = float(loss_str)
                    steps.append(step)
                    losses.append(loss)
                except ValueError:
                    continue
    
    return np.array(steps), np.array(losses)


def plot_loss_curves(log_files, output_path=None, title=None):
    """Plot loss curves from one or more log files"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        log_path = Path(log_file)
        if not log_path.exists():
            print(f"Warning: {log_file} not found, skipping...")
            continue
        
        steps, losses = parse_log_file(log_path)
        
        if len(steps) == 0:
            print(f"Warning: No loss data found in {log_file}")
            continue
        
        # Sort by step to ensure correct plotting order
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        losses = losses[sort_idx]
        
        # Get a nice label from filename
        label = log_path.stem.replace('train_', '').replace('_', ' ').title()
        
        ax.plot(steps, losses, label=label, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title or 'Training Loss Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Use log scale for y-axis if losses span multiple orders of magnitude
    if len(log_files) > 0:
        all_losses = []
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                _, losses = parse_log_file(log_path)
                if len(losses) > 0:
                    all_losses.extend(losses)
        
        if len(all_losses) > 0:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            if max_loss / min_loss > 100:  # If span is > 100x, use log scale
                ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves from training logs')
    parser.add_argument('log_files', nargs='+', help='Path(s) to training log file(s)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output path for the plot (default: show interactively)')
    parser.add_argument('--title', '-t', type=str, default=None,
                       help='Title for the plot')
    
    args = parser.parse_args()
    
    # If no log files specified, try to find common log files
    if len(args.log_files) == 0:
        base_dir = Path(__file__).parent.parent
        common_logs = [
            base_dir / 'train_dynamics3.log',
            base_dir / 'train_dynamics.log',
            base_dir / 'train_tokenizer.log',
            base_dir / 'train_lam.log',
        ]
        args.log_files = [str(f) for f in common_logs if f.exists()]
        
        if len(args.log_files) == 0:
            print("No log files found. Please specify log file paths.")
            sys.exit(1)
    
    plot_loss_curves(args.log_files, args.output, args.title)


if __name__ == '__main__':
    main()
