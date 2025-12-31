"""Optimizer and scheduler setup"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Dict, Any
import json
import os
from pathlib import Path


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> optim.Optimizer:
    """
    Create AdamW optimizer.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration with keys:
            - max_lr: Maximum learning rate
            - beta1: Beta1 parameter
            - beta2: Beta2 parameter
            - weight_decay: Weight decay
    
    Returns:
        AdamW optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=config['max_lr'],
        betas=(config.get('beta1', 0.9), config.get('beta2', 0.9)),
        weight_decay=config.get('weight_decay', 1e-4),
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Scheduler configuration with keys:
            - lr_schedule: "cosine" or "constant"
            - max_lr: Maximum learning rate
            - min_lr: Minimum learning rate
            - warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
    
    Returns:
        Learning rate scheduler
    """
    # #region agent log
    log_path = Path("/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log")
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "optimizers.py:35", "message": "create_scheduler called", "data": {"num_training_steps": num_training_steps, "warmup_steps": config.get('warmup_steps', 0), "lr_schedule": config.get('lr_schedule', 'cosine')}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
    except: pass
    # #endregion
    lr_schedule = config.get('lr_schedule', 'cosine')
    max_lr = config['max_lr']
    min_lr = config.get('min_lr', max_lr)
    warmup_steps = config.get('warmup_steps', 0)
    
    # Cap warmup_steps to num_training_steps to avoid division by zero
    # If warmup_steps >= num_training_steps, we'll only do warmup (no decay phase)
    warmup_steps = min(warmup_steps, num_training_steps)
    
    if lr_schedule == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(step: int) -> float:
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "optimizers.py:72", "message": "lr_lambda called", "data": {"step": step, "warmup_steps": warmup_steps, "num_training_steps": num_training_steps, "denominator": num_training_steps - warmup_steps}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
            except: pass
            # #endregion
            if step < warmup_steps:
                # Linear warmup
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "optimizers.py:79", "message": "warmup branch", "data": {"step": step, "warmup_steps": warmup_steps, "lr_ratio": step / warmup_steps if warmup_steps > 0 else 0}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
                except: pass
                # #endregion
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                # Cosine decay
                # Handle case where warmup_steps == num_training_steps (no decay phase)
                if warmup_steps >= num_training_steps:
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "optimizers.py:88", "message": "no decay phase - using max_lr", "data": {"step": step, "warmup_steps": warmup_steps, "num_training_steps": num_training_steps}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    return 1.0  # Use max_lr when no decay phase
                
                # #region agent log
                try:
                    denominator = num_training_steps - warmup_steps
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "optimizers.py:95", "message": "cosine decay branch - BEFORE division", "data": {"step": step, "warmup_steps": warmup_steps, "num_training_steps": num_training_steps, "denominator": denominator, "numerator": step - warmup_steps}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
                except: pass
                # #endregion
                progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + torch.cos(torch.tensor(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif lr_schedule == 'constant':
        # Constant learning rate (with optional warmup)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


# Fix import for math
import math
