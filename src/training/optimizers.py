"""Optimizer and scheduler setup"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Dict, Any


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
    lr_schedule = config.get('lr_schedule', 'cosine')
    max_lr = config['max_lr']
    min_lr = config.get('min_lr', max_lr)
    warmup_steps = config.get('warmup_steps', 0)
    
    if lr_schedule == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + torch.cos(torch.tensor(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif lr_schedule == 'constant':
        # Constant learning rate (with optional warmup)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


# Fix import for math
import math
