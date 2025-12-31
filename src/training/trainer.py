"""Main training loop"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

from ..utils.config import load_config
from .optimizers import create_optimizer, create_scheduler
from .losses import reconstruction_loss, vq_loss, maskgit_loss, lam_loss


class Trainer:
    """Trainer for Genie models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        
        # Mixed precision
        self.use_amp = self.config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient checkpointing
        if self.config.get('gradient_checkpointing', False):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config.get('training', {}))
        num_steps = len(train_loader) * self.config.get('training', {}).get('max_steps', 10000) // len(train_loader)
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.get('training', {}),
            num_steps,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = self.config.get('training', {}).get('save_every', 5000)
        self.eval_every = self.config.get('training', {}).get('eval_every', 1000)
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            if isinstance(batch, torch.Tensor):
                # Video tokenizer or simple case
                output = self.model(batch)
                if isinstance(output, tuple):
                    pred, tokens, vq_loss_dict = output
                    vq_losses = vq_loss(vq_loss_dict)
                    loss = reconstruction_loss(pred, batch) + vq_losses['vq_loss']
                else:
                    loss = reconstruction_loss(output, batch)
            else:
                # LAM or Dynamics model
                if len(batch) == 2:
                    # LAM: (past_frames, next_frame)
                    past_frames, next_frame = batch
                    pred, actions, vq_loss_dict = self.model(past_frames, next_frame)
                    loss_dict = lam_loss(pred, next_frame, vq_loss_dict)
                    loss = loss_dict['total_loss']
                elif len(batch) == 3:
                    # Dynamics: (tokens, actions, targets)
                    tokens, actions, targets = batch
                    mask = self.model.generate_mask(
                        tokens.shape,
                        self.config.get('training', {}).get('mask_prob', 0.5),
                        self.device,
                    )
                    logits = self.model(tokens, actions, mask)
                    loss_dict = maskgit_loss(logits, targets, mask)
                    loss = loss_dict['maskgit_loss']
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            if self.config.get('training', {}).get('max_grad_norm'):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training', {}).get('max_grad_norm', 1.0),
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            if self.config.get('training', {}).get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training', {}).get('max_grad_norm', 1.0),
                )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return {'loss': loss.item()}
    
    def train(self, max_steps: Optional[int] = None):
        """Main training loop"""
        max_steps = max_steps or self.config.get('training', {}).get('max_steps', 10000)
        
        pbar = tqdm(total=max_steps, desc="Training")
        
        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                metrics = self.train_step(batch)
                pbar.update(1)
                pbar.set_postfix(metrics)
                
                # Evaluation
                if self.val_loader and self.global_step % self.eval_every == 0:
                    val_metrics = self.evaluate()
                    self.logger.info(f"Step {self.global_step}: {val_metrics}")
                
                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint()
            
            self.epoch += 1
        
        pbar.close()
        self.save_checkpoint()  # Final checkpoint
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        pred, _, _ = output
                        loss = reconstruction_loss(pred, batch)
                    else:
                        loss = reconstruction_loss(output, batch)
                else:
                    # Handle different batch formats
                    loss = torch.tensor(0.0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
