"""Main training loop"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys
from tqdm import tqdm
import gc

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
        
        # Logging (initialize early so we can use it)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Mixed precision (fix deprecation warning)
        self.use_amp = self.config.get('mixed_precision', True)
        try:
            # Try new API first (PyTorch 2.0+)
            from torch.amp import GradScaler as NewGradScaler
            self.scaler = NewGradScaler('cuda') if self.use_amp else None
        except ImportError:
            # Fall back to old API
            self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient checkpointing
        if self.config.get('training', {}).get('gradient_checkpointing', False):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config.get('training', {}))
        num_steps = len(train_loader) * self.config.get('training', {}).get('max_steps', 10000) // len(train_loader)
        # #region agent log
        import json
        from pathlib import Path
        log_path = Path("/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log")
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "trainer.py:53", "message": "scheduler creation", "data": {"num_steps": num_steps, "max_steps": self.config.get('training', {}).get('max_steps', 10000), "train_loader_len": len(train_loader)}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
        except: pass
        # #endregion
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
        
        # Memory management configuration
        self.memory_offload_interval = self.config.get('training', {}).get('memory_offload_interval', 100)  # More frequent offloading
        self.aggressive_cleanup_interval = self.config.get('training', {}).get('aggressive_cleanup_interval', 25)  # More frequent cleanup
        self.memory_threshold_gb = self.config.get('training', {}).get('memory_threshold_gb', 8.5)  # Trigger checkpoint if memory > 8.5GB
        self.last_checkpoint_memory_gb = 0.0
        self.light_cleanup_interval = 10  # Light cleanup every 10 steps
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        # Use set_to_none=True for more memory-efficient gradient clearing
        self.optimizer.zero_grad(set_to_none=True)
        
        # Print first step to confirm training started
        if self.global_step == 0:
            print(f"Training started! Processing first batch...", flush=True)
            sys.stdout.flush()
        
        # Clear cache before forward pass to free up memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Move batch to device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        
        # Forward pass with mixed precision
        try:
            # Try new API first (PyTorch 2.0+)
            from torch.amp import autocast as new_autocast
            with new_autocast('cuda', enabled=self.use_amp):
                if isinstance(batch, torch.Tensor):
                    # Video tokenizer or simple case
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        pred, tokens, vq_loss_dict = output
                        vq_losses = vq_loss(vq_loss_dict)
                        loss = reconstruction_loss(pred, batch) + vq_losses['vq_loss']
                        # Explicitly delete intermediate tensors to free memory
                        del pred, tokens, vq_loss_dict, vq_losses
                    else:
                        loss = reconstruction_loss(output, batch)
                        del output
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
        except ImportError:
            # Fall back to old API
            with autocast(enabled=self.use_amp):
                if isinstance(batch, torch.Tensor):
                    # Video tokenizer or simple case
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        pred, tokens, vq_loss_dict = output
                        vq_losses = vq_loss(vq_loss_dict)
                        loss = reconstruction_loss(pred, batch) + vq_losses['vq_loss']
                        # Explicitly delete intermediate tensors to free memory
                        del pred, tokens, vq_loss_dict, vq_losses
                    else:
                        loss = reconstruction_loss(output, batch)
                        del output
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
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            msg = f"NaN/Inf loss detected at step {self.global_step}, skipping step"
            self.logger.warning(msg)
            print(f"WARNING: {msg}", flush=True)
            # Check if model has NaN parameters
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    error_msg = f"NaN/Inf detected in parameter {name} at step {self.global_step}"
                    self.logger.error(error_msg)
                    print(f"ERROR: {error_msg}", flush=True)
            return {'loss': float('nan')}
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # Gradient clipping (always apply if max_grad_norm is set)
            if self.config.get('training', {}).get('max_grad_norm'):
                self.scaler.unscale_(self.optimizer)
                # Check for NaN gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training', {}).get('max_grad_norm', 1.0),
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"NaN/Inf gradients detected at step {self.global_step}, skipping step")
                    self.optimizer.zero_grad()
                    # Must call update() to reset scaler state after unscale_()
                    self.scaler.update()
                    # Clear cache after NaN to free memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    return {'loss': float('nan')}
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping (always apply if max_grad_norm is set)
            if self.config.get('training', {}).get('max_grad_norm'):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training', {}).get('max_grad_norm', 1.0),
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"NaN/Inf gradients detected at step {self.global_step}, skipping step")
                    self.optimizer.zero_grad()
                    # Clear cache after NaN to free memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    return {'loss': float('nan')}
            self.optimizer.step()
        
        # Clear cache after EVERY optimizer step to prevent fragmentation
        # Also zero gradients with set_to_none=True to free gradient memory immediately
        if self.device == 'cuda':
            # More aggressive cache clearing after optimizer step
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Use set_to_none=True to free gradient memory immediately (more memory efficient)
            # This is done here because we already called zero_grad() at the start, but
            # set_to_none=True ensures gradients are actually freed, not just zeroed
        
        # #region agent log
        import json
        from pathlib import Path
        log_path = Path("/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log")
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "trainer.py:174", "message": "before scheduler.step", "data": {"global_step": self.global_step, "max_steps": self.config.get('training', {}).get('max_steps', 10000)}, "timestamp": int(__import__('time').time() * 1000)}) + "\n")
        except: pass
        # #endregion
        self.scheduler.step()
        self.global_step += 1
        
        # #region agent log
        import json
        import time
        log_path = Path("/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log")
        if self.device == 'cuda':
            try:
                mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3  # GB
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "trainer.py:193", "message": "memory_after_step", "data": {"step": self.global_step, "allocated_gb": round(mem_allocated, 3), "reserved_gb": round(mem_reserved, 3), "free_gb": round(mem_free, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
            except: pass
        # #endregion
        
        # Aggressive memory management with periodic GPU offloading
        if self.device == 'cuda':
            # Light cleanup every N steps (more frequent)
            if self.global_step % self.light_cleanup_interval == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Force garbage collection for small objects
                if self.global_step % (self.light_cleanup_interval * 2) == 0:
                    gc.collect()
            
            # Aggressive cleanup every N steps (default 50, reduced from 100)
            if self.global_step % self.aggressive_cleanup_interval == 0:
                # #region agent log
                try:
                    mem_before = torch.cuda.memory_allocated() / 1024**3
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "trainer.py:210", "message": "before_aggressive_cleanup", "data": {"step": self.global_step, "mem_gb": round(mem_before, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # #region agent log
                try:
                    mem_after = torch.cuda.memory_allocated() / 1024**3
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "trainer.py:220", "message": "after_aggressive_cleanup", "data": {"step": self.global_step, "mem_gb": round(mem_after, 3), "freed_gb": round(mem_before - mem_after, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
                except: pass
                # #endregion
            
            # Check memory threshold and trigger checkpoint-based recovery if needed
            current_mem_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_mem_gb = torch.cuda.memory_reserved() / 1024**3
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Log memory usage periodically for monitoring
            if self.global_step % 100 == 0:
                self.logger.debug(f"Step {self.global_step}: Memory - Allocated: {current_mem_gb:.2f}GB, Reserved: {reserved_mem_gb:.2f}GB, Total: {total_mem_gb:.2f}GB")
            
            # Warn if memory usage is getting high (above 90% of total)
            if current_mem_gb > total_mem_gb * 0.9:
                self.logger.warning(f"High memory usage detected: {current_mem_gb:.2f}GB / {total_mem_gb:.2f}GB ({current_mem_gb/total_mem_gb*100:.1f}%) at step {self.global_step}")
            
            if current_mem_gb > self.memory_threshold_gb and self.global_step > 0:
                # Only checkpoint if memory increased significantly since last checkpoint
                if current_mem_gb - self.last_checkpoint_memory_gb > 0.5:  # At least 500MB increase
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "trainer.py:245", "message": "memory_threshold_exceeded", "data": {"step": self.global_step, "mem_gb": round(current_mem_gb, 3), "threshold_gb": self.memory_threshold_gb}, "timestamp": int(time.time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    
                    self.logger.warning(f"Memory threshold exceeded ({current_mem_gb:.2f}GB > {self.memory_threshold_gb}GB) at step {self.global_step}. Saving checkpoint and performing aggressive cleanup.")
                    # Save checkpoint to reset memory state
                    self.save_checkpoint()
                    self.last_checkpoint_memory_gb = current_mem_gb
                    
                    # Aggressive cleanup after checkpoint
                    self._offload_gpu_memory()
                    
                    # #region agent log
                    try:
                        mem_after_recovery = torch.cuda.memory_allocated() / 1024**3
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "trainer.py:260", "message": "after_memory_recovery", "data": {"step": self.global_step, "mem_gb": round(mem_after_recovery, 3), "freed_gb": round(current_mem_gb - mem_after_recovery, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
            
            # Periodic GPU offloading to prevent memory accumulation (every N steps, default 200, reduced from 500)
            if self.global_step % self.memory_offload_interval == 0 and self.global_step > 0:
                # #region agent log
                try:
                    mem_before_offload = torch.cuda.memory_allocated() / 1024**3
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "trainer.py:228", "message": "before_gpu_offload", "data": {"step": self.global_step, "mem_gb": round(mem_before_offload, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                self._offload_gpu_memory()
                
                # #region agent log
                try:
                    mem_after_offload = torch.cuda.memory_allocated() / 1024**3
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "trainer.py:236", "message": "after_gpu_offload", "data": {"step": self.global_step, "mem_gb": round(mem_after_offload, 3), "freed_gb": round(mem_before_offload - mem_after_offload, 3)}, "timestamp": int(time.time() * 1000)}) + "\n")
                except: pass
                # #endregion
        
        return {'loss': loss.item()}
    
    def train(self, max_steps: Optional[int] = None):
        """Main training loop"""
        max_steps = max_steps or self.config.get('training', {}).get('max_steps', 10000)
        
        # Use stderr for tqdm (unbuffered) and disable buffering
        pbar = tqdm(
            total=max_steps, 
            desc="Training",
            file=sys.stderr,  # Use stderr (unbuffered by default)
            dynamic_ncols=True,
            mininterval=0.5,  # Update at least every 0.5 seconds
            maxinterval=1.0,  # Update at most every 1 second
        )
        
        # Print initial status
        print(f"Starting training for {max_steps} steps...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        
        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                try:
                    metrics = self.train_step(batch)
                    pbar.update(1)
                    pbar.set_postfix(metrics)
                    pbar.refresh()  # Force refresh
                    
                    # Periodic flush to ensure output appears
                    if self.global_step % 10 == 0:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    
                    # Evaluation
                    if self.val_loader and self.global_step % self.eval_every == 0:
                        val_metrics = self.evaluate()
                        self.logger.info(f"Step {self.global_step}: {val_metrics}")
                        print(f"Step {self.global_step}: {val_metrics}", flush=True)
                    
                    # Checkpointing
                    if self.global_step % self.save_every == 0:
                        self.save_checkpoint()
                        print(f"Checkpoint saved at step {self.global_step}", flush=True)
                        
                except torch.cuda.OutOfMemoryError as e:
                    self.logger.error(f"CUDA OOM at step {self.global_step}: {e}")
                    # Aggressive cleanup to try to recover
                    if self.device == 'cuda':
                        try:
                            # Clear all gradients
                            self.optimizer.zero_grad(set_to_none=True)
                            # Aggressive memory cleanup
                            self._offload_gpu_memory()
                            # Try to free up more memory
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception as cleanup_error:
                            self.logger.warning(f"Error during OOM cleanup: {cleanup_error}")
                    # Save checkpoint before exiting
                    self.save_checkpoint()
                    self.logger.warning(f"Saved checkpoint at step {self.global_step} due to OOM")
                    raise  # Re-raise to stop training
            
            self.epoch += 1
        
        pbar.close()
        self.save_checkpoint()  # Final checkpoint
        print(f"\nTraining completed! Final step: {self.global_step}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
    
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
    
    def _offload_gpu_memory(self):
        """Aggressively offload GPU memory by clearing caches and forcing garbage collection"""
        if self.device != 'cuda':
            return
        
        try:
            # Step 1: Zero out gradients to free gradient memory
            self.optimizer.zero_grad(set_to_none=True)
            
            # Step 2: Synchronize all CUDA operations to ensure everything is done
            torch.cuda.synchronize()
            
            # Step 3: Force Python garbage collection to free unreferenced objects
            gc.collect()
            
            # Step 4: Clear CUDA cache multiple times with synchronization
            # This helps with memory fragmentation
            for _ in range(15):  # Increased from 10 to 15 for even more aggressive clearing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Step 5: Reset peak memory stats to get accurate readings
            torch.cuda.reset_peak_memory_stats()
            
            # Step 6: Additional aggressive cleanup - try to compact memory
            # This forces PyTorch to release fragmented memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            self.logger.debug(f"GPU memory offload completed at step {self.global_step}")
            
        except Exception as e:
            self.logger.warning(f"Error during GPU memory offload at step {self.global_step}: {e}")
            # Fallback: just clear cache
            torch.cuda.empty_cache()
            gc.collect()
