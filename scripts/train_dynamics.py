"""Training script for Dynamics Model"""

import argparse
import torch
import gc
import os
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Force unbuffered output to prevent "stuck" appearance
os.environ['PYTHONUNBUFFERED'] = '1'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dynamics import DynamicsModel
from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.data import VideoDataset, VideoTransform
from src.training import Trainer
from src.utils.config import load_config


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_dynamics_dataset(dataset, tokenizer, lam, sequence_length, device):
    """Create dataset that yields (tokens, actions, target_tokens)"""
    class DynamicsDataset:
        def __init__(self, base_dataset, tokenizer, lam, sequence_length, device):
            self.base_dataset = base_dataset
            self.tokenizer = tokenizer
            self.lam = lam
            self.sequence_length = sequence_length
            self.device = device
            self.access_count = 0
            self.cleanup_interval = 10  # Clear cache every N accesses
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            self.access_count += 1
            
            frames = self.base_dataset[idx].to(self.device)  # (T, C, H, W)
            
            # Tokenize frames
            frames_batch = frames.unsqueeze(0)  # (1, T, C, H, W)
            with torch.no_grad():
                tokens = self.tokenizer.encode(frames_batch)  # (1, T, H_patches, W_patches)
            tokens = tokens.squeeze(0)  # (T, H_patches, W_patches)
            del frames_batch  # Free memory
            
            # Get actions from LAM (spatial action maps)
            past_frames = frames[:-1].unsqueeze(0)  # (1, T-1, C, H, W)
            next_frame = frames[-1].unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                _, actions, _ = self.lam(past_frames, next_frame)  # (1, T-1, H_patches, W_patches)
            actions = actions.squeeze(0)  # (T-1, H_patches, W_patches)
            del past_frames, next_frame, frames  # Free memory
            
            # Keep full spatial action map (as per paper: additive embeddings use spatial actions)
            # Each spatial position has its own action embedding added to corresponding token
            
            # Target tokens are next tokens
            input_tokens = tokens[:-1]  # (T-1, H_patches, W_patches)
            target_tokens = tokens[1:]  # (T-1, H_patches, W_patches)
            del tokens  # Free memory
            
            # Periodic memory cleanup during data loading
            if self.access_count % self.cleanup_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Keep everything on GPU - pin_memory will be disabled for GPU tensors
            return input_tokens, actions, target_tokens
    
    return DynamicsDataset(dataset, tokenizer, lam, sequence_length, device)


def main():
    # Immediate output to confirm script started
    print("="*70, flush=True)
    print("Dynamics Model Training Script Started", flush=True)
    print(f"Time: {__import__('datetime').datetime.now()}", flush=True)
    print("="*70, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    parser = argparse.ArgumentParser(description="Train Dynamics Model")
    parser.add_argument("--config", type=str, default="configs/dynamics_config.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to use (e.g., 'pong', 'pole_position'). If None, uses all datasets")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer checkpoint")
    parser.add_argument("--lam_path", type=str, required=True, help="Path to trained LAM checkpoint")
    parser.add_argument("--lam_config", type=str, default="configs/lam_config.yaml", help="LAM config file path (must match checkpoint)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (overrides config)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints (auto-generated if not specified)")
    args = parser.parse_args()
    
    print(f"Arguments parsed successfully", flush=True)
    sys.stdout.flush()
    
    # Load config
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Override max_steps if provided
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
        print(f"Overriding max_steps to {args.max_steps}", flush=True)
    
    # Automatically generate checkpoint directory with timestamp if not specified
    if args.checkpoint_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = config.get('output', {}).get('checkpoint_dir', 'checkpoints/dynamics')
        config['output']['checkpoint_dir'] = f"{base_dir}/run_{timestamp}"
        print(f"Checkpoints will be saved to: {config['output']['checkpoint_dir']}", flush=True)
    else:
        config['output']['checkpoint_dir'] = args.checkpoint_dir
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}", flush=True)
    sys.stdout.flush()
    
    # Enable aggressive memory management for dynamics training
    # (dynamics uses 3 models: tokenizer, LAM, and dynamics model)
    config['training']['memory_offload_interval'] = 50  # More frequent offloading
    config['training']['aggressive_cleanup_interval'] = 10  # More frequent cleanup
    config['training']['memory_threshold_gb'] = 7.0  # Lower threshold for earlier cleanup
    print("Enabled aggressive memory management for dynamics training")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...", flush=True)
    sys.stdout.flush()
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    print("  Creating tokenizer model...", flush=True)
    sys.stdout.flush()
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    print("  Loading tokenizer checkpoint...", flush=True)
    sys.stdout.flush()
    tokenizer_checkpoint = torch.load(args.tokenizer_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in tokenizer_checkpoint:
        tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from step {tokenizer_checkpoint.get('global_step', 'unknown')}", flush=True)
    else:
        # Try loading directly (if checkpoint is just the state dict)
        tokenizer.load_state_dict(tokenizer_checkpoint)
        print("  Loaded checkpoint (direct state dict)", flush=True)
    
    print("  Moving tokenizer to device...", flush=True)
    sys.stdout.flush()
    tokenizer = tokenizer.to(device).eval()
    print(f"  Tokenizer codebook size: {tokenizer_config['model']['codebook']['num_codes']}", flush=True)
    sys.stdout.flush()
    
    # Load LAM
    print(f"Loading LAM from {args.lam_path}...", flush=True)
    print(f"Using LAM config: {args.lam_config}", flush=True)
    sys.stdout.flush()
    lam_config = load_config(args.lam_config)
    print("  Creating LAM model...", flush=True)
    sys.stdout.flush()
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    print("  Loading LAM checkpoint...", flush=True)
    sys.stdout.flush()
    lam_checkpoint = torch.load(args.lam_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in lam_checkpoint:
        lam.load_state_dict(lam_checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from step {lam_checkpoint.get('global_step', 'unknown')}", flush=True)
    else:
        # Try loading directly (if checkpoint is just the state dict)
        lam.load_state_dict(lam_checkpoint)
        print("  Loaded checkpoint (direct state dict)", flush=True)
    
    print("  Moving LAM to device...", flush=True)
    sys.stdout.flush()
    lam = lam.to(device).eval()
    print(f"  LAM codebook size: {lam_config['model']['codebook']['num_codes']}", flush=True)
    sys.stdout.flush()
    
    # Clear GPU memory after loading models
    print("Clearing GPU memory after model loading...")
    del tokenizer_checkpoint, lam_checkpoint
    clear_gpu_memory()
    
    # Create dataset
    transform = VideoTransform(
        normalize=True,
        horizontal_flip=True,
        random_crop=False,
    )
    
    base_dataset = VideoDataset(
        data_dir=args.data_dir,
        sequence_length=config['data']['sequence_length'],
        resolution=(128, 72),  # Default resolution
        transform=transform,
        dataset_name=args.dataset,
    )
    
    dataset = create_dynamics_dataset(
        base_dataset,
        tokenizer,
        lam,
        config['data']['sequence_length'],
        device,
    )
    
    # Create data loader
    num_workers = config['data'].get('num_workers', 0)
    # pin_memory only works with CPU tensors - disable it since we're keeping tensors on GPU
    # This is fine: with num_workers=0 and GPU tensors, pin_memory isn't needed anyway
    train_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disabled because tensors are already on GPU
    )
    
    # Create model
    print("Creating dynamics model...", flush=True)
    sys.stdout.flush()
    # Verify vocab_size matches tokenizer codebook size
    tokenizer_vocab_size = tokenizer_config['model']['codebook']['num_codes']
    dynamics_vocab_size = config['model']['token_embedding']['vocab_size']
    if tokenizer_vocab_size != dynamics_vocab_size:
        print(f"  WARNING: vocab_size mismatch! Tokenizer has {tokenizer_vocab_size} codes, but dynamics config has {dynamics_vocab_size}", flush=True)
        print(f"  Updating dynamics config to use vocab_size={tokenizer_vocab_size}", flush=True)
        config['model']['token_embedding']['vocab_size'] = tokenizer_vocab_size
    
    model = DynamicsModel(
        architecture_config=config['model']['architecture'],
        token_embedding_config=config['model']['token_embedding'],
        action_embedding_config=config['model']['action_embedding'],
    )
    print(f"  Dynamics model vocab_size: {config['model']['token_embedding']['vocab_size']}", flush=True)
    print(f"  Dynamics model num_actions: {config['model']['action_embedding']['num_actions']}", flush=True)
    sys.stdout.flush()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=args.device,
    )
    
    # Print training info
    print("\n" + "="*70, flush=True)
    print("Starting Dynamics Model Training", flush=True)
    print("="*70, flush=True)
    print(f"Max steps: {config['training']['max_steps']}", flush=True)
    print(f"Save every: {config['training'].get('save_every', 500)} steps", flush=True)
    print(f"Eval every: {config['training'].get('eval_every', 250)} steps", flush=True)
    print(f"Batch size: {config['data']['batch_size']}", flush=True)
    print(f"Sequence length: {config['data']['sequence_length']}", flush=True)
    print("="*70 + "\n", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Train
    trainer.train(max_steps=config['training']['max_steps'])
    
    # Save final checkpoint with standard name (in the same run directory)
    final_checkpoint_path = trainer.checkpoint_dir / "checkpoint.pth"
    
    checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'global_step': trainer.global_step,
        'epoch': trainer.epoch,
        'config': config,
    }
    torch.save(checkpoint, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
