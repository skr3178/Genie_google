"""Training script for Dynamics Model"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dynamics import DynamicsModel
from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.data import VideoDataset, VideoTransform
from src.training import Trainer
from src.utils.config import load_config


def create_dynamics_dataset(dataset, tokenizer, lam, sequence_length, device):
    """Create dataset that yields (tokens, actions, target_tokens)"""
    class DynamicsDataset:
        def __init__(self, base_dataset, tokenizer, lam, sequence_length, device):
            self.base_dataset = base_dataset
            self.tokenizer = tokenizer
            self.lam = lam
            self.sequence_length = sequence_length
            self.device = device
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
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
            del past_frames, next_frame  # Free memory
            
            # Keep full spatial action map (as per paper: additive embeddings use spatial actions)
            # Each spatial position has its own action embedding added to corresponding token
            
            # Target tokens are next tokens
            input_tokens = tokens[:-1]  # (T-1, H_patches, W_patches)
            target_tokens = tokens[1:]  # (T-1, H_patches, W_patches)
            del tokens  # Free memory
            
            # Keep everything on GPU - pin_memory will be disabled for GPU tensors
            return input_tokens, actions, target_tokens
    
    return DynamicsDataset(dataset, tokenizer, lam, sequence_length, device)


def main():
    parser = argparse.ArgumentParser(description="Train Dynamics Model")
    parser.add_argument("--config", type=str, default="configs/dynamics_config.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to use (e.g., 'pong', 'pole_position'). If None, uses all datasets")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer checkpoint")
    parser.add_argument("--lam_path", type=str, required=True, help="Path to trained LAM checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Override max_steps if provided
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
        print(f"Overriding max_steps to {args.max_steps}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    tokenizer_checkpoint = torch.load(args.tokenizer_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in tokenizer_checkpoint:
        tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from step {tokenizer_checkpoint.get('global_step', 'unknown')}")
    else:
        # Try loading directly (if checkpoint is just the state dict)
        tokenizer.load_state_dict(tokenizer_checkpoint)
        print("  Loaded checkpoint (direct state dict)")
    
    tokenizer = tokenizer.to(device).eval()
    print(f"  Tokenizer codebook size: {tokenizer_config['model']['codebook']['num_codes']}")
    
    # Load LAM
    print(f"Loading LAM from {args.lam_path}...")
    lam_config = load_config("configs/lam_config.yaml")
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    lam_checkpoint = torch.load(args.lam_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in lam_checkpoint:
        lam.load_state_dict(lam_checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from step {lam_checkpoint.get('global_step', 'unknown')}")
    else:
        # Try loading directly (if checkpoint is just the state dict)
        lam.load_state_dict(lam_checkpoint)
        print("  Loaded checkpoint (direct state dict)")
    
    lam = lam.to(device).eval()
    print(f"  LAM codebook size: {lam_config['model']['codebook']['num_codes']}")
    
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
    print("Creating dynamics model...")
    # Verify vocab_size matches tokenizer codebook size
    tokenizer_vocab_size = tokenizer_config['model']['codebook']['num_codes']
    dynamics_vocab_size = config['model']['token_embedding']['vocab_size']
    if tokenizer_vocab_size != dynamics_vocab_size:
        print(f"  WARNING: vocab_size mismatch! Tokenizer has {tokenizer_vocab_size} codes, but dynamics config has {dynamics_vocab_size}")
        print(f"  Updating dynamics config to use vocab_size={tokenizer_vocab_size}")
        config['model']['token_embedding']['vocab_size'] = tokenizer_vocab_size
    
    model = DynamicsModel(
        architecture_config=config['model']['architecture'],
        token_embedding_config=config['model']['token_embedding'],
        action_embedding_config=config['model']['action_embedding'],
    )
    print(f"  Dynamics model vocab_size: {config['model']['token_embedding']['vocab_size']}")
    print(f"  Dynamics model num_actions: {config['model']['action_embedding']['num_actions']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=args.device,
    )
    
    # Train
    trainer.train(max_steps=config['training']['max_steps'])
    
    print("Training completed!")


if __name__ == "__main__":
    main()
