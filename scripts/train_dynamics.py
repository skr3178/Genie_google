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
            tokens = self.tokenizer.encode(frames_batch)  # (1, T, H_patches, W_patches)
            tokens = tokens.squeeze(0)  # (T, H_patches, W_patches)
            
            # Get actions from LAM (spatial action maps)
            past_frames = frames[:-1].unsqueeze(0)  # (1, T-1, C, H, W)
            next_frame = frames[-1].unsqueeze(0)  # (1, C, H, W)
            _, actions, _ = self.lam(past_frames, next_frame)  # (1, T-1, H_patches, W_patches)
            actions = actions.squeeze(0)  # (T-1, H_patches, W_patches)
            
            # Keep full spatial action map (as per paper: additive embeddings use spatial actions)
            # Each spatial position has its own action embedding added to corresponding token
            
            # Target tokens are next tokens
            input_tokens = tokens[:-1]  # (T-1, H_patches, W_patches)
            target_tokens = tokens[1:]  # (T-1, H_patches, W_patches)
            
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
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    tokenizer_checkpoint = torch.load(args.tokenizer_path, map_location=device)
    tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    tokenizer = tokenizer.to(device).eval()
    
    # Load LAM
    print("Loading LAM...")
    lam_config = load_config("configs/lam_config.yaml")
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    lam_checkpoint = torch.load(args.lam_path, map_location=device)
    lam.load_state_dict(lam_checkpoint['model_state_dict'])
    lam = lam.to(device).eval()
    
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
    train_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
    )
    
    # Create model
    model = DynamicsModel(
        architecture_config=config['model']['architecture'],
        token_embedding_config=config['model']['token_embedding'],
        action_embedding_config=config['model']['action_embedding'],
    )
    
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
