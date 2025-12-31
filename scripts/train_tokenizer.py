"""Training script for Video Tokenizer"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.video_tokenizer import VideoTokenizer
from src.data import VideoDataset, VideoTransform
from src.training import Trainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train Video Tokenizer")
    parser.add_argument("--config", type=str, default="configs/tokenizer_config.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to use (e.g., 'pong', 'pole_position'). If None, uses all datasets")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (overrides config)")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N steps (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.save_every is not None:
        config['training']['save_every'] = args.save_every
    
    # Create dataset
    transform = VideoTransform(
        normalize=True,
        horizontal_flip=True,
        random_crop=False,
    )
    
    dataset = VideoDataset(
        data_dir=args.data_dir,
        sequence_length=config['data']['sequence_length'],
        resolution=tuple(config['data']['resolution'][:2]),
        transform=transform,
        dataset_name=args.dataset,
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
    model = VideoTokenizer(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
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
    
    # Save final checkpoint with standard name
    checkpoint_dir = Path(config.get('output', {}).get('checkpoint_dir', 'checkpoints/tokenizer'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / "checkpoint.pth"
    
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
