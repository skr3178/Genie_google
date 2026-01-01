"""Script to check model sizes for all three models"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dynamics import DynamicsModel
from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.utils.config import load_config


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_size(params):
    """Format parameter count in readable format"""
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K"
    else:
        return str(params)


def main():
    print("=" * 80)
    print("Model Size Analysis")
    print("=" * 80)
    print()
    
    # Load configs
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    lam_config = load_config("configs/lam_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    
    total_all = 0
    
    # 1. Video Tokenizer
    print("1. Video Tokenizer (ST-ViViT)")
    print("-" * 80)
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    total_params, trainable_params = count_parameters(tokenizer)
    total_all += total_params
    print(f"  Total parameters: {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable parameters: {trainable_params:,} ({format_size(trainable_params)})")
    print(f"  Config: {tokenizer_config['model']['encoder']['num_layers']} encoder layers, "
          f"{tokenizer_config['model']['decoder']['num_layers']} decoder layers, "
          f"d_model={tokenizer_config['model']['encoder']['d_model']}")
    print()
    
    # 2. LAM
    print("2. Latent Action Model (LAM)")
    print("-" * 80)
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    total_params, trainable_params = count_parameters(lam)
    total_all += total_params
    print(f"  Total parameters: {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable parameters: {trainable_params:,} ({format_size(trainable_params)})")
    print(f"  Config: {lam_config['model']['encoder']['num_layers']} encoder layers, "
          f"{lam_config['model']['decoder']['num_layers']} decoder layers, "
          f"d_model={lam_config['model']['encoder']['d_model']}")
    print()
    
    # 3. Dynamics Model
    print("3. Dynamics Model")
    print("-" * 80)
    model = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    )
    total_params, trainable_params = count_parameters(model)
    total_all += total_params
    print(f"  Total parameters: {total_params:,} ({format_size(total_params)})")
    print(f"  Trainable parameters: {trainable_params:,} ({format_size(trainable_params)})")
    print(f"  Config: {dynamics_config['model']['architecture']['num_layers']} layers, "
          f"d_model={dynamics_config['model']['architecture']['d_model']}, "
          f"num_heads={dynamics_config['model']['architecture']['num_heads']}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total parameters (all models): {total_all:,} ({format_size(total_all)})")
    print()
    print("Note: During training, only one model is trained at a time:")
    print("  - Stage 1: Video Tokenizer only")
    print("  - Stage 2: LAM only (tokenizer frozen)")
    print("  - Stage 3: Dynamics Model only (tokenizer and LAM frozen)")
    print()


if __name__ == "__main__":
    main()
