"""Inference script for Genie model"""

import argparse
import torch
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import GenieGenerator
from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.models.dynamics import DynamicsModel
from src.utils.config import load_config


def load_image(path: str) -> torch.Tensor:
    """Load image and convert to tensor"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # (C, H, W)
    img = torch.from_numpy(img).float() / 255.0
    return img


def main():
    parser = argparse.ArgumentParser(description="Run Genie inference")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt image")
    parser.add_argument("--actions", type=str, required=True, help="Comma-separated action indices (0-7)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--lam_path", type=str, required=True, help="Path to LAM checkpoint")
    parser.add_argument("--dynamics_path", type=str, required=True, help="Path to dynamics checkpoint")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load configs
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    lam_config = load_config("configs/lam_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    
    # Create models
    print("Loading models...")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    
    dynamics_model = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    )
    
    # Load checkpoints
    tokenizer_checkpoint = torch.load(args.tokenizer_path, map_location=device)
    tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    
    lam_checkpoint = torch.load(args.lam_path, map_location=device)
    lam.load_state_dict(lam_checkpoint['model_state_dict'])
    
    dynamics_checkpoint = torch.load(args.dynamics_path, map_location=device)
    dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
    
    # Create generator
    generator = GenieGenerator(
        tokenizer=tokenizer,
        lam=lam,
        dynamics_model=dynamics_model,
        device=device,
        maskgit_steps=dynamics_config.get('inference', {}).get('maskgit_steps', 25),
        temperature=dynamics_config.get('inference', {}).get('temperature', 2.0),
    )
    
    # Load prompt image
    prompt_frame = load_image(args.prompt)
    
    # Parse actions
    actions = [int(a) for a in args.actions.split(',')]
    
    # Generate video
    print(f"Generating {args.num_frames} frames...")
    video = generator.generate(prompt_frame, actions, num_frames=args.num_frames)
    
    # Save video (convert to numpy and save)
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)
    # Save as frames or use video writer
    print(f"Video generated! Shape: {video_np.shape}")
    print(f"Save to {args.output} (implement video saving)")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
