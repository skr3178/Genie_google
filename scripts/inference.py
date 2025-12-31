"""Inference script for Genie model"""

import argparse
import torch
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import GenieGenerator
from src.models.video_tokenizer import VideoTokenizer
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
    parser.add_argument("--dynamics_path", type=str, required=True, help="Path to dynamics checkpoint")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path (or directory for frames)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--save_frames", action="store_true", help="Save as individual PNG frames instead of video")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load configs
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    
    # Create models
    print("Loading models...")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    
    dynamics_model = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    )
    
    # Load checkpoints
    print("Loading checkpoints...")
    print("  Note: LAM is NOT needed for inference - only tokenizer and dynamics model are used.")
    
    tokenizer_checkpoint = torch.load(args.tokenizer_path, map_location=device)
    if 'model_state_dict' in tokenizer_checkpoint:
        tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    else:
        tokenizer.load_state_dict(tokenizer_checkpoint)
    print(f"  ✓ Loaded tokenizer from {args.tokenizer_path}")
    
    dynamics_checkpoint = torch.load(args.dynamics_path, map_location=device)
    if 'model_state_dict' in dynamics_checkpoint:
        dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
    else:
        dynamics_model.load_state_dict(dynamics_checkpoint)
    print(f"  ✓ Loaded dynamics model from {args.dynamics_path}")
    
    # Create generator
    generator = GenieGenerator(
        tokenizer=tokenizer,
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
    
    # Convert to numpy and prepare for saving
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)
    print(f"Video generated! Shape: {video_np.shape}")
    
    # Convert from (T, C, H, W) to (T, H, W, C) for saving
    video_rgb = video_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    
    # Save video or frames
    output_path = Path(args.output)
    
    if args.save_frames or output_path.suffix != '.mp4':
        # Save as individual frames (works better in Cursor)
        if output_path.suffix == '.mp4':
            frames_dir = output_path.with_suffix('')
        else:
            frames_dir = output_path
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(video_rgb):
            frame_path = frames_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)
        print(f"✓ Saved {len(video_rgb)} frames to {frames_dir}/")
        print(f"  You can view these frames directly in Cursor!")
    else:
        # Save as MP4 using OpenCV with better codec
        H, W = video_rgb.shape[1:3]
        fps = 10.0
        
        # Try different codecs in order of preference
        codecs = [
            ('avc1', 'H.264/AVC'),  # Best compatibility
            ('mp4v', 'MPEG-4'),      # Fallback
            ('XVID', 'Xvid'),        # Another fallback
        ]
        
        video_saved = False
        for codec_name, codec_desc in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                temp_path = str(output_path).replace('.mp4', f'_temp_{codec_name}.mp4')
                out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
                
                if not out.isOpened():
                    print(f"  Warning: Failed to open VideoWriter with {codec_desc}, trying next...")
                    continue
                
                for frame in video_rgb:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                
                # Check if file was written successfully
                if Path(temp_path).stat().st_size > 1000:  # At least 1KB
                    import shutil
                    shutil.move(temp_path, str(output_path))
                    print(f"✓ Saved video to {output_path} using {codec_desc}")
                    video_saved = True
                    break
                else:
                    Path(temp_path).unlink()  # Delete small/corrupted file
            except Exception as e:
                print(f"  Warning: Error with {codec_desc}: {e}")
                continue
        
        if not video_saved:
            print(f"⚠ Failed to save as MP4, saving as individual frames instead...")
            # Fallback to frames
            frames_dir = output_path.with_suffix('')
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(video_rgb):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
            print(f"✓ Saved {len(video_rgb)} frames to {frames_dir}/")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
