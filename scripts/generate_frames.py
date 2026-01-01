"""Generate frames using dynamics checkpoints from checkpoints/dynamics directory"""

import torch
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import argparse
from datetime import datetime
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
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


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str = "*.pt"):
    """Find the latest checkpoint in a directory"""
    checkpoints = list(checkpoint_dir.glob(pattern)) + list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Generate frames using dynamics checkpoints")
    parser.add_argument("--dynamics_dir", type=str, 
                       default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/dynamics",
                       help="Directory containing dynamics checkpoints")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer checkpoint (auto-finds latest if not specified)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Path to prompt image (creates dummy if not specified)")
    parser.add_argument("--actions", type=str, default="0,1,2,3,4,5,6,7",
                       help="Comma-separated action indices (0-7)")
    parser.add_argument("--output", type=str, default="generated_frames",
                       help="Output directory for frames")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                       help="Specific checkpoint step to use (e.g., 2000). Uses latest if not specified")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find dynamics checkpoint
    dynamics_dir = Path(args.dynamics_dir)
    if not dynamics_dir.exists():
        print(f"Error: Dynamics checkpoint directory not found: {dynamics_dir}")
        return
    
    if args.checkpoint_step is not None:
        dynamics_path = dynamics_dir / f"checkpoint_step_{args.checkpoint_step}.pt"
        if not dynamics_path.exists():
            dynamics_path = dynamics_dir / f"checkpoint_step_{args.checkpoint_step}.pth"
    else:
        dynamics_path = find_latest_checkpoint(dynamics_dir)
    
    if dynamics_path is None or not dynamics_path.exists():
        print(f"Error: No dynamics checkpoint found in {dynamics_dir}")
        return
    
    print(f"Using dynamics checkpoint: {dynamics_path}")
    
    # Find tokenizer checkpoint
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
    else:
        tokenizer_dir = Path("checkpoints/tokenizer")
        tokenizer_path = find_latest_checkpoint(tokenizer_dir)
    
    if tokenizer_path is None or not tokenizer_path.exists():
        print(f"Error: No tokenizer checkpoint found. Please specify --tokenizer_path")
        return
    
    print(f"Using tokenizer checkpoint: {tokenizer_path}")
    
    # Load configs
    print("\nLoading configs...")
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    
    # Create models
    print("Creating models...")
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
    tokenizer_checkpoint = torch.load(tokenizer_path, map_location=device)
    if 'model_state_dict' in tokenizer_checkpoint:
        tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    else:
        tokenizer.load_state_dict(tokenizer_checkpoint)
    print(f"  ✓ Loaded tokenizer from {tokenizer_path}")
    
    dynamics_checkpoint = torch.load(dynamics_path, map_location=device)
    if 'model_state_dict' in dynamics_checkpoint:
        dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
    else:
        dynamics_model.load_state_dict(dynamics_checkpoint)
    print(f"  ✓ Loaded dynamics model from {dynamics_path}")
    
    # Create generator
    generator = GenieGenerator(
        tokenizer=tokenizer,
        dynamics_model=dynamics_model,
        device=device,
        maskgit_steps=dynamics_config.get('inference', {}).get('maskgit_steps', 25),
        temperature=dynamics_config.get('inference', {}).get('temperature', 2.0),
    )
    
    # Load or create prompt image
    if args.prompt:
        print(f"Loading prompt image from {args.prompt}...")
        prompt_frame = load_image(args.prompt)
    else:
        print("No prompt specified, creating dummy prompt frame...")
        prompt_frame = torch.rand(3, 128, 72)  # (C, H, W)
    
    # Parse actions
    actions = [int(a) for a in args.actions.split(',')]
    
    # Generate video
    print(f"\nGenerating {args.num_frames} frames...")
    print(f"  Actions: {actions}")
    video = generator.generate(prompt_frame, actions, num_frames=args.num_frames)
    
    # Convert to numpy and prepare for saving
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)
    print(f"Video generated! Shape: {video_np.shape}")
    
    # Convert from (T, C, H, W) to (T, H, W, C) for saving
    video_rgb = video_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    
    # Create timestamped subfolder within output directory
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get checkpoint step for folder name
    checkpoint_step = args.checkpoint_step
    if checkpoint_step is None:
        # Extract step from checkpoint path
        checkpoint_name = dynamics_path.stem
        if 'step_' in checkpoint_name:
            try:
                checkpoint_step = int(checkpoint_name.split('step_')[1].split('.')[0])
            except:
                checkpoint_step = "latest"
        else:
            checkpoint_step = "latest"
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}_step_{checkpoint_step}"
    output_dir = base_output_dir / run_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual frames
    for i, frame in enumerate(video_rgb):
        frame_path = output_dir / f"frame_{i:04d}.png"
        Image.fromarray(frame).save(frame_path)
    
    print(f"✓ Saved {len(video_rgb)} frames to {output_dir}/")
    
    # Save video file using H.264 codec
    video_path = output_dir / "generated_video.mp4"
    fps = 10.0
    
    video_saved = False
    if IMAGEIO_AVAILABLE:
        # Use imageio for reliable H.264 encoding
        try:
            # imageio uses ffmpeg which properly supports H.264 in MP4 containers
            imageio.mimwrite(
                str(video_path),
                video_rgb,
                fps=fps,
                codec='libx264',  # H.264 codec
                quality=8,  # High quality (0-10 scale, 10 is best)
                pixelformat='yuv420p'  # Ensures compatibility
            )
            # Verify file was created
            if video_path.exists() and video_path.stat().st_size > 1000:
                print(f"✓ Saved video to {video_path} using H.264 (libx264)")
                video_saved = True
            else:
                print(f"  Warning: Video file appears to be too small or missing")
        except Exception as e:
            print(f"  Warning: Error creating video with imageio: {e}")
            print(f"  Falling back to OpenCV...")
    
    if not video_saved:
        # Fallback to OpenCV with better H.264 settings
        try:
            import cv2
            H, W = video_rgb.shape[1:3]
            
            # Try H.264 codecs in order of preference
            h264_codecs = [
                ('H264', 'H.264'),  # Try H264 first
                ('avc1', 'H.264/AVC'),  # Then avc1
                ('X264', 'x264'),  # x264 encoder
            ]
            
            for codec_name, codec_desc in h264_codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    temp_path = str(video_path).replace('.mp4', f'_temp_{codec_name}.mp4')
                    out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
                    
                    if not out.isOpened():
                        continue
                    
                    for frame in video_rgb:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    out.release()
                    
                    # Check if file was written successfully
                    if Path(temp_path).exists() and Path(temp_path).stat().st_size > 1000:
                        import shutil
                        shutil.move(temp_path, str(video_path))
                        print(f"✓ Saved video to {video_path} using {codec_desc}")
                        video_saved = True
                        break
                    else:
                        if Path(temp_path).exists():
                            Path(temp_path).unlink()
                except Exception as e:
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                    continue
            
            if not video_saved:
                print(f"⚠ Failed to save as H.264 MP4, but frames are available in {output_dir}/")
        except ImportError:
            print(f"⚠ OpenCV not available, and imageio failed. Frames are available in {output_dir}/")
    
    print(f"\n✓ Generation completed!")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"    - {len(video_rgb)} frames (frame_*.png)")
    if video_saved:
        print(f"    - Video file: generated_video.mp4")


if __name__ == "__main__":
    main()
