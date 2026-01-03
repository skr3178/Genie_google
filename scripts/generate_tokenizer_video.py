"""Generate a video from tokenizer checkpoint by encoding/decoding a dataset sequence"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False

from src.models.video_tokenizer import VideoTokenizer
from src.utils.config import load_config


def load_video_sequence(h5_path: str, start_idx: int, sequence_length: int, resolution: tuple = (128, 72)):
    """
    Load a sequence of frames from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Starting frame index
        sequence_length: Number of frames to load
        resolution: (H, W) target resolution
    
    Returns:
        Video tensor of shape (T, C, H, W) normalized to [0, 1]
    """
    with h5py.File(h5_path, 'r') as f:
        if 'frames' in f:
            frames = f['frames']
        else:
            first_key = list(f.keys())[0]
            frames = f[first_key]
        
        num_frames = frames.shape[0]
        
        # Check bounds
        if start_idx + sequence_length > num_frames:
            start_idx = max(0, num_frames - sequence_length)
        
        # Load frames
        frame_data = frames[start_idx:start_idx + sequence_length]  # (T, H, W, 3)
        
        # Convert to numpy if needed
        if isinstance(frame_data, h5py.Dataset):
            frame_data = frame_data[:]
        
        # Resize if needed
        if frame_data.shape[1:3] != resolution:
            resized_frames = []
            for frame in frame_data:
                img = Image.fromarray(frame)
                img = img.resize((resolution[1], resolution[0]), Image.Resampling.LANCZOS)  # (W, H)
                resized_frames.append(np.array(img))
            frame_data = np.array(resized_frames)
        
        # Convert to (T, C, H, W) and normalize to [0, 1]
        frame_data = frame_data.transpose(0, 3, 1, 2)  # (T, C, H, W)
        frame_data = frame_data.astype(np.float32) / 255.0
        
        return torch.from_numpy(frame_data).float()


def save_video(frames: np.ndarray, output_path: Path, fps: float = 10.0):
    """Save frames as a video file"""
    
    video_saved = False
    
    if IMAGEIO_AVAILABLE:
        try:
            imageio.mimwrite(
                str(output_path),
                frames,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"  ✓ Saved video to {output_path} using H.264 (libx264)")
                video_saved = True
        except Exception as e:
            print(f"  Warning: Error creating video with imageio: {e}")
            if CV2_AVAILABLE:
                print(f"  Falling back to OpenCV...")
    
    if not video_saved and CV2_AVAILABLE:
        try:
            H, W = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            temp_path = str(output_path).replace('.mp4', '_temp.mp4')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            
            if out.isOpened():
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                if Path(temp_path).stat().st_size > 1000:
                    import shutil
                    shutil.move(temp_path, str(output_path))
                    print(f"  ✓ Saved video to {output_path} using OpenCV (H.264)")
                    video_saved = True
                else:
                    Path(temp_path).unlink()
        except Exception as e:
            print(f"  Warning: Error with OpenCV: {e}")
    
    if not video_saved:
        raise RuntimeError("Failed to create video with both imageio and OpenCV")
    
    return video_saved


def main():
    parser = argparse.ArgumentParser(description="Generate video from tokenizer checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pong_frames.h5",
        help="Path to dataset HDF5 file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizer_config.yaml",
        help="Tokenizer config file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output video path (default: evaluations/tokenizer/video_checkpoint_step_XXXX.mp4)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting frame index in dataset"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating Video from Tokenizer Checkpoint")
    print("=" * 70)
    
    # Calculate number of frames needed
    num_frames = int(args.duration * args.fps)
    print(f"\nTarget: {args.duration} seconds at {args.fps} fps = {num_frames} frames")
    
    # Load config
    print("\n1. Loading config...")
    config = load_config(args.config)
    resolution = tuple(config['data']['resolution'][:2])  # (H, W)
    print(f"   Resolution: {resolution}")
    
    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = VideoTokenizer(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    ).to(args.device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"   ✗ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('global_step', 'unknown')
    else:
        tokenizer.load_state_dict(checkpoint)
        step = 'unknown'
    
    tokenizer = tokenizer.to(args.device)
    tokenizer.eval()
    print(f"   ✓ Loaded tokenizer from {checkpoint_path}")
    print(f"   Training step: {step}")
    
    # Load dataset
    print("\n3. Loading video sequences...")
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"   ✗ Error: Dataset not found at {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        if 'frames' in f:
            total_frames = f['frames'].shape[0]
        else:
            first_key = list(f.keys())[0]
            total_frames = f[first_key].shape[0]
    
    print(f"   Total frames in dataset: {total_frames}")
    
    # Check if we have enough frames
    if args.start_idx + num_frames > total_frames:
        print(f"   ⚠ Warning: Not enough frames. Adjusting start_idx...")
        args.start_idx = max(0, total_frames - num_frames)
    
    print(f"   Loading frames {args.start_idx} to {args.start_idx + num_frames}...")
    
    # Load video sequence
    video = load_video_sequence(
        str(data_path),
        args.start_idx,
        num_frames,
        resolution=resolution
    )  # (T, C, H, W)
    
    # Process in batches if needed (to handle memory)
    sequence_length = config['data'].get('sequence_length', 8)
    all_reconstructed_frames = []
    
    print("\n4. Running tokenizer inference...")
    with torch.no_grad():
        # Process in chunks of sequence_length
        for i in range(0, num_frames, sequence_length):
            end_idx = min(i + sequence_length, num_frames)
            chunk = video[i:end_idx]  # (chunk_T, C, H, W)
            
            # Pad if needed to match sequence_length
            if chunk.shape[0] < sequence_length:
                padding = torch.zeros(
                    sequence_length - chunk.shape[0],
                    chunk.shape[1],
                    chunk.shape[2],
                    chunk.shape[3]
                )
                chunk = torch.cat([chunk, padding], dim=0)
            
            # Add batch dimension: (1, T, C, H, W)
            video_batch = chunk.unsqueeze(0).to(args.device)
            
            # Encode and decode
            reconstructed, tokens, vq_loss_dict = tokenizer(video_batch)
            
            # Remove batch dimension and padding
            reconstructed_single = reconstructed[0].cpu()  # (T, C, H, W)
            actual_length = end_idx - i
            reconstructed_single = reconstructed_single[:actual_length]
            
            all_reconstructed_frames.append(reconstructed_single)
            
            # Clear GPU memory
            del video_batch, reconstructed, tokens, vq_loss_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all frames
    reconstructed_video = torch.cat(all_reconstructed_frames, dim=0)  # (total_T, C, H, W)
    
    print(f"   ✓ Generated {reconstructed_video.shape[0]} frames")
    print(f"   Reconstructed stats: min={reconstructed_video.min():.6f}, max={reconstructed_video.max():.6f}, mean={reconstructed_video.mean():.6f}")
    
    # Normalize if needed
    if reconstructed_video.max() < 0.01:
        print(f"   ⚠ WARNING: Reconstructed values are very small, attempting to normalize...")
        recon_min = reconstructed_video.min()
        recon_max = reconstructed_video.max()
        if recon_max > recon_min:
            reconstructed_video = (reconstructed_video - recon_min) / (recon_max - recon_min)
            print(f"   Re-scaled to: min={reconstructed_video.min():.6f}, max={reconstructed_video.max():.6f}")
    
    # Convert to numpy for video creation
    print("\n5. Creating video...")
    
    # Convert to numpy and ensure [0, 1] range
    recon_np = reconstructed_video.numpy()
    recon_np = np.clip(recon_np, 0, 1)
    
    # Convert to (T, H, W, C) and scale to [0, 255]
    recon_np = recon_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    recon_np = (recon_np * 255).astype(np.uint8)
    
    # Determine output path
    if args.output_path is None:
        output_dir = Path("evaluations/tokenizer")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"video_checkpoint_step_{step}.mp4"
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save video
    save_video(recon_np, output_path, fps=args.fps)
    
    print("\n" + "=" * 70)
    print("Video generation complete!")
    print("=" * 70)
    print(f"Video saved to: {output_path}")
    print(f"  - Frames: {len(recon_np)}")
    print(f"  - Duration: {len(recon_np) / args.fps:.2f} seconds")
    print(f"  - FPS: {args.fps}")


if __name__ == "__main__":
    main()
