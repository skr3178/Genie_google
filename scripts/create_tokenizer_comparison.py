"""Create side-by-side comparison video from tokenizer checkpoint"""

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


def create_side_by_side_video(original_frames: np.ndarray, reconstructed_frames: np.ndarray, 
                               output_path: Path, fps: float = 10.0):
    """Create a side-by-side comparison video"""
    
    if len(original_frames) != len(reconstructed_frames):
        raise ValueError(f"Mismatch in frame counts: {len(original_frames)} vs {len(reconstructed_frames)}")
    
    # Create side-by-side frames
    combined_frames = []
    for orig, recon in zip(original_frames, reconstructed_frames):
        # Ensure same height
        h = max(orig.shape[0], recon.shape[0])
        w_orig = orig.shape[1]
        w_recon = recon.shape[1]
        
        # Resize if needed to match height
        if orig.shape[0] != h:
            orig_img = Image.fromarray(orig)
            orig = np.array(orig_img.resize((w_orig, h), Image.Resampling.LANCZOS))
        if recon.shape[0] != h:
            recon_img = Image.fromarray(recon)
            recon = np.array(recon_img.resize((w_recon, h), Image.Resampling.LANCZOS))
        
        # Combine side by side
        combined = np.hstack([orig, recon])
        combined_frames.append(combined)
    
    # Save video
    video_saved = False
    
    if IMAGEIO_AVAILABLE:
        try:
            # Use imageio with H.264 codec for maximum compatibility
            imageio.mimwrite(
                str(output_path),
                combined_frames,
                fps=fps,
                codec='libx264',  # H.264 codec
                quality=8,  # High quality (0-10 scale, 10 is best)
                pixelformat='yuv420p'  # Ensures compatibility
            )
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"  ✓ Saved video to {output_path} using H.264 (libx264)")
                video_saved = True
            else:
                print(f"  Warning: Video file appears to be too small or missing")
        except Exception as e:
            print(f"  Warning: Error creating video with imageio: {e}")
            if CV2_AVAILABLE:
                print(f"  Falling back to OpenCV...")
    
    if not video_saved and CV2_AVAILABLE:
        # Fallback to OpenCV with H.264 settings
        try:
            H, W = combined_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264/AVC codec
            temp_path = str(output_path).replace('.mp4', '_temp.mp4')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            
            if out.isOpened():
                for frame in combined_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                # Check if file was written successfully
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
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video from tokenizer checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/tokenizer/checkpoint_step_1202.pt",
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
        help="Output video path (default: evaluations/tokenizer/comparison_checkpoint_step_XXXX.mp4)"
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize reconstructed values to full [0, 1] range for better visibility (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization (show raw model output)"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to process and concatenate for longer video (default: 1)"
    )
    parser.add_argument(
        "--sequence_stride",
        type=int,
        default=None,
        help="Stride between sequences (default: sequence_length, i.e., no overlap)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating Tokenizer Comparison Video")
    print("=" * 70)
    
    # Load config
    print("\n1. Loading config...")
    config = load_config(args.config)
    sequence_length = config['data']['sequence_length']
    resolution = tuple(config['data']['resolution'][:2])  # (H, W)
    print(f"   Sequence length: {sequence_length}")
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
    
    # Load checkpoint to CPU first to avoid OOM
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('global_step', 'unknown')
    else:
        tokenizer.load_state_dict(checkpoint)
        step = 'unknown'
    
    # Move model to device after loading
    tokenizer = tokenizer.to(args.device)
    tokenizer.eval()
    print(f"   ✓ Loaded tokenizer from {checkpoint_path}")
    print(f"   Training step: {step}")
    
    # Load dataset
    print("\n3. Loading video sequences...")
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"   ✗ Error: Dataset not found at {data_path}")
        print(f"   Please provide a valid --data_path")
        return
    
    # Get total number of frames in dataset
    with h5py.File(data_path, 'r') as f:
        if 'frames' in f:
            total_frames = f['frames'].shape[0]
        else:
            first_key = list(f.keys())[0]
            total_frames = f[first_key].shape[0]
    
    # Determine stride between sequences
    stride = args.sequence_stride if args.sequence_stride is not None else sequence_length
    
    print(f"   Total frames in dataset: {total_frames}")
    print(f"   Processing {args.num_sequences} sequence(s) with stride {stride}")
    
    # Process multiple sequences
    all_original_frames = []
    all_reconstructed_frames = []
    
    print("\n4. Running tokenizer inference on sequences...")
    with torch.no_grad():
        for seq_idx in range(args.num_sequences):
            start_frame = args.start_idx + seq_idx * stride
            
            # Check if we have enough frames
            if start_frame + sequence_length > total_frames:
                print(f"   ⚠ Warning: Not enough frames for sequence {seq_idx + 1}, stopping early")
                break
            
            print(f"   Processing sequence {seq_idx + 1}/{args.num_sequences} (frames {start_frame} to {start_frame + sequence_length - 1})...")
            
            # Load video sequence
            video = load_video_sequence(
                str(data_path),
                start_frame,
                sequence_length,
                resolution=resolution
            )  # (T, C, H, W)
            
            # Add batch dimension: (1, T, C, H, W)
            video_batch = video.unsqueeze(0).to(args.device)
            
            # Encode and decode
            reconstructed, tokens, vq_loss_dict = tokenizer(video_batch)
            
            # Remove batch dimension
            video_single = video_batch[0].cpu()  # (T, C, H, W)
            reconstructed_single = reconstructed[0].cpu()  # (T, C, H, W)
            
            # Store frames
            all_original_frames.append(video_single)
            all_reconstructed_frames.append(reconstructed_single)
            
            # Clear GPU memory
            del video_batch, reconstructed, tokens, vq_loss_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all sequences
    if not all_original_frames:
        print("   ✗ Error: No sequences were processed")
        return
    
    video_single = torch.cat(all_original_frames, dim=0)  # (total_T, C, H, W)
    reconstructed_single = torch.cat(all_reconstructed_frames, dim=0)  # (total_T, C, H, W)
    
    # Diagnostic output
    print(f"\n   ✓ Generated {len(all_original_frames)} sequence(s), {video_single.shape[0]} total frames")
    print(f"   Reconstructed stats: min={reconstructed_single.min():.6f}, max={reconstructed_single.max():.6f}, mean={reconstructed_single.mean():.6f}")
    print(f"   Original stats: min={video_single.min():.4f}, max={video_single.max():.4f}, mean={video_single.mean():.4f}")
    
    # Check if values are too small (black output)
    if reconstructed_single.max() < 0.01:
        print(f"   ⚠ WARNING: Reconstructed values are very small (max={reconstructed_single.max():.6f})")
        print(f"   This will result in black output. The model may not be properly trained.")
        print(f"   Attempting to normalize/re-scale...")
        
        # Try to re-scale if values are non-zero but very small
        if reconstructed_single.max() > 0:
            # Re-scale to [0, 1] range
            recon_min = reconstructed_single.min()
            recon_max = reconstructed_single.max()
            if recon_max > recon_min:
                reconstructed_single = (reconstructed_single - recon_min) / (recon_max - recon_min)
                print(f"   Re-scaled to: min={reconstructed_single.min():.6f}, max={reconstructed_single.max():.6f}")
            else:
                print(f"   ⚠ All values are the same, cannot re-scale")
        else:
            print(f"   ⚠ All values are zero or negative!")
    
    # Convert to numpy for video creation
    print("\n5. Creating comparison video...")
    
    # Convert to numpy and ensure [0, 1] range
    orig_np = video_single.numpy()
    recon_np = reconstructed_single.numpy()
    
    # Clamp to [0, 1]
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    # Final check before conversion
    print(f"   After clipping - Original: min={orig_np.min():.4f}, max={orig_np.max():.4f}")
    print(f"   After clipping - Reconstructed: min={recon_np.min():.4f}, max={recon_np.max():.4f}")
    
    # Normalize reconstructed to full [0, 1] range for better visibility
    # This helps if the model output is dimmer than expected
    if args.normalize:
        recon_min = recon_np.min()
        recon_max = recon_np.max()
        if recon_max > recon_min and recon_max < 0.8:  # Only normalize if values are compressed
            print(f"   Normalizing reconstructed values to full range...")
            recon_np = (recon_np - recon_min) / (recon_max - recon_min)
            print(f"   After normalization - Reconstructed: min={recon_np.min():.4f}, max={recon_np.max():.4f}")
        else:
            print(f"   Reconstructed values already in good range, skipping normalization")
    else:
        print(f"   Normalization disabled, using raw model output")
    
    # Convert to (T, H, W, C) and scale to [0, 255]
    orig_np = orig_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    recon_np = recon_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    
    orig_np = (orig_np * 255).astype(np.uint8)
    recon_np = (recon_np * 255).astype(np.uint8)
    
    print(f"   After uint8 conversion - Original: min={orig_np.min()}, max={orig_np.max()}")
    print(f"   After uint8 conversion - Reconstructed: min={recon_np.min()}, max={recon_np.max()}")
    
    # Determine output path
    if args.output_path is None:
        output_dir = Path("evaluations/tokenizer")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.num_sequences > 1:
            output_path = output_dir / f"comparison_checkpoint_step_{step}_long_{args.num_sequences}seq.mp4"
        else:
            output_path = output_dir / f"comparison_checkpoint_step_{step}.mp4"
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create side-by-side video
    create_side_by_side_video(orig_np, recon_np, output_path, fps=args.fps)
    
    print("\n" + "=" * 70)
    print("Video creation complete!")
    print("=" * 70)
    print(f"Comparison video saved to: {output_path}")
    print(f"  - Original frames: {len(orig_np)}")
    print(f"  - Reconstructed frames: {len(recon_np)}")
    print(f"  - FPS: {args.fps}")


if __name__ == "__main__":
    main()
