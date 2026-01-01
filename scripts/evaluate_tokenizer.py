"""Evaluate tokenizer by encoding/decoding videos and comparing with originals"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.video_tokenizer import VideoTokenizer
from src.utils.config import load_config
from Metrics.PSNR.psnr_metric import psnr, psnr_per_frame


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


def create_comparison_grid(original: torch.Tensor, reconstructed: torch.Tensor, save_path: str):
    """
    Create a side-by-side comparison grid of original and reconstructed videos.
    
    Args:
        original: Original video tensor (T, C, H, W) in [0, 1]
        reconstructed: Reconstructed video tensor (T, C, H, W) in [0, 1]
        save_path: Path to save the comparison image
    """
    T = original.shape[0]
    
    # Convert to numpy and ensure [0, 1] range
    orig_np = original.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # Clamp to [0, 1]
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    # Convert to (T, H, W, C) for visualization
    orig_np = orig_np.transpose(0, 2, 3, 1)
    recon_np = recon_np.transpose(0, 2, 3, 1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 2 * T))
    gs = gridspec.GridSpec(T, 2, figure=fig, wspace=0.1, hspace=0.1)
    
    for t in range(T):
        # Original frame
        ax_orig = fig.add_subplot(gs[t, 0])
        ax_orig.imshow(orig_np[t])
        ax_orig.set_title(f'Original Frame {t+1}', fontsize=10)
        ax_orig.axis('off')
        
        # Reconstructed frame
        ax_recon = fig.add_subplot(gs[t, 1])
        ax_recon.imshow(recon_np[t])
        ax_recon.set_title(f'Reconstructed Frame {t+1}', fontsize=10)
        ax_recon.axis('off')
    
    plt.suptitle('Original vs Reconstructed Video Comparison', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved comparison grid to {save_path}")


def save_video_frames(video: torch.Tensor, output_dir: str, prefix: str = "frame"):
    """
    Save individual frames from a video.
    
    Args:
        video: Video tensor (T, C, H, W) in [0, 1]
        output_dir: Directory to save frames
        prefix: Prefix for frame filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_np = video.cpu().numpy()
    video_np = np.clip(video_np, 0, 1)
    video_np = (video_np * 255).astype(np.uint8)
    video_np = video_np.transpose(0, 2, 3, 1)  # (T, H, W, C)
    
    for t in range(video_np.shape[0]):
        frame_path = output_dir / f"{prefix}_{t:03d}.png"
        Image.fromarray(video_np[t]).save(frame_path)
    
    print(f"  ✓ Saved {video_np.shape[0]} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer on pong dataset")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/tokenizer/checkpoint_step_1707.pt",
        help="Path to tokenizer checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pong_frames.h5",
        help="Path to pong dataset HDF5 file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizer_config.yaml",
        help="Tokenizer config file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of video sequences to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluations/tokenizer",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting frame index in dataset (will sample sequences starting from here)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Tokenizer Evaluation")
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
    checkpoint_path = Path(args.tokenizer_path)
    if not checkpoint_path.exists():
        print(f"   ✗ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('global_step', 'unknown')
    else:
        tokenizer.load_state_dict(checkpoint)
        step = 'unknown'
    
    tokenizer.eval()
    print(f"   ✓ Loaded tokenizer from {checkpoint_path}")
    print(f"   Training step: {step}")
    
    # Load dataset
    print("\n3. Loading dataset...")
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"   ✗ Error: Dataset not found at {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        if 'frames' in f:
            num_frames = f['frames'].shape[0]
        else:
            first_key = list(f.keys())[0]
            num_frames = f[first_key].shape[0]
    
    print(f"   ✓ Found {num_frames} frames in dataset")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate samples
    print(f"\n4. Evaluating {args.num_samples} video sequences...")
    all_psnr_values = []
    all_mse_values = []
    
    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            print(f"\n   Sample {sample_idx + 1}/{args.num_samples}:")
            
            # Calculate start index for this sample
            # Space them out across the dataset
            max_start = max(0, num_frames - sequence_length - 1)
            if args.num_samples > 1:
                start_idx = args.start_idx + (sample_idx * max_start // max(1, args.num_samples - 1))
            else:
                start_idx = args.start_idx
            start_idx = min(start_idx, max_start)
            
            print(f"     Loading frames {start_idx} to {start_idx + sequence_length}...")
            
            # Load video sequence
            video = load_video_sequence(
                str(data_path),
                start_idx,
                sequence_length,
                resolution=resolution
            )  # (T, C, H, W)
            
            # Add batch dimension: (1, T, C, H, W)
            video_batch = video.unsqueeze(0).to(args.device)
            
            # Encode and decode
            print("     Encoding and decoding...")
            reconstructed, tokens, vq_loss_dict = tokenizer(video_batch)
            
            # Remove batch dimension for metrics
            video_single = video_batch[0]  # (T, C, H, W)
            reconstructed_single = reconstructed[0]  # (T, C, H, W)
            
            # Compute metrics
            psnr_val = psnr(video_single, reconstructed_single, max_val=1.0)
            mse_val = torch.mean((video_single - reconstructed_single) ** 2).item()
            
            all_psnr_values.append(psnr_val.item())
            all_mse_values.append(mse_val)
            
            print(f"     PSNR: {psnr_val.item():.2f} dB")
            print(f"     MSE: {mse_val:.6f}")
            
            # Per-frame PSNR
            psnr_per_frame_vals = psnr_per_frame(video_single, reconstructed_single, max_val=1.0)
            print(f"     Per-frame PSNR: {psnr_per_frame_vals.cpu().numpy()}")
            
            # VQ metrics
            if vq_loss_dict:
                print(f"     VQ Loss: {vq_loss_dict.get('loss', 'N/A')}")
                print(f"     Commitment Loss: {vq_loss_dict.get('commitment_loss', 'N/A')}")
                print(f"     Codebook Loss: {vq_loss_dict.get('codebook_loss', 'N/A')}")
            
            # Save comparison for first few samples
            if sample_idx < 3:
                comparison_path = output_dir / f"comparison_sample_{sample_idx + 1}.png"
                create_comparison_grid(video_single.cpu(), reconstructed_single.cpu(), str(comparison_path))
                
                # Save individual frames
                orig_dir = output_dir / f"sample_{sample_idx + 1}_original"
                recon_dir = output_dir / f"sample_{sample_idx + 1}_reconstructed"
                save_video_frames(video_single.cpu(), str(orig_dir), "original")
                save_video_frames(reconstructed_single.cpu(), str(recon_dir), "reconstructed")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Number of samples: {args.num_samples}")
    print(f"Average PSNR: {np.mean(all_psnr_values):.2f} dB")
    print(f"Std PSNR: {np.std(all_psnr_values):.2f} dB")
    print(f"Min PSNR: {np.min(all_psnr_values):.2f} dB")
    print(f"Max PSNR: {np.max(all_psnr_values):.2f} dB")
    print(f"\nAverage MSE: {np.mean(all_mse_values):.6f}")
    print(f"Std MSE: {np.std(all_mse_values):.6f}")
    
    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Tokenizer Evaluation Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Tokenizer checkpoint: {args.tokenizer_path}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Training step: {step}\n")
        f.write(f"\nNumber of samples: {args.num_samples}\n")
        f.write(f"Average PSNR: {np.mean(all_psnr_values):.2f} dB\n")
        f.write(f"Std PSNR: {np.std(all_psnr_values):.2f} dB\n")
        f.write(f"Min PSNR: {np.min(all_psnr_values):.2f} dB\n")
        f.write(f"Max PSNR: {np.max(all_psnr_values):.2f} dB\n")
        f.write(f"\nAverage MSE: {np.mean(all_mse_values):.6f}\n")
        f.write(f"Std MSE: {np.std(all_mse_values):.6f}\n")
        f.write(f"\nPer-sample PSNR:\n")
        for i, psnr_val in enumerate(all_psnr_values):
            f.write(f"  Sample {i+1}: {psnr_val:.2f} dB\n")
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")
    print(f"  - Comparison grids: comparison_sample_*.png")
    print(f"  - Individual frames: sample_*_original/ and sample_*_reconstructed/")
    print(f"  - Summary: summary.txt")


if __name__ == "__main__":
    main()
