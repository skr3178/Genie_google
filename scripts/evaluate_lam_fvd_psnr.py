"""Evaluate LAM model with FVD and PSNR metrics"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import sys
import json
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lam import LAM
from src.utils.config import load_config
from Metrics.PSNR.psnr_metric import psnr
from Metrics.FVD.fvd_metric import FVDMetric


def load_lam_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load LAM model from checkpoint"""
    # Load to CPU first to avoid OOM, then move to device
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = load_config("configs/lam_config.yaml")
    
    model = LAM(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model, config, checkpoint


def load_video_sequence(h5_path: str, start_idx: int, sequence_length: int, resolution: tuple = (128, 72)):
    """Load a sequence of frames from HDF5 file"""
    with h5py.File(h5_path, 'r') as f:
        if 'frames' in f:
            frames = f['frames']
        else:
            first_key = list(f.keys())[0]
            frames = f[first_key]
        
        num_frames = frames.shape[0]
        
        if start_idx + sequence_length > num_frames:
            start_idx = max(0, num_frames - sequence_length)
        
        frame_data = frames[start_idx:start_idx + sequence_length]
        
        if isinstance(frame_data, h5py.Dataset):
            frame_data = frame_data[:]
        
        if frame_data.shape[1:3] != resolution:
            resized_frames = []
            for frame in frame_data:
                img = Image.fromarray(frame)
                img = img.resize((resolution[1], resolution[0]), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frame_data = np.array(resized_frames)
        
        frame_data = frame_data.transpose(0, 3, 1, 2)  # (T, C, H, W)
        frame_data = frame_data.astype(np.float32) / 255.0
        
        return torch.from_numpy(frame_data).float()


def evaluate_lam_psnr(model, config, h5_path: str, num_samples: int = 100, device: str = "cuda"):
    """Evaluate PSNR on test videos"""
    print(f"\nEvaluating PSNR on {num_samples} samples...")
    
    sequence_length = config['data']['sequence_length']
    resolution = tuple(config['data']['resolution'][:2])
    
    # Get total number of frames
    with h5py.File(h5_path, 'r') as f:
        if 'frames' in f:
            num_frames = f['frames'].shape[0]
        else:
            first_key = list(f.keys())[0]
            num_frames = f[first_key].shape[0]
    
    # Sample random start indices
    max_start = num_frames - sequence_length
    if max_start < num_samples:
        num_samples = max_start
        print(f"Warning: Only {num_samples} samples available")
    
    start_indices = np.random.choice(max_start, size=num_samples, replace=False)
    
    all_psnr_values = []
    
    with torch.no_grad():
        for idx in tqdm(start_indices, desc="Computing PSNR"):
            # Load video sequence
            video = load_video_sequence(h5_path, int(idx), sequence_length, resolution)
            video = video.to(device)
            
            # Split into past frames and next frame
            past_frames = video[:-1].unsqueeze(0)  # (1, T-1, C, H, W)
            next_frame = video[-1].unsqueeze(0)  # (1, C, H, W)
            
            # Reconstruct
            reconstructed, _, _ = model(past_frames, next_frame)
            
            # Apply sigmoid to get [0, 1] range
            reconstructed = torch.sigmoid(reconstructed)
            
            # Normalize GT to [0, 1] if needed
            gt_frame = next_frame.squeeze(0)  # (C, H, W)
            if gt_frame.max() > 1.0:
                gt_frame = gt_frame / 255.0
            
            # Compute PSNR
            psnr_val = psnr(gt_frame.unsqueeze(0), reconstructed, max_val=1.0)
            all_psnr_values.append(psnr_val.item())
    
    return {
        'psnr_mean': float(np.mean(all_psnr_values)),
        'psnr_std': float(np.std(all_psnr_values)),
        'psnr_min': float(np.min(all_psnr_values)),
        'psnr_max': float(np.max(all_psnr_values)),
        'num_samples': len(all_psnr_values),
    }


def evaluate_lam_fvd(model, config, h5_path: str, num_videos: int = 50, device: str = "cuda", fvd_model_path: str = None):
    """Evaluate FVD on test videos"""
    print(f"\nEvaluating FVD on {num_videos} videos...")
    
    sequence_length = config['data']['sequence_length']
    resolution = tuple(config['data']['resolution'][:2])
    
    # Get total number of frames
    with h5py.File(h5_path, 'r') as f:
        if 'frames' in f:
            num_frames = f['frames'].shape[0]
        else:
            first_key = list(f.keys())[0]
            num_frames = f[first_key].shape[0]
    
    # Sample random start indices
    max_start = num_frames - sequence_length
    if max_start < num_videos:
        num_videos = max_start
        print(f"Warning: Only {num_videos} videos available")
    
    start_indices = np.random.choice(max_start, size=num_videos, replace=False)
    
    real_videos = []
    generated_videos = []
    
    with torch.no_grad():
        for idx in tqdm(start_indices, desc="Generating videos for FVD"):
            # Load video sequence
            video = load_video_sequence(h5_path, int(idx), sequence_length, resolution)
            video = video.to(device)
            
            # Split into past frames and next frame
            past_frames = video[:-1].unsqueeze(0)  # (1, T-1, C, H, W)
            next_frame = video[-1].unsqueeze(0)  # (1, C, H, W)
            
            # Reconstruct
            reconstructed, _, _ = model(past_frames, next_frame)
            
            # Apply sigmoid to get [0, 1] range
            reconstructed = torch.sigmoid(reconstructed)
            
            # Normalize GT to [0, 1] if needed
            gt_frame = next_frame.squeeze(0)  # (C, H, W)
            if gt_frame.max() > 1.0:
                gt_frame = gt_frame / 255.0
            
            # Store as (T, C, H, W) for FVD
            # Real: past frames + GT next frame
            real_vid = torch.cat([past_frames.squeeze(0), gt_frame.unsqueeze(0)], dim=0)  # (T, C, H, W)
            # Generated: past frames + reconstructed next frame
            # reconstructed is (1, C, H, W), squeeze to (C, H, W) then unsqueeze to (1, C, H, W)
            recon_frame = reconstructed.squeeze(0)  # (C, H, W)
            gen_vid = torch.cat([past_frames.squeeze(0), recon_frame.unsqueeze(0)], dim=0)  # (T, C, H, W)
            
            real_videos.append(real_vid.cpu())
            generated_videos.append(gen_vid.cpu())
    
    # Stack to (N, T, C, H, W)
    real_videos = torch.stack(real_videos)  # (N, T, C, H, W)
    generated_videos = torch.stack(generated_videos)  # (N, T, C, H, W)
    
    # Compute FVD
    print("Computing FVD...")
    fvd_metric = FVDMetric(model_path=fvd_model_path, device=device)
    fvd_value = fvd_metric.compute(real_videos, generated_videos)
    
    return {
        'fvd': float(fvd_value),
        'num_videos': len(real_videos),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LAM with FVD and PSNR")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LAM checkpoint")
    parser.add_argument("--data_path", type=str, default="data/pong_frames.h5", help="Path to HDF5 dataset")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_samples_psnr", type=int, default=100, help="Number of samples for PSNR")
    parser.add_argument("--num_videos_fvd", type=int, default=50, help="Number of videos for FVD")
    parser.add_argument("--eval_psnr", action="store_true", help="Evaluate PSNR")
    parser.add_argument("--eval_fvd", action="store_true", help="Evaluate FVD")
    parser.add_argument("--fvd_model_path", type=str, default=None, help="Path to I3D model for FVD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Default: evaluate both if neither specified
    if not args.eval_psnr and not args.eval_fvd:
        args.eval_psnr = True
        args.eval_fvd = True
    
    print("=" * 70)
    print("LAM FVD and PSNR Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load model
    print("\nLoading LAM model...")
    model, config, checkpoint = load_lam_from_checkpoint(args.checkpoint, args.device)
    print(f"✓ Model loaded")
    print(f"  Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"  Num codes: {config['model']['codebook']['num_codes']}")
    print(f"  Resolution: {config['data']['resolution']}")
    
    results = {}
    
    # Evaluate PSNR
    if args.eval_psnr:
        print("\n" + "=" * 70)
        print("PSNR Evaluation")
        print("=" * 70)
        psnr_results = evaluate_lam_psnr(
            model, config, args.data_path, 
            num_samples=args.num_samples_psnr,
            device=args.device
        )
        results['psnr'] = psnr_results
        print(f"\nPSNR Results:")
        print(f"  Mean: {psnr_results['psnr_mean']:.4f} dB")
        print(f"  Std: {psnr_results['psnr_std']:.4f} dB")
        print(f"  Min: {psnr_results['psnr_min']:.4f} dB")
        print(f"  Max: {psnr_results['psnr_max']:.4f} dB")
        print(f"  Samples: {psnr_results['num_samples']}")
    
    # Evaluate FVD
    if args.eval_fvd:
        print("\n" + "=" * 70)
        print("FVD Evaluation")
        print("=" * 70)
        fvd_results = evaluate_lam_fvd(
            model, config, args.data_path,
            num_videos=args.num_videos_fvd,
            device=args.device,
            fvd_model_path=args.fvd_model_path
        )
        results['fvd'] = fvd_results
        print(f"\nFVD Results:")
        print(f"  FVD: {fvd_results['fvd']:.4f}")
        print(f"  Videos: {fvd_results['num_videos']}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    else:
        # Auto-generate output path
        checkpoint_name = Path(args.checkpoint).stem
        output_dir = Path("evaluations/lam")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{checkpoint_name}_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
