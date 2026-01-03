"""Evaluate Dynamics Model checkpoint

This script evaluates the dynamics model by:
1. Loading video sequences and tokenizing them
2. Using LAM to get actions
3. Using the dynamics model to predict next frame tokens
4. Decoding back to pixels and comparing with ground truth
5. Computing metrics (token accuracy, PSNR, MSE)
6. Creating comparison videos
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import sys
import gc
from PIL import Image
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from src.models.dynamics import DynamicsModel
from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.utils.config import load_config
from Metrics.PSNR.psnr_metric import psnr


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


def add_border(frame: np.ndarray, border_width: int = 3, color: tuple = (255, 0, 0)) -> np.ndarray:
    """Add a colored border around a frame"""
    H, W, C = frame.shape
    bordered = np.zeros((H + 2 * border_width, W + 2 * border_width, C), dtype=frame.dtype)
    # Fill with border color
    bordered[:, :] = color
    # Place original frame in center
    bordered[border_width:border_width + H, border_width:border_width + W] = frame
    return bordered


def create_comparison_video(
    frames_list: list,
    output_path: Path,
    fps: float = 5.0,
    labels: list = None,
    border_width: int = 3,
    border_color: tuple = (255, 0, 0),  # Red
):
    """Create a comparison video showing multiple frame sequences side by side with borders"""
    if not IMAGEIO_AVAILABLE:
        print("  Warning: imageio not available, skipping video creation")
        return False
    
    all_frames = []
    num_sequences = len(frames_list)
    num_frames = len(frames_list[0])
    
    for frame_idx in range(num_frames):
        # Get frames from all sequences at this time step
        row_frames = []
        for seq_idx, seq in enumerate(frames_list):
            frame = seq[frame_idx]
            
            # Convert to numpy if tensor
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # Ensure (H, W, C) format
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            
            # Normalize to [0, 255]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            # Add label if provided
            if labels and seq_idx < len(labels):
                from PIL import Image, ImageDraw, ImageFont
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
                except:
                    font = ImageFont.load_default()
                
                # Draw label with outline
                label = labels[seq_idx]
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((5 + dx, 5 + dy), label, fill=(0, 0, 0), font=font)
                draw.text((5, 5), label, fill=(255, 255, 255), font=font)
                frame = np.array(frame_pil)
            
            # Add red border around this frame
            frame = add_border(frame, border_width=border_width, color=border_color)
            
            row_frames.append(frame)
        
        # Stack frames horizontally with gap
        gap = np.zeros((row_frames[0].shape[0], 5, 3), dtype=np.uint8)
        combined = row_frames[0]
        for frame in row_frames[1:]:
            combined = np.hstack([combined, gap, frame])
        
        all_frames.append(combined)
    
    # Save video
    try:
        imageio.mimwrite(
            str(output_path),
            all_frames,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        print(f"  ✓ Saved comparison video to {output_path}")
        return True
    except Exception as e:
        print(f"  Warning: Error saving video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dynamics Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to dynamics model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dynamics_config_3actions.yaml",
        help="Dynamics model config file"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint"
    )
    parser.add_argument(
        "--tokenizer_config",
        type=str,
        default="configs/tokenizer_config.yaml",
        help="Tokenizer config file"
    )
    parser.add_argument(
        "--lam_path",
        type=str,
        required=True,
        help="Path to LAM checkpoint"
    )
    parser.add_argument(
        "--lam_config",
        type=str,
        default="configs/lam_config_paper.yaml",
        help="LAM config file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pong_frames.h5",
        help="Path to dataset HDF5 file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of video sequences to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluations/dynamics",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--maskgit_steps",
        type=int,
        default=12,
        help="Number of MaskGIT iterative refinement steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting frame index in dataset"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Dynamics Model Evaluation")
    print("=" * 70)
    
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    print("\n1. Loading configurations...")
    dynamics_config = load_config(args.config)
    tokenizer_config = load_config(args.tokenizer_config)
    lam_config = load_config(args.lam_config)
    
    sequence_length = dynamics_config['data']['sequence_length']
    resolution = tuple(lam_config['data']['resolution'][:2])  # (H, W)
    
    print(f"   Sequence length: {sequence_length}")
    print(f"   Resolution: {resolution}")
    print(f"   Num actions: {dynamics_config['model']['action_embedding']['num_actions']}")
    
    # Load tokenizer (to CPU first, then move to GPU)
    print("\n2. Loading tokenizer...")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    
    tokenizer_ckpt = torch.load(args.tokenizer_path, map_location='cpu')
    if 'model_state_dict' in tokenizer_ckpt:
        tokenizer.load_state_dict(tokenizer_ckpt['model_state_dict'])
        print(f"   ✓ Loaded tokenizer from step {tokenizer_ckpt.get('global_step', 'unknown')}")
    else:
        tokenizer.load_state_dict(tokenizer_ckpt)
        print(f"   ✓ Loaded tokenizer (direct state dict)")
    del tokenizer_ckpt
    gc.collect()
    tokenizer = tokenizer.to(device).eval()
    
    # Load LAM (to CPU first, then move to GPU)
    print("\n3. Loading LAM...")
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    )
    
    lam_ckpt = torch.load(args.lam_path, map_location='cpu')
    if 'model_state_dict' in lam_ckpt:
        lam.load_state_dict(lam_ckpt['model_state_dict'])
        print(f"   ✓ Loaded LAM from step {lam_ckpt.get('global_step', 'unknown')}")
    else:
        lam.load_state_dict(lam_ckpt)
        print(f"   ✓ Loaded LAM (direct state dict)")
    del lam_ckpt
    gc.collect()
    lam = lam.to(device).eval()
    
    # Clear memory after loading models
    clear_gpu_memory()
    
    # Load dynamics model
    print("\n4. Loading dynamics model...")
    
    # Ensure vocab_size matches tokenizer
    tokenizer_vocab_size = tokenizer_config['model']['codebook']['num_codes']
    if dynamics_config['model']['token_embedding']['vocab_size'] != tokenizer_vocab_size:
        print(f"   Updating vocab_size from {dynamics_config['model']['token_embedding']['vocab_size']} to {tokenizer_vocab_size}")
        dynamics_config['model']['token_embedding']['vocab_size'] = tokenizer_vocab_size
    
    dynamics_model = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    )
    
    dynamics_ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in dynamics_ckpt:
        dynamics_model.load_state_dict(dynamics_ckpt['model_state_dict'])
        step = dynamics_ckpt.get('global_step', 'unknown')
        print(f"   ✓ Loaded dynamics model from step {step}")
    else:
        dynamics_model.load_state_dict(dynamics_ckpt)
        step = 'unknown'
        print(f"   ✓ Loaded dynamics model (direct state dict)")
    del dynamics_ckpt
    gc.collect()
    dynamics_model = dynamics_model.to(device).eval()
    clear_gpu_memory()
    
    # Print model info
    total_params = sum(p.numel() for p in dynamics_model.parameters())
    print(f"   Dynamics model parameters: {total_params:,}")
    
    # Load dataset info
    print("\n5. Loading dataset...")
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
    print(f"   ✓ Found {total_frames} frames in dataset")
    
    # Evaluate
    print(f"\n6. Evaluating on {args.num_samples} samples...")
    print(f"   MaskGIT steps: {args.maskgit_steps}")
    print(f"   Temperature: {args.temperature}")
    
    all_token_accuracies = []
    all_psnr_values = []
    all_mse_values = []
    
    comparison_frames_gt = []
    comparison_frames_pred = []
    comparison_frames_input = []
    
    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            print(f"\n   Sample {sample_idx + 1}/{args.num_samples}:")
            
            # Calculate start index for this sample
            max_start = max(0, total_frames - sequence_length - 2)
            if args.num_samples > 1:
                start_idx = args.start_idx + (sample_idx * max_start // max(1, args.num_samples - 1))
            else:
                start_idx = args.start_idx
            start_idx = min(start_idx, max_start)
            
            # Load video sequence (need sequence_length + 1 for next frame prediction)
            video = load_video_sequence(
                str(data_path),
                start_idx,
                sequence_length + 1,
                resolution=resolution
            )  # (T+1, C, H, W)
            
            # Split into input frames and target frame
            input_frames = video[:-1]  # (T, C, H, W)
            target_frame = video[-1]   # (C, H, W)
            
            # Add batch dimension
            input_frames_batch = input_frames.unsqueeze(0).to(device)  # (1, T, C, H, W)
            target_frame_batch = target_frame.unsqueeze(0).to(device)  # (1, C, H, W)
            
            # Tokenize all frames
            all_frames_batch = video.unsqueeze(0).to(device)  # (1, T+1, C, H, W)
            all_tokens = tokenizer.encode(all_frames_batch)  # (1, T+1, H_patches, W_patches)
            
            input_tokens = all_tokens[:, :-1]  # (1, T, H_patches, W_patches)
            target_tokens = all_tokens[:, -1]  # (1, H_patches, W_patches)
            
            # Get actions from LAM
            # LAM expects (B, T, C, H, W) for past frames and (B, C, H, W) for next frame
            _, actions, _ = lam(input_frames_batch, target_frame_batch)  # actions: (1, T, H_patches, W_patches)
            
            print(f"     Input tokens shape: {input_tokens.shape}")
            print(f"     Target tokens shape: {target_tokens.shape}")
            print(f"     Actions shape: {actions.shape}")
            print(f"     Unique actions: {len(torch.unique(actions))}")
            
            # Method 1: Direct next-token prediction (teacher forcing)
            # Predict the last frame tokens given all previous tokens and actions
            # For MaskGIT, we mask only the last frame's tokens
            B, T, H_patches, W_patches = input_tokens.shape
            
            # Create input with all tokens (including placeholder for target)
            # We'll use the target tokens but mask them
            full_tokens = all_tokens.clone()  # (1, T+1, H_patches, W_patches)
            
            # Create mask: mask only the last frame
            mask = torch.zeros(1, T + 1, H_patches, W_patches, device=device)
            mask[:, -1, :, :] = 1.0  # Mask only the last frame
            
            # Extend actions to match full sequence (add a dummy action for last frame)
            # The last action is what transitions from T-1 to T
            full_actions = torch.cat([
                actions,
                actions[:, -1:, :, :]  # Repeat last action
            ], dim=1)  # (1, T+1, H_patches, W_patches)
            
            # Forward pass with mask
            logits = dynamics_model(full_tokens, full_actions, mask)  # (1, T+1, H_patches, W_patches, vocab_size)
            
            # Get predictions for masked (last) frame
            last_frame_logits = logits[:, -1]  # (1, H_patches, W_patches, vocab_size)
            predicted_tokens = last_frame_logits.argmax(dim=-1)  # (1, H_patches, W_patches)
            
            # Calculate token accuracy
            token_accuracy = (predicted_tokens == target_tokens).float().mean().item()
            all_token_accuracies.append(token_accuracy)
            print(f"     Token accuracy (teacher forcing): {token_accuracy:.4f}")
            
            # Method 2: MaskGIT iterative refinement (autoregressive generation)
            # Start from masked tokens and iteratively refine
            # Initialize with random tokens for the last frame
            init_tokens = full_tokens.clone()
            init_tokens[:, -1] = torch.randint(0, tokenizer_config['model']['codebook']['num_codes'], 
                                                (1, H_patches, W_patches), device=device)
            
            # Use iterative refinement
            refined_tokens = dynamics_model.iterative_refinement(
                init_tokens,
                full_actions,
                steps=args.maskgit_steps,
                temperature=args.temperature,
            )
            
            # Get the last frame tokens
            refined_last_tokens = refined_tokens[:, -1]  # (1, H_patches, W_patches)
            refined_accuracy = (refined_last_tokens == target_tokens).float().mean().item()
            print(f"     Token accuracy (MaskGIT {args.maskgit_steps} steps): {refined_accuracy:.4f}")
            
            # Decode tokens back to pixels
            # Decode ground truth
            target_frame_decoded = tokenizer.decode(target_tokens.unsqueeze(1))  # (1, 1, C, H, W)
            target_frame_decoded = target_frame_decoded.squeeze(1)  # (1, C, H, W)
            
            # Decode predicted tokens (teacher forcing)
            predicted_frame = tokenizer.decode(predicted_tokens.unsqueeze(1))  # (1, 1, C, H, W)
            predicted_frame = predicted_frame.squeeze(1)  # (1, C, H, W)
            
            # Calculate pixel-level metrics
            # Compare with original target frame (not decoded ground truth, to measure true quality)
            mse = F.mse_loss(predicted_frame, target_frame_batch).item()
            psnr_val = psnr(predicted_frame[0], target_frame_batch[0], max_val=1.0).item()
            
            all_mse_values.append(mse)
            all_psnr_values.append(psnr_val)
            
            print(f"     MSE: {mse:.6f}")
            print(f"     PSNR: {psnr_val:.2f} dB")
            
            # Store frames for visualization (all samples)
            comparison_frames_input.append(input_frames[-1].cpu())  # Last input frame
            comparison_frames_gt.append(target_frame.cpu())
            comparison_frames_pred.append(predicted_frame[0].cpu())
            
            # Clear memory periodically
            if sample_idx % 3 == 0:
                clear_gpu_memory()
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Training step: {step}")
    print(f"Number of samples: {args.num_samples}")
    print(f"\nToken Accuracy (teacher forcing):")
    print(f"  Mean: {np.mean(all_token_accuracies):.4f}")
    print(f"  Std:  {np.std(all_token_accuracies):.4f}")
    print(f"  Min:  {np.min(all_token_accuracies):.4f}")
    print(f"  Max:  {np.max(all_token_accuracies):.4f}")
    print(f"\nPixel-level Metrics:")
    print(f"  PSNR Mean: {np.mean(all_psnr_values):.2f} dB")
    print(f"  PSNR Std:  {np.std(all_psnr_values):.2f} dB")
    print(f"  MSE Mean:  {np.mean(all_mse_values):.6f}")
    print(f"  MSE Std:   {np.std(all_mse_values):.6f}")
    
    # Save metrics to JSON
    metrics = {
        'checkpoint': args.checkpoint,
        'step': step,
        'num_samples': args.num_samples,
        'maskgit_steps': args.maskgit_steps,
        'temperature': args.temperature,
        'token_accuracy': {
            'mean': float(np.mean(all_token_accuracies)),
            'std': float(np.std(all_token_accuracies)),
            'min': float(np.min(all_token_accuracies)),
            'max': float(np.max(all_token_accuracies)),
            'values': [float(v) for v in all_token_accuracies],
        },
        'psnr': {
            'mean': float(np.mean(all_psnr_values)),
            'std': float(np.std(all_psnr_values)),
            'min': float(np.min(all_psnr_values)),
            'max': float(np.max(all_psnr_values)),
            'values': [float(v) for v in all_psnr_values],
        },
        'mse': {
            'mean': float(np.mean(all_mse_values)),
            'std': float(np.std(all_mse_values)),
            'min': float(np.min(all_mse_values)),
            'max': float(np.max(all_mse_values)),
            'values': [float(v) for v in all_mse_values],
        },
    }
    
    metrics_path = output_dir / f"dynamics_metrics_step_{step}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Create comparison video
    if comparison_frames_gt:
        print("\n7. Creating comparison video...")
        video_path = output_dir / f"dynamics_comparison_step_{step}.mp4"
        create_comparison_video(
            [comparison_frames_input, comparison_frames_gt, comparison_frames_pred],
            video_path,
            fps=2.0,
            labels=["Input (t-1)", "Ground Truth (t)", "Predicted (t)"]
        )
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
