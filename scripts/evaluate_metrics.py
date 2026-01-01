"""Evaluation script for Genie models using FVD and Δ𝑡PSNR metrics"""

import torch
import argparse
from pathlib import Path
import sys
from typing import Dict, List
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.models.dynamics import DynamicsModel
from src.utils.config import load_config
from Metrics.PSNR.delta_psnr import DeltaPSNR
from Metrics.PSNR.psnr_metric import psnr
from Metrics.FVD.fvd_metric import FVDMetric


def load_models(
    tokenizer_path: str,
    lam_path: str,
    dynamics_path: str,
    tokenizer_config_path: str,
    lam_config_path: str,
    dynamics_config_path: str,
    device: str = "cuda",
):
    """Load all Genie models"""
    # Load configs
    tokenizer_config = load_config(tokenizer_config_path)
    lam_config = load_config(lam_config_path)
    dynamics_config = load_config(dynamics_config_path)
    
    # Initialize models
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
    
    # Load weights
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer_checkpoint = torch.load(tokenizer_path, map_location=device)
    if 'model_state_dict' in tokenizer_checkpoint:
        tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    else:
        tokenizer.load_state_dict(tokenizer_checkpoint)
    
    print(f"Loading LAM from {lam_path}...")
    lam_checkpoint = torch.load(lam_path, map_location=device)
    if 'model_state_dict' in lam_checkpoint:
        lam.load_state_dict(lam_checkpoint['model_state_dict'])
    else:
        lam.load_state_dict(lam_checkpoint)
    
    print(f"Loading dynamics model from {dynamics_path}...")
    dynamics_checkpoint = torch.load(dynamics_path, map_location=device)
    if 'model_state_dict' in dynamics_checkpoint:
        dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
    else:
        dynamics_model.load_state_dict(dynamics_checkpoint)
    
    return tokenizer, lam, dynamics_model


def evaluate_delta_psnr(
    tokenizer,
    lam,
    dynamics_model,
    test_videos: torch.Tensor,
    device: str = "cuda",
    t: int = 4,
    num_samples: int = 100,
) -> Dict:
    """
    Evaluate Δ𝑡PSNR metric on test videos.
    
    Args:
        tokenizer: Trained video tokenizer
        lam: Trained LAM
        dynamics_model: Trained dynamics model
        test_videos: Test videos of shape (N, T, C, H, W)
        device: Device to run on
        t: Time step to evaluate (default: 4)
        num_samples: Number of videos to evaluate
    
    Returns:
        Dictionary with metrics
    """
    delta_psnr_metric = DeltaPSNR(
        tokenizer=tokenizer,
        lam=lam,
        dynamics_model=dynamics_model,
        device=device,
        t=t,
    )
    
    num_videos = min(num_samples, test_videos.shape[0])
    delta_psnr_values = []
    psnr_inferred_values = []
    psnr_random_values = []
    
    print(f"Evaluating Δ𝑡PSNR on {num_videos} videos...")
    for i in range(num_videos):
        video = test_videos[i]  # (T, C, H, W)
        
        try:
            delta_psnr, metrics = delta_psnr_metric.compute(video)
            delta_psnr_values.append(delta_psnr)
            psnr_inferred_values.append(metrics['psnr_inferred'])
            psnr_random_values.append(metrics['psnr_random'])
        except Exception as e:
            print(f"Error processing video {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_videos} videos...")
    
    results = {
        'delta_psnr_mean': float(np.mean(delta_psnr_values)) if delta_psnr_values else None,
        'delta_psnr_std': float(np.std(delta_psnr_values)) if delta_psnr_values else None,
        'psnr_inferred_mean': float(np.mean(psnr_inferred_values)) if psnr_inferred_values else None,
        'psnr_random_mean': float(np.mean(psnr_random_values)) if psnr_random_values else None,
        'num_samples': len(delta_psnr_values),
        't': t,
    }
    
    return results


def evaluate_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    device: str = "cuda",
    model_path: str = None,
) -> float:
    """
    Evaluate FVD metric.
    
    Args:
        real_videos: Real videos of shape (N, T, C, H, W) or (N, T, H, W, C)
        generated_videos: Generated videos of same shape
        device: Device to run on
        model_path: Path to I3D model weights
    
    Returns:
        FVD value
    """
    fvd_metric = FVDMetric(model_path=model_path, device=device)
    fvd_value = fvd_metric.compute(real_videos, generated_videos)
    return fvd_value


def main():
    parser = argparse.ArgumentParser(description="Evaluate Genie models with FVD and Δ𝑡PSNR")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--lam_path", type=str, required=True, help="Path to LAM checkpoint")
    parser.add_argument("--dynamics_path", type=str, required=True, help="Path to dynamics model checkpoint")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="Path to tokenizer config")
    parser.add_argument("--lam_config", type=str, required=True, help="Path to LAM config")
    parser.add_argument("--dynamics_config", type=str, required=True, help="Path to dynamics config")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test video data (torch tensor)")
    parser.add_argument("--output", type=str, default="metrics_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--t", type=int, default=4, help="Time step for Δ𝑡PSNR (default: 4)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for evaluation")
    parser.add_argument("--fvd_model_path", type=str, default=None, help="Path to I3D model for FVD")
    parser.add_argument("--eval_delta_psnr", action="store_true", help="Evaluate Δ𝑡PSNR metric")
    parser.add_argument("--eval_fvd", action="store_true", help="Evaluate FVD metric (requires generated videos)")
    parser.add_argument("--generated_videos", type=str, default=None, help="Path to generated videos for FVD")
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    tokenizer, lam, dynamics_model = load_models(
        args.tokenizer_path,
        args.lam_path,
        args.dynamics_path,
        args.tokenizer_config,
        args.lam_config,
        args.dynamics_config,
        args.device,
    )
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = torch.load(args.test_data)
    if isinstance(test_data, dict):
        # Assume videos are in a 'videos' key
        test_videos = test_data['videos']
    else:
        test_videos = test_data
    
    # Ensure videos are in (N, T, C, H, W) format
    if test_videos.dim() == 4:
        # (T, C, H, W) - single video
        test_videos = test_videos.unsqueeze(0)
    elif test_videos.dim() == 5:
        # Already in (N, T, C, H, W) format
        pass
    else:
        raise ValueError(f"Unexpected video tensor shape: {test_videos.shape}")
    
    results = {}
    
    # Evaluate Δ𝑡PSNR
    if args.eval_delta_psnr:
        print("\n" + "="*60)
        print("Evaluating Δ𝑡PSNR metric...")
        print("="*60)
        delta_psnr_results = evaluate_delta_psnr(
            tokenizer,
            lam,
            dynamics_model,
            test_videos,
            device=args.device,
            t=args.t,
            num_samples=args.num_samples,
        )
        results['delta_psnr'] = delta_psnr_results
        print(f"\nΔ𝑡PSNR Results:")
        print(f"  Mean: {delta_psnr_results['delta_psnr_mean']:.4f}")
        print(f"  Std: {delta_psnr_results['delta_psnr_std']:.4f}")
        print(f"  PSNR (inferred): {delta_psnr_results['psnr_inferred_mean']:.4f}")
        print(f"  PSNR (random): {delta_psnr_results['psnr_random_mean']:.4f}")
    
    # Evaluate FVD
    if args.eval_fvd:
        if args.generated_videos is None:
            print("Warning: --generated_videos not provided, skipping FVD evaluation")
        else:
            print("\n" + "="*60)
            print("Evaluating FVD metric...")
            print("="*60)
            generated_videos = torch.load(args.generated_videos)
            if isinstance(generated_videos, dict):
                generated_videos = generated_videos['videos']
            
            # Ensure same number of videos
            num_videos = min(test_videos.shape[0], generated_videos.shape[0])
            fvd_value = evaluate_fvd(
                test_videos[:num_videos],
                generated_videos[:num_videos],
                device=args.device,
                model_path=args.fvd_model_path,
            )
            results['fvd'] = float(fvd_value)
            print(f"\nFVD: {fvd_value:.4f}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    import numpy as np
    main()
