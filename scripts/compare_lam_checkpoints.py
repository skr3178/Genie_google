"""Compare multiple LAM checkpoints using PSNR and FVD metrics"""

import argparse
import torch
import json
from pathlib import Path
import sys
from typing import List, Dict
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation functions directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluate_lam_fvd_psnr",
    project_root / "scripts" / "evaluate_lam_fvd_psnr.py"
)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

load_lam_from_checkpoint = eval_module.load_lam_from_checkpoint
evaluate_lam_psnr = eval_module.evaluate_lam_psnr
evaluate_lam_fvd = eval_module.evaluate_lam_fvd


def compare_checkpoints(
    checkpoint_paths: List[str],
    data_path: str,
    num_samples_psnr: int = 50,
    num_videos_fvd: int = 20,
    device: str = "cuda",
    fvd_model_path: str = None,
    eval_psnr: bool = True,
    eval_fvd: bool = True,
) -> pd.DataFrame:
    """Compare multiple LAM checkpoints"""
    
    results = []
    
    for checkpoint_path in checkpoint_paths:
        print("\n" + "=" * 70)
        print(f"Evaluating: {Path(checkpoint_path).name}")
        print("=" * 70)
        
        # Load model
        model, config, checkpoint = load_lam_from_checkpoint(checkpoint_path, device)
        
        checkpoint_name = Path(checkpoint_path).stem
        global_step = checkpoint.get('global_step', 'unknown')
        num_codes = config['model']['codebook']['num_codes']
        
        result = {
            'checkpoint': checkpoint_name,
            'path': checkpoint_path,
            'global_step': global_step,
            'num_codes': num_codes,
        }
        
        # Evaluate PSNR
        if eval_psnr:
            try:
                print(f"\nEvaluating PSNR on {num_samples_psnr} samples...")
                psnr_results = evaluate_lam_psnr(
                    model, config, data_path,
                    num_samples=num_samples_psnr,
                    device=device
                )
                result.update({
                    'psnr_mean': psnr_results['psnr_mean'],
                    'psnr_std': psnr_results['psnr_std'],
                    'psnr_min': psnr_results['psnr_min'],
                    'psnr_max': psnr_results['psnr_max'],
                })
                print(f"✓ PSNR: {psnr_results['psnr_mean']:.4f} ± {psnr_results['psnr_std']:.4f} dB")
            except Exception as e:
                print(f"✗ PSNR evaluation failed: {e}")
                result.update({
                    'psnr_mean': None,
                    'psnr_std': None,
                    'psnr_min': None,
                    'psnr_max': None,
                })
        
        # Evaluate FVD
        if eval_fvd:
            try:
                print(f"\nEvaluating FVD on {num_videos_fvd} videos...")
                fvd_results = evaluate_lam_fvd(
                    model, config, data_path,
                    num_videos=num_videos_fvd,
                    device=device,
                    fvd_model_path=fvd_model_path
                )
                result['fvd'] = fvd_results['fvd']
                print(f"✓ FVD: {fvd_results['fvd']:.4f}")
            except Exception as e:
                print(f"✗ FVD evaluation failed: {e}")
                result['fvd'] = None
        
        results.append(result)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare multiple LAM checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                       help="Paths to LAM checkpoints to compare")
    parser.add_argument("--data_path", type=str, default="data/pong_frames.h5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--output", type=str, default="evaluations/lam/comparison_results.csv",
                       help="Output CSV file")
    parser.add_argument("--output_json", type=str, default="evaluations/lam/comparison_results.json",
                       help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_samples_psnr", type=int, default=50,
                       help="Number of samples for PSNR")
    parser.add_argument("--num_videos_fvd", type=int, default=20,
                       help="Number of videos for FVD")
    parser.add_argument("--eval_psnr", action="store_true", help="Evaluate PSNR")
    parser.add_argument("--eval_fvd", action="store_true", help="Evaluate FVD")
    parser.add_argument("--fvd_model_path", type=str, default=None,
                       help="Path to I3D model for FVD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default: evaluate both if neither specified
    if not args.eval_psnr and not args.eval_fvd:
        args.eval_psnr = True
        args.eval_fvd = True
    
    # Set random seed
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("LAM Checkpoint Comparison")
    print("=" * 70)
    print(f"Checkpoints to compare: {len(args.checkpoints)}")
    for i, cp in enumerate(args.checkpoints, 1):
        print(f"  {i}. {cp}")
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Compare checkpoints
    df = compare_checkpoints(
        args.checkpoints,
        args.data_path,
        num_samples_psnr=args.num_samples_psnr,
        num_videos_fvd=args.num_videos_fvd,
        device=args.device,
        fvd_model_path=args.fvd_model_path,
        eval_psnr=args.eval_psnr,
        eval_fvd=args.eval_fvd,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Also save as JSON
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)
    print(f"✓ Results saved to {output_json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    
    if args.eval_psnr:
        print("\n" + "-" * 70)
        print("PSNR Rankings (higher is better):")
        print("-" * 70)
        psnr_sorted = df.sort_values('psnr_mean', ascending=False, na_position='last')
        for i, row in psnr_sorted.iterrows():
            if pd.notna(row['psnr_mean']):
                print(f"  {row['checkpoint']}: {row['psnr_mean']:.4f} dB (step {row['global_step']})")
    
    if args.eval_fvd:
        print("\n" + "-" * 70)
        print("FVD Rankings (lower is better):")
        print("-" * 70)
        fvd_sorted = df.sort_values('fvd', ascending=True, na_position='last')
        for i, row in fvd_sorted.iterrows():
            if pd.notna(row['fvd']):
                print(f"  {row['checkpoint']}: {row['fvd']:.4f} (step {row['global_step']})")
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
