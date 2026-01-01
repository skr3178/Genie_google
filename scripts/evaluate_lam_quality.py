"""Comprehensive LAM quality evaluation script

This script evaluates whether the LAM has learned well by analyzing:
1. Codebook utilization and perplexity
2. Action distribution analysis
3. Reconstruction quality (if data available)
4. Training convergence indicators
5. Action-conditioned reconstruction consistency
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lam import LAM
from src.utils.config import load_config


def load_lam_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load LAM model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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


def analyze_codebook(model, config, device: str = "cuda"):
    """Analyze codebook properties"""
    print("\n" + "=" * 70)
    print("CODEBOOK ANALYSIS")
    print("=" * 70)
    
    # Get codebook embeddings (stored as a buffer, not nn.Embedding)
    codebook = model.quantizer.codebook.data.cpu().numpy()
    num_codes = config['model']['codebook']['num_codes']
    latent_dim = config['model']['codebook']['latent_dim']
    
    print(f"\nCodebook shape: {codebook.shape}")
    print(f"Number of codes: {num_codes}")
    print(f"Latent dimension: {latent_dim}")
    
    # Compute pairwise distances between codes
    distances = np.zeros((num_codes, num_codes))
    for i in range(num_codes):
        for j in range(num_codes):
            distances[i, j] = np.linalg.norm(codebook[i] - codebook[j])
    
    # Distance statistics (excluding diagonal)
    mask = ~np.eye(num_codes, dtype=bool)
    dist_flat = distances[mask]
    
    print(f"\nCodebook distance statistics:")
    print(f"  Min distance: {dist_flat.min():.4f}")
    print(f"  Max distance: {dist_flat.max():.4f}")
    print(f"  Mean distance: {dist_flat.mean():.4f}")
    print(f"  Std distance: {dist_flat.std():.4f}")
    
    # Check for collapsed codes (too close together)
    collapse_threshold = 0.1
    close_pairs = np.sum(dist_flat < collapse_threshold) // 2
    if close_pairs > 0:
        print(f"\n⚠️  WARNING: {close_pairs} code pairs are very close (< {collapse_threshold})")
        print("   This may indicate codebook collapse")
    else:
        print(f"\n✓ No code pairs too close together (good separation)")
    
    # Code norms
    norms = np.linalg.norm(codebook, axis=1)
    print(f"\nCode embedding norms:")
    print(f"  Min: {norms.min():.4f}")
    print(f"  Max: {norms.max():.4f}")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std: {norms.std():.4f}")
    
    return distances


def analyze_action_distribution(model, config, device: str = "cuda", num_samples: int = 50):
    """Analyze action distribution over random inputs"""
    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    batch_size = 4
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    num_codes = config['model']['codebook']['num_codes']
    
    all_actions = []
    all_perplexities = []
    
    print(f"\nSampling {num_samples} batches...")
    
    with torch.no_grad():
        for i in range(num_samples):
            past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
            next_frame = torch.randn(batch_size, C, H, W, device=device)
            
            _, actions, vq_loss_dict = model(past_frames, next_frame)
            all_actions.append(actions.cpu().numpy().flatten())
            all_perplexities.append(vq_loss_dict['perplexity'].item())
    
    # Aggregate statistics
    all_actions = np.concatenate(all_actions)
    avg_perplexity = np.mean(all_perplexities)
    
    # Count action usage
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    print(f"\nAction distribution (random inputs):")
    print(f"  Total actions sampled: {total_actions:,}")
    print(f"  Unique actions used: {len(action_counts)}/{num_codes}")
    
    # Expected uniform distribution
    expected_uniform = total_actions / num_codes
    
    print(f"\n  Action usage (expected ~{expected_uniform:.0f} each for uniform):")
    for action in range(num_codes):
        count = action_counts.get(action, 0)
        pct = 100 * count / total_actions
        bar = "█" * int(pct / 2)
        deviation = abs(count - expected_uniform) / expected_uniform * 100
        status = "✓" if deviation < 50 else "⚠️"
        print(f"    Action {action}: {count:6d} ({pct:5.1f}%) {bar} {status}")
    
    # Perplexity analysis
    print(f"\n  Average perplexity: {avg_perplexity:.2f} / {num_codes}")
    perplexity_pct = avg_perplexity / num_codes * 100
    print(f"  Perplexity percentage: {perplexity_pct:.1f}%")
    
    if perplexity_pct >= 80:
        print(f"  ✓ Excellent codebook usage (>80%)")
    elif perplexity_pct >= 60:
        print(f"  ✓ Good codebook usage (>60%)")
    elif perplexity_pct >= 40:
        print(f"  ⚠️ Moderate codebook usage - consider more training")
    else:
        print(f"  ❌ Poor codebook usage - significant codebook collapse")
    
    # Chi-squared test for uniformity
    observed = np.array([action_counts.get(i, 0) for i in range(num_codes)])
    expected = np.full(num_codes, expected_uniform)
    chi_sq = np.sum((observed - expected) ** 2 / expected)
    
    print(f"\n  Chi-squared statistic: {chi_sq:.2f}")
    print(f"  (Lower is better, 0 = perfect uniform distribution)")
    
    return action_counts, avg_perplexity


def analyze_reconstruction_consistency(model, config, device: str = "cuda"):
    """Test if same action leads to consistent reconstructions"""
    print("\n" + "=" * 70)
    print("RECONSTRUCTION CONSISTENCY ANALYSIS")
    print("=" * 70)
    
    batch_size = 1
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    
    # Fix random seed for reproducible input
    torch.manual_seed(42)
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    
    # Test with multiple different "next frames"
    reconstructions = []
    actions_list = []
    
    print(f"\nTesting reconstruction consistency with same past frames...")
    
    with torch.no_grad():
        for i in range(5):
            torch.manual_seed(100 + i)
            next_frame = torch.randn(batch_size, C, H, W, device=device)
            
            reconstructed, actions, _ = model(past_frames, next_frame)
            reconstructions.append(reconstructed.cpu())
            actions_list.append(actions.cpu().numpy())
    
    # Compare actions across different next frames
    print(f"\n  Actions for different next frames:")
    for i, actions in enumerate(actions_list):
        unique = len(np.unique(actions))
        print(f"    Next frame {i+1}: {unique} unique actions, sample: {actions.flatten()[:10]}...")
    
    # Check if same next frame gives same reconstruction
    print(f"\n  Determinism test (same input → same output):")
    torch.manual_seed(42)
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    torch.manual_seed(100)
    next_frame = torch.randn(batch_size, C, H, W, device=device)
    
    with torch.no_grad():
        recon1, act1, _ = model(past_frames, next_frame)
        recon2, act2, _ = model(past_frames, next_frame)
    
    recon_match = torch.allclose(recon1, recon2, atol=1e-5)
    act_match = (act1 == act2).all().item()
    
    print(f"    Actions match: {'✓' if act_match else '❌'}")
    print(f"    Reconstructions match: {'✓' if recon_match else '❌'}")


def analyze_action_sensitivity(model, config, device: str = "cuda"):
    """Test if different next frames produce different actions"""
    print("\n" + "=" * 70)
    print("ACTION SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    batch_size = 1
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    
    print(f"\nTesting if LAM produces different actions for different transitions...")
    
    # Fixed past frames
    torch.manual_seed(42)
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    
    # Generate many different next frames
    all_actions = []
    with torch.no_grad():
        for i in range(100):
            torch.manual_seed(1000 + i)
            next_frame = torch.randn(batch_size, C, H, W, device=device)
            _, actions, _ = model(past_frames, next_frame)
            all_actions.append(actions.cpu().numpy().flatten())
    
    all_actions = np.array(all_actions)
    
    # Check variance in actions
    unique_action_sets = len(set([tuple(a) for a in all_actions]))
    
    print(f"\n  Unique action patterns: {unique_action_sets}/100")
    
    if unique_action_sets >= 80:
        print(f"  ✓ High action sensitivity - model discriminates well between inputs")
    elif unique_action_sets >= 50:
        print(f"  ✓ Moderate action sensitivity - reasonable discrimination")
    elif unique_action_sets >= 20:
        print(f"  ⚠️ Low action sensitivity - consider more training")
    else:
        print(f"  ❌ Very low action sensitivity - model may not be learning meaningful actions")
    
    # Per-position variance
    action_variance = np.var(all_actions, axis=0)
    print(f"\n  Action variance per position (first 10): {action_variance[:10].round(2)}")
    print(f"  Mean variance: {action_variance.mean():.4f}")
    print(f"  (Higher variance = more diverse action usage)")


def analyze_training_progress(checkpoint):
    """Analyze training progress from checkpoint"""
    print("\n" + "=" * 70)
    print("TRAINING PROGRESS ANALYSIS")
    print("=" * 70)
    
    global_step = checkpoint.get('global_step', 'Unknown')
    epoch = checkpoint.get('epoch', 'Unknown')
    config = checkpoint.get('config', {})
    
    print(f"\n  Training step: {global_step}")
    print(f"  Epoch: {epoch}")
    
    max_steps = config.get('training', {}).get('max_steps', 20000)
    progress = global_step / max_steps * 100 if isinstance(global_step, int) else 0
    
    print(f"  Progress: {progress:.1f}% ({global_step}/{max_steps})")
    
    # Recommendations based on progress
    print(f"\n  Training recommendations:")
    if progress < 25:
        print(f"  ⚠️ Training is at early stage ({progress:.0f}%)")
        print(f"     → Consider training longer for better results")
        print(f"     → At least 50% of max_steps is recommended")
    elif progress < 50:
        print(f"  ⚠️ Training is still early ({progress:.0f}%)")
        print(f"     → Model may improve with more training")
    elif progress < 75:
        print(f"  ✓ Training is at moderate stage ({progress:.0f}%)")
        print(f"     → Results should be reasonable, more training may help")
    else:
        print(f"  ✓ Training is well advanced ({progress:.0f}%)")
        print(f"     → Model should have learned well")
    
    return global_step, max_steps


def generate_quality_report(
    model, config, checkpoint, device: str = "cuda", output_dir: str = "evaluations/lam"
):
    """Generate a comprehensive quality report"""
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT SUMMARY")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all metrics
    metrics = {}
    
    # Training progress
    global_step, max_steps = analyze_training_progress(checkpoint)
    metrics['training_progress'] = global_step / max_steps if max_steps > 0 else 0
    
    # Codebook analysis
    distances = analyze_codebook(model, config, device)
    
    # Action distribution
    action_counts, avg_perplexity = analyze_action_distribution(model, config, device)
    num_codes = config['model']['codebook']['num_codes']
    metrics['perplexity'] = avg_perplexity
    metrics['perplexity_ratio'] = avg_perplexity / num_codes
    
    # Reconstruction consistency
    analyze_reconstruction_consistency(model, config, device)
    
    # Action sensitivity
    analyze_action_sensitivity(model, config, device)
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL QUALITY ASSESSMENT")
    print("=" * 70)
    
    issues = []
    positives = []
    
    # Check perplexity
    if metrics['perplexity_ratio'] >= 0.7:
        positives.append("Excellent codebook utilization")
    elif metrics['perplexity_ratio'] >= 0.5:
        positives.append("Good codebook utilization")
    else:
        issues.append(f"Low codebook utilization ({metrics['perplexity_ratio']*100:.0f}%)")
    
    # Check training progress
    if metrics['training_progress'] >= 0.5:
        positives.append(f"Adequate training progress ({metrics['training_progress']*100:.0f}%)")
    else:
        issues.append(f"Limited training progress ({metrics['training_progress']*100:.0f}%)")
    
    print("\n✓ POSITIVES:")
    for p in positives:
        print(f"  • {p}")
    
    if issues:
        print("\n⚠️ POTENTIAL ISSUES:")
        for i in issues:
            print(f"  • {i}")
    
    # Final recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION:")
    print("-" * 70)
    
    if len(issues) == 0 and metrics['training_progress'] >= 0.5:
        print("\n✓ LAM appears to be well-trained!")
        print("  You can proceed to train the Dynamics Model using this checkpoint.")
        print(f"\n  Command to train dynamics:")
        print(f"  conda run -n robot_wm python scripts/train_dynamics.py \\")
        print(f"      --lam_path {checkpoint_path} \\")
        print(f"      --tokenizer_path checkpoints/tokenizer/checkpoint.pth \\")
        print(f"      --data_dir data --dataset <your_dataset>")
    elif metrics['training_progress'] < 0.25:
        print("\n⚠️ LAM needs more training!")
        print(f"  Current progress: {metrics['training_progress']*100:.0f}%")
        print(f"  Recommendation: Train for at least {int(max_steps * 0.5)} steps (50% of max)")
        print(f"\n  To resume training, you may need to implement checkpoint resumption")
        print(f"  or start fresh with more steps:")
        print(f"  conda run -n robot_wm python scripts/train_lam.py --max_steps {int(max_steps * 0.5)}")
    else:
        print("\n⚠️ LAM may benefit from additional training")
        print(f"  Current step: {global_step}")
        print(f"  Consider training to at least: {int(max_steps * 0.75)} steps")
        print(f"\n  However, you can try using this checkpoint and evaluate downstream performance")
    
    # Save metrics to file
    import json
    metrics_file = output_path / "lam_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'global_step': int(global_step) if isinstance(global_step, (int, np.integer)) else global_step,
            'max_steps': max_steps,
            'perplexity': float(avg_perplexity),
            'num_codes': num_codes,
            'perplexity_ratio': float(metrics['perplexity_ratio']),
            'training_progress': float(metrics['training_progress']),
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LAM quality")
    parser.add_argument("--checkpoint", type=str,
                       default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/lam/checkpoint_step_5000.pt",
                       help="Path to LAM checkpoint")
    parser.add_argument("--config", type=str, default="configs/lam_config.yaml",
                       help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="evaluations/lam",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of sample batches for distribution analysis")
    args = parser.parse_args()
    
    global checkpoint_path
    checkpoint_path = args.checkpoint
    
    print(f"\n{'='*70}")
    print(f"LAM QUALITY EVALUATION")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"{'='*70}")
    
    # Load model
    print("\nLoading model...")
    model, config, checkpoint = load_lam_from_checkpoint(args.checkpoint, args.device)
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Run comprehensive evaluation
    metrics = generate_quality_report(
        model, config, checkpoint, 
        device=args.device, 
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
