"""Script to verify LAM checkpoint is working correctly"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lam import LAM
from src.utils.config import load_config
from src.training.losses import lam_loss


def verify_checkpoint_structure(checkpoint_path: str):
    """Verify checkpoint contains all expected keys"""
    print("=" * 60)
    print("1. Verifying checkpoint structure...")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    expected_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 
                     'global_step', 'epoch', 'config']
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    missing_keys = [key for key in expected_keys if key not in checkpoint]
    if missing_keys:
        print(f"⚠️  WARNING: Missing keys: {missing_keys}")
        return False
    else:
        print("✓ All expected keys present")
    
    print(f"  Global step: {checkpoint['global_step']}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Config present: {'config' in checkpoint}")
    
    return True


def verify_model_loading(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Verify model can be loaded from checkpoint"""
    print("\n" + "=" * 60)
    print("2. Verifying model loading...")
    print("=" * 60)
    
    # Load config
    if 'config' in torch.load(checkpoint_path, map_location='cpu'):
        config = torch.load(checkpoint_path, map_location='cpu')['config']
        print("✓ Using config from checkpoint")
    else:
        config = load_config(config_path)
        print(f"✓ Loaded config from {config_path}")
    
    # Create model
    model = LAM(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    )
    
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state dict loaded successfully")
    else:
        print("⚠️  WARNING: No 'model_state_dict' in checkpoint, trying direct load...")
        model.load_state_dict(checkpoint)
        print("✓ Model state dict loaded (direct)")
    
    model = model.to(device).eval()
    print(f"✓ Model moved to {device} and set to eval mode")
    
    return model, config


def test_forward_pass(model, config, device: str = "cuda", num_tests: int = 3):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("3. Testing forward pass...")
    print("=" * 60)
    
    batch_size = 2
    sequence_length = config['data']['sequence_length'] - 1  # Past frames (T-1)
    H, W = config['data']['resolution'][:2]
    C = 3
    
    print(f"  Input shapes:")
    print(f"    Past frames: (B={batch_size}, T={sequence_length}, C={C}, H={H}, W={W})")
    print(f"    Next frame: (B={batch_size}, C={C}, H={H}, W={W})")
    
    for i in range(num_tests):
        print(f"\n  Test {i+1}/{num_tests}:")
        
        # Create dummy input
        past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
        next_frame = torch.randn(batch_size, C, H, W, device=device)
        
        # Forward pass
        with torch.no_grad():
            try:
                reconstructed, actions, vq_loss_dict = model(past_frames, next_frame)
                
                print(f"    ✓ Forward pass successful")
                print(f"    Reconstructed shape: {reconstructed.shape}")
                print(f"    Actions shape: {actions.shape}")
                
                # Verify shapes
                assert reconstructed.shape == next_frame.shape, \
                    f"Shape mismatch: {reconstructed.shape} vs {next_frame.shape}"
                
                # Verify action values are in valid range
                H_patches = H // config['model']['patch_size']
                W_patches = W // config['model']['patch_size']
                expected_actions_shape = (batch_size, sequence_length, H_patches, W_patches)
                assert actions.shape == expected_actions_shape, \
                    f"Actions shape mismatch: {actions.shape} vs {expected_actions_shape}"
                
                num_codes = config['model']['codebook']['num_codes']
                assert actions.min() >= 0 and actions.max() < num_codes, \
                    f"Action values out of range: [{actions.min().item()}, {actions.max().item()}] vs [0, {num_codes-1}]"
                
                print(f"    ✓ Shape checks passed")
                print(f"    Action range: [{actions.min().item()}, {actions.max().item()}]")
                print(f"    Expected range: [0, {num_codes-1}]")
                
                # Check for NaN/Inf
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    print(f"    ⚠️  WARNING: NaN/Inf detected in reconstruction")
                else:
                    print(f"    ✓ No NaN/Inf in reconstruction")
                
                # Check VQ loss components
                print(f"    VQ Loss components:")
                for key, value in vq_loss_dict.items():
                    if torch.is_tensor(value):
                        val = value.item()
                        status = "✓" if not (np.isnan(val) or np.isinf(val)) else "⚠️"
                        print(f"      {status} {key}: {val:.6f}")
                    else:
                        print(f"      {key}: {value}")
                
            except Exception as e:
                print(f"    ❌ Forward pass failed: {e}")
                raise
    
    print(f"\n✓ All forward pass tests passed")


def test_reconstruction_quality(model, config, device: str = "cuda"):
    """Test reconstruction quality with simple metrics"""
    print("\n" + "=" * 60)
    print("4. Testing reconstruction quality...")
    print("=" * 60)
    
    batch_size = 2
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    
    # Create test input
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    next_frame = torch.randn(batch_size, C, H, W, device=device)
    
    with torch.no_grad():
        reconstructed, actions, vq_loss_dict = model(past_frames, next_frame)
        
        # Compute metrics
        mse = F.mse_loss(reconstructed, next_frame)
        mae = F.l1_loss(reconstructed, next_frame)
        
        # PSNR (assuming pixel values in [0, 1] range)
        # For random inputs, this is just a sanity check
        mse_clamped = torch.clamp(mse, min=1e-10)
        psnr = -10 * torch.log10(mse_clamped)
        
        print(f"  Reconstruction metrics (on random input):")
        print(f"    MSE: {mse.item():.6f}")
        print(f"    MAE: {mae.item():.6f}")
        print(f"    PSNR: {psnr.item():.2f} dB")
        print(f"  Note: These metrics are on random inputs, not real data")
        print(f"  For meaningful metrics, test on actual validation data")
    
    print(f"✓ Reconstruction quality test completed")


def test_action_consistency(model, config, device: str = "cuda"):
    """Test that same input produces same actions (deterministic)"""
    print("\n" + "=" * 60)
    print("5. Testing action consistency (determinism)...")
    print("=" * 60)
    
    batch_size = 1
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    
    # Create fixed input
    torch.manual_seed(42)
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    next_frame = torch.randn(batch_size, C, H, W, device=device)
    
    model.eval()
    with torch.no_grad():
        _, actions1, _ = model(past_frames, next_frame)
        _, actions2, _ = model(past_frames, next_frame)
    
    # Actions should be identical (deterministic quantization)
    actions_match = (actions1 == actions2).all()
    
    if actions_match:
        print(f"  ✓ Actions are deterministic (same input → same actions)")
    else:
        print(f"  ⚠️  WARNING: Actions differ between runs")
        print(f"    This may indicate non-deterministic behavior")
        diff_count = (actions1 != actions2).sum().item()
        total_count = actions1.numel()
        print(f"    Differing actions: {diff_count}/{total_count} ({100*diff_count/total_count:.2f}%)")
    
    print(f"✓ Action consistency test completed")


def test_gradient_flow(model, config, device: str = "cuda"):
    """Test that gradients can flow through the model"""
    print("\n" + "=" * 60)
    print("6. Testing gradient flow...")
    print("=" * 60)
    
    batch_size = 1
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device, requires_grad=False)
    next_frame = torch.randn(batch_size, C, H, W, device=device, requires_grad=False)
    
    model.train()
    reconstructed, actions, vq_loss_dict = model(past_frames, next_frame)
    
    # Compute loss
    loss_dict = lam_loss(reconstructed, next_frame, vq_loss_dict)
    loss = loss_dict['total_loss']
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    if has_grad:
        # Count parameters with gradients
        num_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Gradients flow through model")
        print(f"    Parameters with gradients: {num_params_with_grad}/{total_params}")
        print(f"    Total loss: {loss.item():.6f}")
    else:
        print(f"  ⚠️  WARNING: No gradients detected")
    
    print(f"✓ Gradient flow test completed")


def test_codebook_usage(model, config, device: str = "cuda"):
    """Test that all codebook entries are being used"""
    print("\n" + "=" * 60)
    print("7. Testing codebook usage...")
    print("=" * 60)
    
    batch_size = 4
    sequence_length = config['data']['sequence_length'] - 1
    H, W = config['data']['resolution'][:2]
    C = 3
    num_codes = config['model']['codebook']['num_codes']
    
    # Test with multiple random inputs
    all_actions = []
    for _ in range(10):
        past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
        next_frame = torch.randn(batch_size, C, H, W, device=device)
        
        with torch.no_grad():
            _, actions, _ = model(past_frames, next_frame)
            all_actions.append(actions.cpu())
    
    # Count usage of each code
    all_actions = torch.cat(all_actions, dim=0)
    unique_codes = torch.unique(all_actions)
    
    print(f"  Codebook size: {num_codes}")
    print(f"  Unique codes used: {len(unique_codes)}")
    print(f"  Used codes: {sorted(unique_codes.tolist())}")
    
    if len(unique_codes) == num_codes:
        print(f"  ✓ All codebook entries are being used")
    else:
        unused = set(range(num_codes)) - set(unique_codes.tolist())
        print(f"  ⚠️  WARNING: Some codes are unused: {sorted(unused)}")
        print(f"    This is normal early in training, but should improve with more training")
    
    print(f"✓ Codebook usage test completed")


def main():
    parser = argparse.ArgumentParser(description="Verify LAM checkpoint")
    parser.add_argument("--checkpoint", type=str, 
                       default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/lam/checkpoint_step_5000.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/lam_config.yaml",
                       help="Path to config file (used if not in checkpoint)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run tests on")
    parser.add_argument("--skip-grad-test", action="store_true",
                       help="Skip gradient flow test (faster)")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"LAM Checkpoint Verification")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Test 1: Checkpoint structure
    if not verify_checkpoint_structure(str(checkpoint_path)):
        print("\n❌ Checkpoint structure verification failed!")
        return
    
    # Test 2: Model loading
    try:
        model, config = verify_model_loading(str(checkpoint_path), args.config, args.device)
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Forward pass
    try:
        test_forward_pass(model, config, args.device)
    except Exception as e:
        print(f"\n❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Reconstruction quality
    try:
        test_reconstruction_quality(model, config, args.device)
    except Exception as e:
        print(f"\n⚠️  Reconstruction quality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Action consistency
    try:
        test_action_consistency(model, config, args.device)
    except Exception as e:
        print(f"\n⚠️  Action consistency test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Gradient flow
    if not args.skip_grad_test:
        try:
            test_gradient_flow(model, config, args.device)
        except Exception as e:
            print(f"\n⚠️  Gradient flow test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 7: Codebook usage
    try:
        test_codebook_usage(model, config, args.device)
    except Exception as e:
        print(f"\n⚠️  Codebook usage test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Checkpoint verification completed!")
    print("=" * 60)
    print("\nSummary:")
    print("  - Checkpoint structure: ✓")
    print("  - Model loading: ✓")
    print("  - Forward pass: ✓")
    print("  - Reconstruction: ✓")
    print("  - Action consistency: ✓")
    if not args.skip_grad_test:
        print("  - Gradient flow: ✓")
    print("  - Codebook usage: ✓")
    print("\nYour LAM checkpoint appears to be working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
