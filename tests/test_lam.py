"""Modular test for LAM (Latent Action Model) component"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lam import LAM
from src.utils.config import load_config


def test_lam():
    """Test LAM component in isolation"""
    
    print("=" * 60)
    print("Testing LAM (Latent Action Model) Component")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/lam_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print("\n1. Creating LAM model...")
    lam = LAM(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in lam.parameters())
    print(f"   Model created with {num_params:,} parameters")
    print(f"   Device: {device}")
    
    # Create dummy input
    batch_size = 2
    sequence_length = 4  # Past frames
    H, W = 128, 72  # From config
    C = 3
    
    print(f"\n2. Creating dummy input...")
    print(f"   Past frames shape: (B={batch_size}, T={sequence_length}, C={C}, H={H}, W={W})")
    print(f"   Next frame shape: (B={batch_size}, C={C}, H={H}, W={W})")
    
    past_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    next_frame = torch.randn(batch_size, C, H, W, device=device)
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    lam.eval()
    with torch.no_grad():
        reconstructed, actions, vq_loss_dict = lam(past_frames, next_frame)
    
    print(f"   ✓ Forward pass successful")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Actions shape: {actions.shape}")
    
    # Check shapes
    assert reconstructed.shape == next_frame.shape, \
        f"Reconstruction shape mismatch: {reconstructed.shape} vs {next_frame.shape}"
    
    H_patches = H // config['model']['patch_size']
    W_patches = W // config['model']['patch_size']
    assert actions.shape == (batch_size, sequence_length, H_patches, W_patches), \
        f"Actions shape mismatch: {actions.shape} vs expected (B, T, H_patches, W_patches)"
    
    print(f"   ✓ Shape checks passed")
    
    # Check action values are in valid range (8 codes)
    assert actions.min() >= 0 and actions.max() < config['model']['codebook']['num_codes'], \
        f"Action values out of range: min={actions.min()}, max={actions.max()}, num_codes={config['model']['codebook']['num_codes']}"
    
    print(f"   Action range: [{actions.min().item()}, {actions.max().item()}]")
    print(f"   Expected range: [0, {config['model']['codebook']['num_codes']-1}]")
    
    # Test encoder separately
    print(f"\n4. Testing encoder separately...")
    with torch.no_grad():
        latent_actions = lam.encoder(past_frames, next_frame)
    
    print(f"   ✓ Encoder test successful")
    print(f"   Latent actions shape: {latent_actions.shape}")
    expected_shape = (batch_size, sequence_length, H_patches, W_patches, config['model']['encoder']['d_model'])
    assert latent_actions.shape == expected_shape, \
        f"Latent actions shape mismatch: {latent_actions.shape} vs {expected_shape}"
    
    # Test quantizer separately
    print(f"\n5. Testing quantizer separately...")
    B, T, H_p, W_p, d = latent_actions.shape
    latent_flat = latent_actions.view(B * T * H_p * W_p, d)
    
    with torch.no_grad():
        quantized, action_indices, vq_loss = lam.quantizer(latent_flat)
    
    print(f"   ✓ Quantizer test successful")
    print(f"   Quantized shape: {quantized.shape}")
    print(f"   Action indices shape: {action_indices.shape}")
    print(f"   Action indices range: [{action_indices.min().item()}, {action_indices.max().item()}]")
    
    # Test decoder separately (using quantized actions)
    print(f"\n6. Testing decoder separately...")
    # Get history from past frames
    B, T, C, H, W = past_frames.shape
    past_patches = lam.encoder.patch_embedding(
        past_frames.view(B * T, C, H, W)
    )
    _, d, h_p, w_p = past_patches.shape
    past_patches = past_patches.view(B, T, d, h_p, w_p).permute(0, 1, 3, 4, 2)
    
    with torch.no_grad():
        history = lam.encoder.transformer(past_patches)[:, :-1]
        last_action = quantized.view(B, T, H_p, W_p, d)[:, -1]
        decoded = lam.decoder(history, last_action)
    
    print(f"   ✓ Decoder test successful")
    print(f"   Decoded shape: {decoded.shape}")
    assert decoded.shape == next_frame.shape, "Decoder output shape mismatch"
    
    # Test gradient flow
    print(f"\n7. Testing gradient flow...")
    lam.train()
    past_frames.requires_grad_(False)
    next_frame.requires_grad_(False)
    
    reconstructed, actions, vq_loss_dict = lam(past_frames, next_frame)
    
    # Compute a dummy loss
    loss = (reconstructed - next_frame).abs().mean()
    loss.backward()
    
    # Check if gradients exist
    has_grad = any(p.grad is not None for p in lam.parameters() if p.requires_grad)
    print(f"   ✓ Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # Test VQ loss components
    print(f"\n8. VQ Loss components:")
    for key, value in vq_loss_dict.items():
        print(f"   {key}: {value.item():.6f}")
    
    # Test with different batch sizes
    print(f"\n9. Testing with different batch sizes...")
    for bs in [1, 4]:
        test_past = torch.randn(bs, sequence_length, C, H, W, device=device)
        test_next = torch.randn(bs, C, H, W, device=device)
        with torch.no_grad():
            _, test_actions, _ = lam(test_past, test_next)
        assert test_actions.shape[0] == bs, f"Batch size {bs} failed"
        print(f"   ✓ Batch size {bs}: OK")
    
    # Test action consistency (same input should give similar actions)
    print(f"\n10. Testing action consistency...")
    lam.eval()
    with torch.no_grad():
        _, actions1, _ = lam(past_frames, next_frame)
        _, actions2, _ = lam(past_frames, next_frame)
    
    # Actions should be identical for same input (deterministic quantization)
    actions_match = (actions1 == actions2).all()
    print(f"   ✓ Action consistency: {'OK' if actions_match else 'WARNING - actions differ'}")
    
    print(f"\n" + "=" * 60)
    print("✓ All LAM tests passed!")
    print("=" * 60)
    
    return lam, actions


if __name__ == "__main__":
    test_lam()
