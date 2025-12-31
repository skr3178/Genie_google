"""Modular test for Video Tokenizer component"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.video_tokenizer import VideoTokenizer
from src.utils.config import load_config


def test_video_tokenizer():
    """Test Video Tokenizer component in isolation"""
    
    print("=" * 60)
    print("Testing Video Tokenizer Component")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/tokenizer_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print("\n1. Creating Video Tokenizer model...")
    tokenizer = VideoTokenizer(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"   Model created with {num_params:,} parameters")
    print(f"   Device: {device}")
    
    # Create dummy input
    batch_size = 2
    sequence_length = 4  # Smaller for testing
    H, W = 128, 72  # From config
    C = 3
    
    print(f"\n2. Creating dummy input...")
    print(f"   Shape: (B={batch_size}, T={sequence_length}, C={C}, H={H}, W={W})")
    video_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    tokenizer.eval()
    with torch.no_grad():
        reconstructed, tokens, vq_loss_dict = tokenizer(video_frames)
    
    print(f"   ✓ Forward pass successful")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Tokens shape: {tokens.shape}")
    print(f"   Expected tokens shape: (B={batch_size}, T={sequence_length}, H_patches={H//config['model']['patch_size']}, W_patches={W//config['model']['patch_size']})")
    
    # Check shapes
    assert reconstructed.shape == video_frames.shape, f"Reconstruction shape mismatch: {reconstructed.shape} vs {video_frames.shape}"
    H_patches = H // config['model']['patch_size']
    W_patches = W // config['model']['patch_size']
    assert tokens.shape == (batch_size, sequence_length, H_patches, W_patches), \
        f"Tokens shape mismatch: {tokens.shape} vs expected (B, T, H_patches, W_patches)"
    
    # Check token values are in valid range
    assert tokens.min() >= 0 and tokens.max() < config['model']['codebook']['num_codes'], \
        f"Token values out of range: min={tokens.min()}, max={tokens.max()}, vocab_size={config['model']['codebook']['num_codes']}"
    
    print(f"   ✓ Shape checks passed")
    print(f"   Token range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    # Test encode/decode separately
    print(f"\n4. Testing encode/decode separately...")
    with torch.no_grad():
        encoded_tokens = tokenizer.encode(video_frames)
        decoded_frames = tokenizer.decode(encoded_tokens)
    
    print(f"   ✓ Encode/decode successful")
    print(f"   Encoded tokens shape: {encoded_tokens.shape}")
    print(f"   Decoded frames shape: {decoded_frames.shape}")
    
    assert encoded_tokens.shape == tokens.shape, "Encode output shape mismatch"
    assert decoded_frames.shape == video_frames.shape, "Decode output shape mismatch"
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    tokenizer.train()
    video_frames.requires_grad_(False)
    reconstructed, tokens, vq_loss_dict = tokenizer(video_frames)
    
    # Compute a dummy loss
    loss = (reconstructed - video_frames).abs().mean()
    loss.backward()
    
    # Check if gradients exist
    has_grad = any(p.grad is not None for p in tokenizer.parameters() if p.requires_grad)
    print(f"   ✓ Gradient flow: {'OK' if has_grad else 'FAILED'}")
    
    # Test VQ loss components
    print(f"\n6. VQ Loss components:")
    for key, value in vq_loss_dict.items():
        print(f"   {key}: {value.item():.6f}")
    
    # Test with different batch sizes
    print(f"\n7. Testing with different batch sizes...")
    for bs in [1, 4]:
        test_frames = torch.randn(bs, sequence_length, C, H, W, device=device)
        with torch.no_grad():
            _, test_tokens, _ = tokenizer(test_frames)
        assert test_tokens.shape[0] == bs, f"Batch size {bs} failed"
        print(f"   ✓ Batch size {bs}: OK")
    
    print(f"\n" + "=" * 60)
    print("✓ All Video Tokenizer tests passed!")
    print("=" * 60)
    
    return tokenizer, tokens


if __name__ == "__main__":
    test_video_tokenizer()
