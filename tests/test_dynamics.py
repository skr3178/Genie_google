"""Modular test for Dynamics Model component"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dynamics import DynamicsModel
from src.utils.config import load_config


def test_dynamics():
    """Test Dynamics Model component in isolation"""
    
    print("=" * 60)
    print("Testing Dynamics Model Component")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/dynamics_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print("\n1. Creating Dynamics Model...")
    dynamics = DynamicsModel(
        architecture_config=config['model']['architecture'],
        token_embedding_config=config['model']['token_embedding'],
        action_embedding_config=config['model']['action_embedding'],
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in dynamics.parameters())
    print(f"   Model created with {num_params:,} parameters")
    print(f"   Device: {device}")
    
    # Create dummy input
    batch_size = 2
    sequence_length = 4
    H_patches = 32  # 128 / 4 (from tokenizer patch_size)
    W_patches = 18  # 72 / 4
    vocab_size = config['model']['token_embedding']['vocab_size']
    num_actions = config['model']['action_embedding']['num_actions']
    
    print(f"\n2. Creating dummy input...")
    print(f"   Tokens shape: (B={batch_size}, T={sequence_length}, H_patches={H_patches}, W_patches={W_patches})")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Num actions: {num_actions}")
    
    # Create tokens (from video tokenizer)
    tokens = torch.randint(0, vocab_size, (batch_size, sequence_length, H_patches, W_patches), device=device)
    
    # Create actions (spatial action maps from LAM)
    actions = torch.randint(0, num_actions, (batch_size, sequence_length, H_patches, W_patches), device=device)
    
    # Test forward pass (without mask)
    print(f"\n3. Testing forward pass (without mask)...")
    dynamics.eval()
    with torch.no_grad():
        logits = dynamics(tokens, actions)
    
    print(f"   ✓ Forward pass successful")
    print(f"   Logits shape: {logits.shape}")
    expected_shape = (batch_size, sequence_length, H_patches, W_patches, vocab_size)
    assert logits.shape == expected_shape, \
        f"Logits shape mismatch: {logits.shape} vs {expected_shape}"
    
    # Test forward pass with mask (MaskGIT training)
    print(f"\n4. Testing forward pass with mask (MaskGIT training)...")
    mask = dynamics.generate_mask(
        shape=(batch_size, sequence_length, H_patches, W_patches),
        mask_prob=0.5,
        device=device,
    )
    
    with torch.no_grad():
        logits_masked = dynamics(tokens, actions, mask=mask)
    
    print(f"   ✓ Masked forward pass successful")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask ratio: {mask.mean().item():.2%}")
    assert logits_masked.shape == expected_shape, "Masked logits shape mismatch"
    
    # Test token embedding
    print(f"\n5. Testing token embedding...")
    with torch.no_grad():
        token_emb = dynamics.decoder.token_embedding(tokens)
    
    print(f"   ✓ Token embedding successful")
    print(f"   Token embedding shape: {token_emb.shape}")
    expected_emb_shape = (batch_size, sequence_length, H_patches, W_patches, config['model']['architecture']['d_model'])
    assert token_emb.shape == expected_emb_shape, "Token embedding shape mismatch"
    
    # Test action embedding (spatial action maps)
    print(f"\n6. Testing action embedding (spatial action maps)...")
    with torch.no_grad():
        action_emb = dynamics.decoder.action_embedding(actions)
    
    print(f"   ✓ Action embedding successful")
    print(f"   Action embedding shape: {action_emb.shape}")
    assert action_emb.shape == expected_emb_shape, "Action embedding shape mismatch"
    
    # Test action embedding (temporal actions - single action per timestep)
    print(f"\n7. Testing action embedding (temporal actions)...")
    temporal_actions = torch.randint(0, num_actions, (batch_size, sequence_length), device=device)
    with torch.no_grad():
        temporal_action_emb = dynamics.decoder.action_embedding(temporal_actions)
    
    print(f"   ✓ Temporal action embedding successful")
    print(f"   Temporal action embedding shape: {temporal_action_emb.shape}")
    expected_temporal_shape = (batch_size, sequence_length, config['model']['architecture']['d_model'])
    assert temporal_action_emb.shape == expected_temporal_shape, "Temporal action embedding shape mismatch"
    
    # Test gradient flow
    print(f"\n8. Testing gradient flow...")
    dynamics.train()
    tokens.requires_grad_(False)
    actions.requires_grad_(False)
    
    logits = dynamics(tokens, actions, mask=mask)
    
    # Compute MaskGIT loss (cross-entropy on masked tokens)
    logits_flat = logits.view(-1, vocab_size)
    tokens_flat = tokens.view(-1)
    mask_flat = mask.view(-1).bool()
    
    loss = F.cross_entropy(
        logits_flat[mask_flat],
        tokens_flat[mask_flat],
    )
    loss.backward()
    
    # Check if gradients exist
    has_grad = any(p.grad is not None for p in dynamics.parameters() if p.requires_grad)
    print(f"   ✓ Gradient flow: {'OK' if has_grad else 'FAILED'}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test iterative refinement (MaskGIT inference)
    print(f"\n9. Testing iterative refinement (MaskGIT inference)...")
    dynamics.eval()
    
    # Start with random tokens
    initial_tokens = torch.randint(0, vocab_size, (batch_size, 1, H_patches, W_patches), device=device)
    initial_actions = torch.randint(0, num_actions, (batch_size, 1, H_patches, W_patches), device=device)
    
    with torch.no_grad():
        refined_tokens = dynamics.iterative_refinement(
            tokens=initial_tokens,
            actions=initial_actions,
            steps=4,  # Fewer steps for testing
            temperature=2.0,
            r=0.5,
        )
    
    print(f"   ✓ Iterative refinement successful")
    print(f"   Initial tokens shape: {initial_tokens.shape}")
    print(f"   Refined tokens shape: {refined_tokens.shape}")
    assert refined_tokens.shape == initial_tokens.shape, "Refined tokens shape mismatch"
    
    # Check token values are in valid range
    assert refined_tokens.min() >= 0 and refined_tokens.max() < vocab_size, \
        f"Refined token values out of range: min={refined_tokens.min()}, max={refined_tokens.max()}"
    
    # Test with different batch sizes
    print(f"\n10. Testing with different batch sizes...")
    for bs in [1, 4]:
        test_tokens = torch.randint(0, vocab_size, (bs, sequence_length, H_patches, W_patches), device=device)
        test_actions = torch.randint(0, num_actions, (bs, sequence_length, H_patches, W_patches), device=device)
        with torch.no_grad():
            test_logits = dynamics(test_tokens, test_actions)
        assert test_logits.shape[0] == bs, f"Batch size {bs} failed"
        print(f"   ✓ Batch size {bs}: OK")
    
    # Test mask generation with different probabilities
    print(f"\n11. Testing mask generation...")
    for mask_prob in [0.3, 0.5, 0.7, 1.0]:
        test_mask = dynamics.generate_mask(
            shape=(batch_size, sequence_length, H_patches, W_patches),
            mask_prob=mask_prob,
            device=device,
        )
        actual_ratio = test_mask.mean().item()
        print(f"   Requested: {mask_prob:.1%}, Actual: {actual_ratio:.1%}")
    
    print(f"\n" + "=" * 60)
    print("✓ All Dynamics Model tests passed!")
    print("=" * 60)
    
    return dynamics, logits


if __name__ == "__main__":
    test_dynamics()
