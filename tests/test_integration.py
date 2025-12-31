"""Integration test combining Video Tokenizer, LAM, and Dynamics Model"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.video_tokenizer import VideoTokenizer
from src.models.lam import LAM
from src.models.dynamics import DynamicsModel
from src.utils.config import load_config


def test_integration():
    """Test all three components working together"""
    
    print("=" * 60)
    print("Integration Test: Video Tokenizer + LAM + Dynamics Model")
    print("=" * 60)
    
    # Load configs
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    lam_config = load_config("configs/lam_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create all models
    print("\n1. Creating all models...")
    
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    ).to(device).eval()
    
    lam = LAM(
        encoder_config=lam_config['model']['encoder'],
        decoder_config=lam_config['model']['decoder'],
        codebook_config=lam_config['model']['codebook'],
        patch_size=lam_config['model']['patch_size'],
    ).to(device).eval()
    
    dynamics = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    ).to(device).eval()
    
    print(f"   ✓ All models created on {device}")
    
    # Create dummy video sequence
    batch_size = 2
    sequence_length = 4
    H, W = 128, 72
    C = 3
    
    print(f"\n2. Creating dummy video sequence...")
    print(f"   Shape: (B={batch_size}, T={sequence_length}, C={C}, H={H}, W={W})")
    video_frames = torch.randn(batch_size, sequence_length, C, H, W, device=device)
    
    # Step 1: Tokenize video frames
    print(f"\n3. Step 1: Tokenizing video frames with Video Tokenizer...")
    with torch.no_grad():
        video_tokens = tokenizer.encode(video_frames)
    
    print(f"   ✓ Video tokenization successful")
    print(f"   Video tokens shape: {video_tokens.shape}")
    
    # Verify we can decode back
    decoded_frames = tokenizer.decode(video_tokens)
    print(f"   ✓ Decode test: {decoded_frames.shape}")
    assert decoded_frames.shape == video_frames.shape, "Decode shape mismatch"
    
    # Step 2: Extract actions from frame transitions using LAM
    print(f"\n4. Step 2: Extracting actions from frame transitions with LAM...")
    
    # Split into past frames and next frame
    past_frames = video_frames[:, :-1]  # (B, T-1, C, H, W)
    next_frame = video_frames[:, -1]    # (B, C, H, W)
    
    with torch.no_grad():
        _, lam_actions, _ = lam(past_frames, next_frame)
    
    print(f"   ✓ Action extraction successful")
    print(f"   LAM actions shape: {lam_actions.shape}")
    print(f"   Past frames shape: {past_frames.shape}")
    print(f"   Next frame shape: {next_frame.shape}")
    
    # Step 3: Use Dynamics Model to predict future tokens
    print(f"\n5. Step 3: Predicting future tokens with Dynamics Model...")
    
    # Align shapes: tokens and actions should have same temporal dimension
    # For dynamics model, we need tokens and actions for each timestep
    past_tokens = video_tokens[:, :-1]  # (B, T-1, H_patches, W_patches)
    past_actions = lam_actions  # (B, T-1, H_patches, W_patches) from LAM
    
    # Get the last action to predict next frame
    last_action = past_actions[:, -1:]  # (B, 1, H_patches, W_patches)
    
    # Use last token and last action to predict next token
    last_token = past_tokens[:, -1:]  # (B, 1, H_patches, W_patches)
    
    with torch.no_grad():
        # Predict next token logits
        next_token_logits = dynamics(last_token, last_action)
    
    print(f"   ✓ Token prediction successful")
    print(f"   Next token logits shape: {next_token_logits.shape}")
    
    # Sample next token
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(
        next_token_probs.view(-1, tokenizer_config['model']['codebook']['num_codes']),
        1
    ).view(batch_size, 1, *video_tokens.shape[2:])
    
    print(f"   ✓ Next token sampled")
    print(f"   Next token shape: {next_token.shape}")
    
    # Step 4: Decode predicted token back to frame
    print(f"\n6. Step 4: Decoding predicted token back to frame...")
    
    with torch.no_grad():
        predicted_frame = tokenizer.decode(next_token)
    
    print(f"   ✓ Frame reconstruction successful")
    print(f"   Predicted frame shape: {predicted_frame.shape}")
    expected_frame_shape = (batch_size, 1, C, H, W)
    assert predicted_frame.shape == expected_frame_shape, \
        f"Predicted frame shape mismatch: {predicted_frame.shape} vs {expected_frame_shape}"
    
    # Step 5: Test full autoregressive generation
    print(f"\n7. Step 5: Testing full autoregressive generation...")
    
    # Start with first frame
    initial_frame = video_frames[:, 0:1]  # (B, 1, C, H, W)
    initial_tokens = tokenizer.encode(initial_frame)  # (B, 1, H_patches, W_patches)
    
    # Generate a few frames autoregressively
    generated_tokens = [initial_tokens]
    num_generate = 3
    
    for step in range(num_generate):
        # Get last token
        last_token = generated_tokens[-1][:, -1:]  # (B, 1, H_patches, W_patches)
        
        # For simplicity, use a dummy action (in real scenario, this would come from LAM)
        # In practice, you'd need to decode the last frame, use LAM to get action, etc.
        dummy_action = torch.randint(
            0, lam_config['model']['codebook']['num_codes'],
            (batch_size, 1, *last_token.shape[2:]),
            device=device
        )
        
        # Predict next token using MaskGIT iterative refinement
        with torch.no_grad():
            # Use iterative refinement for better quality
            next_token = dynamics.iterative_refinement(
                tokens=last_token,
                actions=dummy_action,
                steps=4,  # Fewer steps for testing
                temperature=2.0,
                r=0.5,
            )
        
        generated_tokens.append(next_token)
    
    # Concatenate all generated tokens
    all_generated_tokens = torch.cat(generated_tokens, dim=1)
    print(f"   ✓ Autoregressive generation successful")
    print(f"   Generated tokens shape: {all_generated_tokens.shape}")
    
    # Decode all generated tokens
    with torch.no_grad():
        generated_frames = tokenizer.decode(all_generated_tokens)
    
    print(f"   ✓ Generated frames shape: {generated_frames.shape}")
    expected_generated_shape = (batch_size, num_generate + 1, C, H, W)
    assert generated_frames.shape == expected_generated_shape, \
        f"Generated frames shape mismatch: {generated_frames.shape} vs {expected_generated_shape}"
    
    # Step 6: Test end-to-end pipeline consistency
    print(f"\n8. Step 6: Testing end-to-end pipeline consistency...")
    
    # Check that shapes are compatible throughout
    tokenizer_patch_size = tokenizer_config['model']['patch_size']
    lam_patch_size = lam_config['model']['patch_size']
    
    H_patches_tokenizer = H // tokenizer_patch_size
    W_patches_tokenizer = W // tokenizer_patch_size
    H_patches_lam = H // lam_patch_size
    W_patches_lam = W // lam_patch_size
    
    print(f"   Tokenizer patch size: {tokenizer_patch_size}, patches: {H_patches_tokenizer}x{W_patches_tokenizer}")
    print(f"   LAM patch size: {lam_patch_size}, patches: {H_patches_lam}x{W_patches_lam}")
    
    # Note: LAM and tokenizer use different patch sizes, so action maps and token maps
    # will have different spatial dimensions. In practice, you may need to interpolate
    # or handle this mismatch.
    
    if H_patches_tokenizer != H_patches_lam or W_patches_tokenizer != W_patches_lam:
        print(f"   ⚠ Warning: Patch size mismatch between tokenizer and LAM")
        print(f"   This is expected - you may need to interpolate actions to match token spatial dimensions")
    else:
        print(f"   ✓ Patch sizes compatible")
    
    # Test that all components can work with the same batch size
    print(f"\n9. Testing batch size consistency...")
    for bs in [1, 2, 4]:
        test_frames = torch.randn(bs, sequence_length, C, H, W, device=device)
        
        with torch.no_grad():
            test_tokens = tokenizer.encode(test_frames)
            test_past = test_frames[:, :-1]
            test_next = test_frames[:, -1]
            _, test_actions, _ = lam(test_past, test_next)
            test_logits = dynamics(test_tokens[:, :-1], test_actions)
        
        assert test_tokens.shape[0] == bs
        assert test_actions.shape[0] == bs
        assert test_logits.shape[0] == bs
        print(f"   ✓ Batch size {bs}: OK")
    
    print(f"\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("  - Video Tokenizer: ✓ Encoding/decoding works")
    print("  - LAM: ✓ Action extraction works")
    print("  - Dynamics Model: ✓ Token prediction works")
    print("  - Integration: ✓ All components work together")
    print("=" * 60)
    
    return {
        'tokenizer': tokenizer,
        'lam': lam,
        'dynamics': dynamics,
        'video_tokens': video_tokens,
        'lam_actions': lam_actions,
    }


if __name__ == "__main__":
    test_integration()
