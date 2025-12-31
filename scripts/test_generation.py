"""Quick test script to verify the generation pipeline works"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import GenieGenerator
from src.models.video_tokenizer import VideoTokenizer
from src.models.dynamics import DynamicsModel
from src.utils.config import load_config


def test_generation():
    """Test the complete generation pipeline"""
    
    print("=" * 60)
    print("Testing Genie Video Generation Pipeline")
    print("=" * 60)
    
    # Load configs
    print("\n1. Loading configs...")
    tokenizer_config = load_config("configs/tokenizer_config.yaml")
    dynamics_config = load_config("configs/dynamics_config.yaml")
    print("   Note: LAM is NOT needed for inference - only tokenizer and dynamics model are used.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Create models
    print("\n2. Creating models...")
    tokenizer = VideoTokenizer(
        encoder_config=tokenizer_config['model']['encoder'],
        decoder_config=tokenizer_config['model']['decoder'],
        codebook_config=tokenizer_config['model']['codebook'],
        patch_size=tokenizer_config['model']['patch_size'],
    )
    
    dynamics_model = DynamicsModel(
        architecture_config=dynamics_config['model']['architecture'],
        token_embedding_config=dynamics_config['model']['token_embedding'],
        action_embedding_config=dynamics_config['model']['action_embedding'],
    )
    
    print("   ✓ All models created")
    
    # Check if checkpoints exist
    print("\n3. Checking for checkpoints...")
    checkpoint_dir = Path("checkpoints")
    tokenizer_checkpoints = list((checkpoint_dir / "tokenizer").glob("*.pt")) + list((checkpoint_dir / "tokenizer").glob("*.pth"))
    dynamics_checkpoints = list((checkpoint_dir / "dynamics").glob("*.pt")) + list((checkpoint_dir / "dynamics").glob("*.pth"))
    
    if not tokenizer_checkpoints:
        print("   ⚠ No tokenizer checkpoints found. Skipping checkpoint loading test.")
        return
    if not dynamics_checkpoints:
        print("   ⚠ No dynamics checkpoints found. Skipping checkpoint loading test.")
        return
    
    # Use latest checkpoints
    tokenizer_path = max(tokenizer_checkpoints, key=lambda p: p.stat().st_mtime)
    dynamics_path = max(dynamics_checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Dynamics: {dynamics_path}")
    
    # Load checkpoints
    print("\n4. Loading checkpoints...")
    try:
        tokenizer_checkpoint = torch.load(tokenizer_path, map_location=device)
        if 'model_state_dict' in tokenizer_checkpoint:
            tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
        else:
            tokenizer.load_state_dict(tokenizer_checkpoint)
        print("   ✓ Tokenizer loaded")
        
        dynamics_checkpoint = torch.load(dynamics_path, map_location=device)
        if 'model_state_dict' in dynamics_checkpoint:
            dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
        else:
            dynamics_model.load_state_dict(dynamics_checkpoint)
        print("   ✓ Dynamics model loaded")
    except Exception as e:
        print(f"   ✗ Error loading checkpoints: {e}")
        return
    
    # Create generator
    print("\n5. Creating generator...")
    generator = GenieGenerator(
        tokenizer=tokenizer,
        dynamics_model=dynamics_model,
        device=device,
        maskgit_steps=8,  # Use fewer steps for testing
        temperature=2.0,
    )
    print("   ✓ Generator created")
    
    # Test generation with dummy prompt
    print("\n6. Testing video generation...")
    try:
        # Create dummy prompt frame (normalized to [0, 1])
        dummy_prompt = torch.rand(1, 3, 128, 72).to(device)
        actions = [0, 1, 2, 3, 4, 5, 6, 7]  # Example action sequence
        
        print(f"   Prompt shape: {dummy_prompt.shape}")
        print(f"   Actions: {actions}")
        print(f"   Generating 8 frames...")
        
        video = generator.generate(dummy_prompt, actions, num_frames=8)
        
        print(f"   ✓ Video generated successfully!")
        print(f"   Video shape: {video.shape}")
        print(f"   Video range: [{video.min().item():.3f}, {video.max().item():.3f}]")
        
        # Check for NaN/Inf
        if torch.isnan(video).any() or torch.isinf(video).any():
            print("   ⚠ Warning: Video contains NaN or Inf values")
        else:
            print("   ✓ Video values are valid (no NaN/Inf)")
        
    except Exception as e:
        print(f"   ✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✓ Pipeline test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_generation()
