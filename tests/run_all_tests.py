"""Run all modular tests in sequence"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_video_tokenizer import test_video_tokenizer
from test_lam import test_lam
from test_dynamics import test_dynamics
from test_integration import test_integration


def main():
    """Run all tests in sequence"""
    
    print("\n" + "=" * 80)
    print("Running All Modular Tests for Genie Components")
    print("=" * 80 + "\n")
    
    results = {}
    
    try:
        # Test 1: Video Tokenizer
        print("\n" + "=" * 80)
        print("TEST 1: Video Tokenizer")
        print("=" * 80)
        tokenizer, tokens = test_video_tokenizer()
        results['tokenizer'] = True
    except Exception as e:
        print(f"\n❌ Video Tokenizer test failed: {e}")
        results['tokenizer'] = False
        import traceback
        traceback.print_exc()
    
    try:
        # Test 2: LAM
        print("\n" + "=" * 80)
        print("TEST 2: LAM (Latent Action Model)")
        print("=" * 80)
        lam, actions = test_lam()
        results['lam'] = True
    except Exception as e:
        print(f"\n❌ LAM test failed: {e}")
        results['lam'] = False
        import traceback
        traceback.print_exc()
    
    try:
        # Test 3: Dynamics Model
        print("\n" + "=" * 80)
        print("TEST 3: Dynamics Model")
        print("=" * 80)
        dynamics, logits = test_dynamics()
        results['dynamics'] = True
    except Exception as e:
        print(f"\n❌ Dynamics Model test failed: {e}")
        results['dynamics'] = False
        import traceback
        traceback.print_exc()
    
    try:
        # Test 4: Integration
        print("\n" + "=" * 80)
        print("TEST 4: Integration Test")
        print("=" * 80)
        integration_results = test_integration()
        results['integration'] = True
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        results['integration'] = False
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
