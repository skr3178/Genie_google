# Modular Component Tests

This directory contains modular tests for each component of the Genie model, allowing you to test each component individually before integrating them together.

## Test Files

### Individual Component Tests

1. **`test_video_tokenizer.py`** - Tests the Video Tokenizer component
   - Encodes video frames to discrete tokens
   - Decodes tokens back to frames
   - Tests forward pass, shape consistency, and gradient flow
   - Validates token quantization (1024 codebook)

2. **`test_lam.py`** - Tests the LAM (Latent Action Model) component
   - Encodes past frames + next frame to latent actions
   - Quantizes actions (8 codebook)
   - Decodes actions to reconstruct next frame
   - Tests encoder, quantizer, and decoder separately

3. **`test_dynamics.py`** - Tests the Dynamics Model component
   - Tests token and action embeddings
   - Tests MaskGIT forward pass with/without masking
   - Tests iterative refinement (MaskGIT inference)
   - Validates gradient flow and loss computation

### Integration Test

4. **`test_integration.py`** - Tests all three components working together
   - Video Tokenizer → LAM → Dynamics Model pipeline
   - End-to-end tokenization, action extraction, and token prediction
   - Autoregressive generation test
   - Shape compatibility checks

### Test Runner

5. **`run_all_tests.py`** - Runs all tests in sequence
   - Executes all individual component tests
   - Runs integration test
   - Provides summary of results

## Usage

### Activate Conda Environment

First, activate the conda environment:
```bash
conda activate robot_wm
```

### Run Individual Tests

```bash
# Test Video Tokenizer only
python tests/test_video_tokenizer.py

# Test LAM only
python tests/test_lam.py

# Test Dynamics Model only
python tests/test_dynamics.py

# Test integration
python tests/test_integration.py
```

### Run All Tests

```bash
# Run all tests sequentially
python tests/run_all_tests.py
```

Or use the convenience script:
```bash
bash tests/run_tests.sh
```

## What Each Test Validates

### Video Tokenizer Test
- ✓ Model creation and parameter counting
- ✓ Forward pass (encode → quantize → decode)
- ✓ Separate encode/decode methods
- ✓ Shape consistency (input/output)
- ✓ Token value range validation (0 to vocab_size-1)
- ✓ Gradient flow
- ✓ VQ loss components
- ✓ Different batch sizes

### LAM Test
- ✓ Model creation and parameter counting
- ✓ Forward pass (past frames + next frame → actions)
- ✓ Encoder, quantizer, and decoder separately
- ✓ Shape consistency
- ✓ Action value range validation (0 to 7)
- ✓ Gradient flow
- ✓ VQ loss components
- ✓ Action consistency (deterministic quantization)

### Dynamics Model Test
- ✓ Model creation and parameter counting
- ✓ Forward pass with/without masking
- ✓ Token and action embeddings (spatial and temporal)
- ✓ MaskGIT mask generation
- ✓ Gradient flow and loss computation
- ✓ Iterative refinement (MaskGIT inference)
- ✓ Different batch sizes
- ✓ Different mask probabilities

### Integration Test
- ✓ All models created and loaded
- ✓ Video tokenization pipeline
- ✓ Action extraction from frame transitions
- ✓ Token prediction using dynamics model
- ✓ Frame reconstruction from predicted tokens
- ✓ Autoregressive generation
- ✓ Shape compatibility checks
- ✓ Batch size consistency

## Expected Output

Each test will print:
- Component name and test progress
- Model parameter counts
- Input/output shapes
- Validation results (✓ or ❌)
- Summary of passed/failed tests

## Notes

1. **Patch Size Mismatch**: The Video Tokenizer uses patch_size=4, while LAM uses patch_size=16. This means action maps and token maps have different spatial dimensions. The integration test notes this - in practice, you may need to interpolate actions to match token spatial dimensions.

2. **Device**: Tests automatically use CUDA if available, otherwise CPU.

3. **Batch Sizes**: Tests use small batch sizes (1-4) for faster execution. You can modify these in the test files.

4. **Sequence Lengths**: Tests use shorter sequences (4 frames) for faster execution. Real training uses 16 frames.

## Troubleshooting

If a test fails:
1. Check the error message and traceback
2. Verify config files exist in `configs/` directory
3. Check that all dependencies are installed
4. Ensure CUDA is available if using GPU (or modify device in test files)
5. Check that model architectures match config files

## Next Steps

After all tests pass:
1. Train each component individually using the training scripts
2. Load trained checkpoints and test with real data
3. Run full inference pipeline with trained models
