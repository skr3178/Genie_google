# Tokenizer Evaluation

This directory contains evaluation scripts and results for the Video Tokenizer model.

## Evaluation Script

The `evaluate_tokenizer.py` script evaluates the tokenizer by:
1. Loading a trained tokenizer checkpoint
2. Loading video sequences from the dataset
3. Encoding and decoding the videos
4. Computing reconstruction metrics (PSNR, MSE)
5. Creating visual comparisons (side-by-side grids)
6. Saving individual frames for inspection

## Usage

```bash
# Activate conda environment
conda activate robot_wm

# Run evaluation
python evaluations/evaluate_tokenizer.py \
    --tokenizer_path checkpoints/tokenizer/checkpoint_step_1707.pt \
    --data_path data/pong_frames.h5 \
    --num_samples 5 \
    --output_dir evaluations/tokenizer
```

### Arguments

- `--tokenizer_path`: Path to tokenizer checkpoint (required)
- `--data_path`: Path to HDF5 dataset file (default: `data/pong_frames.h5`)
- `--config`: Path to tokenizer config file (default: `configs/tokenizer_config.yaml`)
- `--num_samples`: Number of video sequences to evaluate (default: 5)
- `--output_dir`: Output directory for results (default: `evaluations/tokenizer`)
- `--device`: Device to run on (default: `cuda` if available, else `cpu`)
- `--start_idx`: Starting frame index in dataset (default: 0)

## Evaluation Results

### Checkpoint: `checkpoint_step_1707.pt`
**Dataset**: `pong_frames.h5`  
**Training Step**: 1707

### Metrics Summary

| Metric | Value |
|--------|-------|
| **Average PSNR** | 34.90 dB |
| **Std PSNR** | 3.48 dB |
| **Min PSNR** | 33.02 dB |
| **Max PSNR** | 41.85 dB |
| **Average MSE** | 0.000401 |
| **Std MSE** | 0.000169 |

### Per-Sample Results

| Sample | PSNR (dB) | Notes |
|--------|-----------|-------|
| Sample 1 | 33.02 | Standard reconstruction quality |
| Sample 2 | 33.45 | Standard reconstruction quality |
| Sample 3 | 33.05 | Standard reconstruction quality |
| Sample 4 | 33.14 | Standard reconstruction quality |
| Sample 5 | 41.85 | High quality (likely simpler/static scene) |

### Vector Quantization Metrics

- **Commitment Loss**: ~0.0009-0.0010 (indicates encoder is committing to codebook vectors)
- **Codebook Loss**: ~0.003-0.004 (indicates codebook is being updated)

## Output Files

The evaluation generates:

1. **Comparison Grids**: `comparison_sample_*.png`
   - Side-by-side visualization of original vs reconstructed frames
   - Shows all frames in the sequence for easy comparison

2. **Individual Frames**: 
   - `sample_*_original/`: Original video frames
   - `sample_*_reconstructed/`: Reconstructed video frames
   - Saved as PNG images for detailed inspection

3. **Summary**: `summary.txt`
   - Text file with all metrics and per-sample results

## Understanding the Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- **Higher is better** (measured in dB)
- Typical values:
  - 20-30 dB: Acceptable quality
  - 30-40 dB: Good quality
  - 40+ dB: Excellent quality
- Our results (34.90 dB average) indicate **good reconstruction quality** at step 1707

### MSE (Mean Squared Error)
- **Lower is better**
- Measures average squared difference between original and reconstructed pixels
- Our results (0.000401 average) indicate low reconstruction error

### VQ Metrics
- **Commitment Loss**: Encourages encoder to commit to codebook vectors
- **Codebook Loss**: Encourages codebook vectors to match encoder outputs
- Both should be relatively low and stable for good training

## Model Configuration

The evaluated tokenizer uses:

- **Encoder**: 6 layers, 384 d_model, 6 heads
- **Decoder**: 8 layers, 384 d_model, 6 heads
- **Codebook**: 512 codes, 32 latent_dim
- **Patch Size**: 4
- **Sequence Length**: 8 frames
- **Resolution**: 128x72 pixels

## Interpretation

At training step 1707, the tokenizer shows:
- ✅ **Good reconstruction quality** (34.90 dB average PSNR)
- ✅ **Consistent performance** across most samples (33-34 dB)
- ✅ **Proper VQ training** (commitment and codebook losses are reasonable)
- ✅ **Learning meaningful representations** (able to reconstruct video structure)

The tokenizer is learning well and can be used for downstream tasks (LAM and Dynamics model training).

## Next Steps

1. Continue training to improve reconstruction quality
2. Evaluate on different datasets to check generalization
3. Use the tokenizer for LAM and Dynamics model training
4. Monitor metrics as training progresses

## Notes

- The evaluation is **independent** of LAM and Dynamics models
- Only requires the tokenizer checkpoint and dataset
- Can be run at any point during tokenizer training to monitor progress
