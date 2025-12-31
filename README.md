# Genie Model Implementation

A minimal skeleton implementation of the Genie model architecture, scaled down for 16GB GPU training.

## Overview

This implementation includes:

- **Video Tokenizer (ST-ViViT)**: Compresses videos into discrete tokens
- **Latent Action Model (LAM)**: Learns discrete action space from frame transitions
- **Dynamics Model**: Predicts future frames autoregressively using MaskGIT
- **ST-Transformer Backbone**: Memory-efficient spatiotemporal transformer with factored attention

## Architecture

The model uses a sequential training approach:
1. Train Video Tokenizer on video frames
2. Train LAM on raw pixels (with tokenizer frozen)
3. Train Dynamics Model on tokens + actions (with tokenizer and LAM frozen)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Hyperparameters are configured via YAML files in `configs/`:

- `tokenizer_config.yaml`: Video tokenizer settings
- `lam_config.yaml`: LAM settings
- `dynamics_config.yaml`: Dynamics model settings

All models are scaled down from the original paper for 16GB GPU:
- Video Tokenizer: 6/10 layers (vs 12/20), d_model 384/512 (vs 512/1024)
- LAM: 8 layers (vs 20), d_model 512 (vs 1024)
- Dynamics: 12 layers (vs 48), d_model 768 (vs 5120)
- Total: ~150M parameters (vs 10.7B original)

## Usage

### Training

#### 1. Train Video Tokenizer

```bash
python scripts/train_tokenizer.py \
    --config configs/tokenizer_config.yaml \
    --data_dir data \
    --device cuda
```

#### 2. Train LAM

```bash
python scripts/train_lam.py \
    --config configs/lam_config.yaml \
    --data_dir data \
    --device cuda
```

#### 3. Train Dynamics Model

```bash
python scripts/train_dynamics.py \
    --config configs/dynamics_config.yaml \
    --data_dir data \
    --tokenizer_path checkpoints/tokenizer/checkpoint_step_30000.pt \
    --lam_path checkpoints/lam/checkpoint_step_20000.pt \
    --device cuda
```

### Inference

```bash
python scripts/inference.py \
    --prompt path/to/prompt_image.png \
    --actions "0,1,2,3,4,5,6,7" \
    --tokenizer_path checkpoints/tokenizer/checkpoint_step_30000.pt \
    --lam_path checkpoints/lam/checkpoint_step_20000.pt \
    --dynamics_path checkpoints/dynamics/checkpoint_step_10000.pt \
    --output output.mp4 \
    --num_frames 16 \
    --device cuda
```

## Data Format

The dataset loader expects H5 files in the `data/` directory with video frames stored as:
- Key: `frames` or `images`
- Shape: `(T, H, W, C)` or `(T, C, H, W)`
- Values: uint8 (0-255) or float (0-1)

## Model Components

### ST-Transformer
Memory-efficient transformer with factored spatial and temporal attention:
- Spatial attention: O(H×W) per frame
- Temporal attention: O(T) per spatial position
- Total complexity: O(T×H×W) instead of O((T×H×W)²)

### Video Tokenizer
ST-ViViT implementation:
- Encoder: 6 layers, d_model=384, patch_size=4
- Decoder: 10 layers, d_model=512, patch_size=4
- Codebook: 1024 codes, dim=32

### LAM
Latent Action Model:
- Encoder: 8 layers, d_model=512, patch_size=16
- Decoder: 8 layers, d_model=512, patch_size=16
- Codebook: 8 codes, dim=32

### Dynamics Model
MaskGIT-based dynamics predictor:
- 12 layers, d_model=768, num_heads=12, k/q_size=128
- Token embeddings: 1024 vocab size
- Action embeddings: 8 actions
- Inference: 25 MaskGIT steps, temperature=2.0

## Training Notes

- Mixed precision training (bfloat16) is enabled by default
- Gradient checkpointing is available for memory efficiency
- All models use AdamW optimizer with cosine learning rate decay
- Training steps are scaled down for faster iteration (10k-30k vs 300k original)

## References

- **STTN**: Conceptual reference for spatial attention patterns (https://github.com/researchmm/STTN)
- **VQ-VAE**: Conceptual reference for vector quantization (https://github.com/MishaLaskin/vqvae)
- All code implemented from scratch based on the Genie paper specifications

## License

This implementation is for educational purposes.
