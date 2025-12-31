# Genie Model Hyperparameters (Extracted from Paper)

Note: Model inputs are normalized between 0 and 1 and the final outputs of the decoder are placed through a sigmoid.

## 1. Latent Action Model (LAM) - 300M Parameters

### Encoder
- num_layers: 20
- d_model: 1024
- num_heads: 16

### Decoder
- num_layers: 20
- d_model: 1024
- num_heads: 16

### Codebook
- num_codes: 8
- patch_size: 16
- latent_dim (embedding size): 32

## 2. Video Tokenizer (ST-ViViT) - 200M Parameters

### Encoder
- num_layers: 12
- d_model: 512
- num_heads: 8
- k/q_size: 64

### Decoder
- num_layers: 20
- d_model: 1024
- num_heads: 16
- k/q_size: 64

### Codebook
- num_codes: 1024
- patch_size: 4
- latent_dim (embedding size): 32

### Video Tokenizer Optimizer (AdamW)
- max_lr: 3e-4
- min_lr: 3e-4
- beta1: 0.9
- beta2: 0.9
- weight_decay: 1e-4
- warmup_steps: 10k
- Training steps: 300k
- Learning rate schedule: Cosine decay

## 3. Dynamics Model (Final Genie) - 10.1B Parameters

### Architecture
- num_layers: 48
- num_heads: 36
- d_model: 5120
- k/q_size: 128
- Total parameters: 10.1B
- FLOPs: 6.6 × 10^22

### Training Configuration
- Batch size: 512
- Training steps: 125k
- Total tokens seen: ~942B
- Hardware: 256 TPUv5p
- Precision: bfloat16
- QK normalization: Yes

### Dynamics Model Optimizer (AdamW)
- max_lr: 3e-5
- min_lr: 3e-6
- beta1: 0.9
- beta2: 0.9
- weight_decay: 1e-4
- warmup_steps: 5k
- Learning rate schedule: Cosine decay

## 4. General Training Parameters

### Data Configuration
- Sequence length: 16 frames
- FPS: 10
- Resolution: 160x90x3
- Dataset: 6.8M video clips (30k hours) from 2D platformer games

### Inference Parameters
- MaskGIT steps: 25
- Temperature: 2
- Sampling: Random sampling

### Total Model Size
- Video Tokenizer: 200M parameters
- Latent Action Model: 300M parameters
- Dynamics Model: 10.1B parameters
- **Total Genie Model: 10.7B parameters** (reported as 11B in paper)


## Steps to create the model: 

1. Data Preparation and Preprocessing Module

Purpose: Handle input videos for training/inference.
Sub-Modules:
Video loading and framing: Extract sequences of T frames (e.g., T=16 at 10 FPS, resolution 160x90x3).
Normalization: Scale pixel values to [0,1].
Augmentation: Random cropping, flipping for diversity.
Filtering: Use a pre-trained classifier (e.g., ResNet18) to select high-quality clips.

Key Considerations for Code:
Use libraries like torchvision or decord for efficient video decoding.
Batch size: 512 for large-scale training.
Dataset: Source from open video datasets (e.g., Kinetics, Something-Something) as a proxy for Internet videos.


2. Memory-Efficient ST-Transformer Backbone

Purpose: Core module used in all components to process spatiotemporal data linearly (O(T * H * W) complexity) instead of quadratically.
Sub-Modules (Based on Figure 4: L stacked spatiotemporal blocks):
Spatial Attention Layer: Multi-head self-attention over spatial tokens (H x W patches) within each frame. Use standard transformer attention but scoped to single time steps.
Temporal Attention Layer: Multi-head self-attention over time steps (T) for each fixed spatial position, with causal masking to prevent future leakage.
Feed-Forward Network (FFN): Applied once per block after both attention layers (MLP with GELU activation; no separate FFN after spatial attention to save compute).
Layer Normalization and Residual Connections: Wrap each sub-layer.
Positional Embeddings: Add learnable spatial (2D sinusoidal) and temporal (1D) embeddings.
Causal Masking: Boolean mask for autoregressive generation (upper-triangular for time).

Key Considerations for Code:
Implement as a custom Transformer layer in PyTorch (e.g., subclass nn.TransformerEncoderLayer).
Dimensions: d_model (e.g., 512-5120), num_heads (8-36), dim_feedforward=4*d_model.
Efficiency: Use torch.nn.MultiheadAttention with custom masks; avoid full joint space-time attention.
Scaling: For videos, patchify frames (e.g., patch size 4 or 16) to reduce tokens.
Hyperparams: L=12-48 layers; use bfloat16 for mixed precision.


3. Video Tokenizer (ST-ViViT)

Purpose: Compress videos into discrete tokens for efficient modeling.
Sub-Modules:
Encoder: ST-Transformer to map raw frames to latent embeddings.
Quantizer: Vector Quantization (VQ) layer with codebook (e.g., 1024 codes, dim=32).
Decoder: ST-Transformer to reconstruct frames from tokens.

Workflow:
Input: Raw video frames → Patchify → ST-Transformer encode → Quantize → Decode → Reconstruct.

Key Considerations for Code:
Build as a VQ-VAE: Loss = reconstruction (MSE) + commitment loss + codebook loss.
Training: AdamW optimizer, cosine LR decay.
Hyperparams: Encoder (12 layers, d_model=512, 8 heads, k/q_size=64); Decoder (20 layers, d_model=1024, 16 heads, k/q_size=64); Codebook (1024 codes, patch_size=4, latent_dim=32).
Total: 200M parameters, trained for 300k steps.
Output: Discrete tokens z_{1:T} (shape: T x (H/4 * W/4)).


4. Latent Action Model (LAM)

Purpose: Learn a discrete action space unsupervised from frame transitions.
Sub-Modules:
Encoder: ST-Transformer processes past frames + next frame to predict continuous latent actions \tilde{a}_t.
Quantizer: VQ codebook (small size, e.g., 8 codes, dim=32) to discretize actions.
Decoder: ST-Transformer reconstructs next frame from history + quantized action.

Workflow:
Input: Pixels x_{1:t} and x_{t+1} → Encode to \tilde{a}t → Quantize to a_t → Decode to \hat{x}{t+1}.

Key Considerations for Code:
Loss: Reconstruction (sigmoid cross-entropy on pixels).
Input: Raw pixels (not tokens) for better controllability.
Training: Causal masking; generate all actions in one pass.
Hyperparams: Encoder/Decoder (20 layers each, d_model=1024, 16 heads); Codebook (8 codes, patch_size=16, latent_dim=32).
Total: 300M parameters.
Inference: Discard LAM; use discrete actions from codebook.


5. Dynamics Model

Purpose: Predict future frames autoregressively given past tokens and actions.
Sub-Modules:
Input Embedding: Token embeddings + additive action embeddings (stopgrad on actions).
MaskGIT Decoder: ST-Transformer with random masking (Bernoulli rate 0.5-1.0).
Output Head: Linear layer to predict token logits.

Workflow:
Input: z_{1:t-1} + \tilde{a}_{1:t-1} → Mask tokens → ST-Transformer → Predict masked z_t.

Key Considerations for Code:
Loss: Cross-entropy on masked tokens.
Training: Autoregressive with causal masks.
Inference: Start from prompt z_1; iteratively sample next tokens (25 MaskGIT steps, temp=2).
Hyperparams (10.1B params): 48 layers, d_model=5120, 36 heads, k/q_size=128.
Training: Batch size 512, 125k steps, ~942B tokens, bfloat16 precision, QK normalization.
Decode tokens back to images using tokenizer decoder.


6. Training and Optimization Pipeline

Purpose: Orchestrate training across components.
Sub-Modules:
Sequential training: Tokenizer → LAM → Dynamics (with optional co-training of LAM + Dynamics).
Optimizer: AdamW with weight decay, cosine LR scheduler.
Stability Tricks: bfloat16, Query-Key (QK) normalization.

Key Considerations for Code:
Total tokens seen: ~942B (for dynamics model).
Steps: 300k (tokenizer), 125k (dynamics model).
Distributed Training: Use torch.distributed or JAX for multi-GPU/TPU (256 TPUv5p for final model).
Optimizer: AdamW with cosine decay, weight decay 1e-4.
Evaluation: Metrics like FVD (Fréchet Video Distance), PSNR for controllability.


7. Inference and Controllability Module

Purpose: Generate interactive environments.
Sub-Modules:
Prompt Handling: Tokenize initial image/sketch/text (via optional text-to-image conditioner).
Action Input: User selects from 8 discrete actions.
Generation Loop: Autoregressive prediction + decoding.

Key Considerations for Code:
Make it interactive (e.g., via a loop or API).
Optional: Condition on text/sketches by adding embeddings.


General Implementation Tips

Framework: PyTorch (for flexibility) or JAX (for TPU efficiency, as used in the paper).
Libraries: transformers for base attention; custom for ST blocks. Use einops for tensor reshaping (patching).
Challenges: Memory management—test on short sequences first. Scale up with gradient checkpointing.
Resources: Refer to the paper's appendix for full hyperparams. Open-source approximations (e.g., on GitHub for VQ-VAE or MaskGIT) can serve as starters.
Ethical Notes: Train on public datasets to avoid biases; ensure generated environments are safe.

This covers the full architecture. If you need code snippets for specific modules (e.g., ST-Transformer block), provide more details!21 sources


## Steps to recreate 2

Below is a full architectural decomposition of what you need to build, from data → model → training → evaluation, with clear sub-modules, interfaces, and failure points. This is written the way you would design a serious research codebase.

1. High-level system blocks (mental map)
┌────────────┐
│  Dataset   │  (videos, actions, rewards)
└─────┬──────┘
      ↓
┌────────────┐
│ Tokenizer  │  (space + time)
└─────┬──────┘
      ↓
┌──────────────────────────┐
│  World Model (Core)      │
│  ├─ Neural Simulator     │
│  ├─ Dynamics Model       │
│  ├─ Reward Model         │
└─────┬────────────────────┘
      ↓
┌────────────┐
│  Planner   │  (MPC / imagination)
└─────┬──────┘
      ↓
┌────────────┐
│  Policy    │  (actor / executor)
└─────┬──────┘
      ↓
┌────────────┐
│ Environment│
└────────────┘

2. Data & Tokenization Layer (non-negotiable)
2.1 Video + State Dataset

You need:

RGB video (or latent video)

Actions

Rewards (or human preferences)

Episode boundaries

(state_t, action_t, reward_t, state_{t+1})


Common pitfall: ignoring alignment between frames and actions.

2.2 Spatiotemporal Tokenizer

This feeds your ST-Transformer.

Submodules:

Patch embedding (ViT-style or CNN)

Temporal stacking

Positional encodings

Spatial

Temporal (relative preferred)

tokens = tokenize(video)  # [B, T, S, D]

3. Core World Model (heart of the paper)

This is where most effort goes.

3.1 Neural Simulator (Imagination Engine)
Responsibilities

Predict future latent states

Support multi-step rollouts

Handle stochasticity

Submodules

Encoder (CNN / ViT)

Latent dynamics (RSSM-style or Transformer)

Decoder (optional, for reconstruction)

z_t = encoder(x_t)
z_{t+1} ~ p(z | z_t, a_t)


Losses

Reconstruction loss

KL loss (if stochastic)

Temporal consistency

3.2 ST-Transformer Backbone (memory-efficient core)

This replaces full attention everywhere.

Required components

Spatial attention block

Temporal attention block

Residual + LayerNorm

MLP

x = spatial_attn(x)
x = temporal_attn(x)
x = mlp(x)


Key constraint

No (T·S)² attention

Explicit factorization

3.3 Dynamics Model (Physics abstraction)

Used for short-term accuracy & control.

Responsibilities

Predict state deltas

Enforce smoothness / constraints

Δs = f_dyn(s_t, a_t)
s_{t+1} = s_t + Δs


Often trained jointly with the neural simulator.

3.4 Reward Model (Task semantics)
Responsibilities

Replace hand-crafted rewards

Encode human or task preference

Variants:

State-based reward

Trajectory-based reward

Preference model (pairwise ranking)

r_t = reward_model(z_t, a_t)

4. Training Objectives (this is where papers differ)

You will have multiple losses:

World model losses

Reconstruction

Dynamics prediction

KL regularization

Reward learning losses

MSE / BCE

Preference ranking loss

Consistency losses

Temporal smoothness

Latent stability

L = L_recon + λ1 L_dyn + λ2 L_reward + λ3 L_KL


Critical: loss balancing determines success or collapse.

5. Imagination & Planning Module

This is what makes it embodied intelligence.

Responsibilities

Roll out futures in latent space

Evaluate with reward model

Select best action sequence

Common choices:

MPC

Cross-Entropy Method (CEM)

Sampling-based planners

for a_seq in sample_actions():
    z_future = imagine(z_t, a_seq)
    score = reward(z_future)

6. Policy / Actor Module

Two common designs:

6.1 Actor-Critic (Dreamer-style)

Policy trained on imagined rollouts

Critic estimates value

6.2 Planner-only (no actor)

Execute MPC directly

Simpler but slower at inference

7. Environment Interface

Must support:

Reset / step

Deterministic replay

Video extraction

Often:

MuJoCo

Isaac Gym

Custom simulator

8. Evaluation & Diagnostics (often skipped, but crucial)
Metrics

Prediction error vs horizon

Reward accuracy

Planning success rate

Memory usage (very important for ST-Transformer)

Visualization

Video rollouts (GT vs imagined)

Attention maps (space & time)

9. Codebase structure (recommended)
src/
 ├─ data/
 ├─ tokenizer/
 ├─ models/
 │   ├─ st_transformer.py
 │   ├─ neural_simulator.py
 │   ├─ dynamics.py
 │   └─ reward.py
 ├─ training/
 ├─ planning/
 ├─ envs/
 └─ evaluation/

10. Failure modes to watch for
Failure	Cause
Memory OOM	Hidden quadratic attention
Drift in rollouts	Weak dynamics loss
Reward hacking	Overfit reward model
Blurry futures	Poor latent bottleneck
Training instability	Bad loss weighting
11. Minimal vs full reproduction
Minimal (weeks)

ST-Transformer

Neural simulator

Simple reward

MPC planner

Full paper-level (months)

Joint world + reward learning

Preference supervision

Long-horizon video rollouts

Multi-task evaluation

Final takeaway

Recreating this paper is not “implement a Transformer”—it is building a modular cognitive system combining perception, imagination, physics, value, and planning under tight memory constraints.



# references code and checkpoints to save the model, create new 

For the STTN: spatial transformer for video imprinting, use this open source code from here: 
https://github.com/researchmm/STTN

Let's add this as well for the Latent action model implementation albiet there is no associated VQ cookbook: https://github.com/MishaLaskin/vqvae


## NOTES

"k_q_size must be divisible by num_heads"
```
python scripts/train_tokenizer.py --dataset pong
```


Summary of fixes applied:
Fixed k_q_size divisibility: Changed encoder k_q_size from 64 to 60 (divisible by num_heads=6)
Fixed tensor view errors: Changed .view() to .reshape() for non-contiguous tensors
Fixed dimension mismatch: Added projection layers between encoder/decoder and quantizer (384→32→384)
Reduced memory usage: Batch size 2→1, decoder dimensions reduced to match encoder
Adjusted codebook: Reduced num_codes from 1024 to 512 to match scaled model

## LAM
```
python scripts/train_lam.py --dataset pong
```


Summary of fixes applied:
Fixed dimension mismatch: Added projection layers (pre_quantizer_proj and post_quantizer_proj) to map between encoder's d_model=512 and quantizer's latent_dim=32
Fixed width mismatch: Added output_padding=(0, 8) to ConvTranspose2d to handle non-divisible width (72 pixels)
Fixed autocast issue: Changed from binary_cross_entropy + sigmoid to binary_cross_entropy_with_logits (removed sigmoid from model output)
Removed all debug print statements