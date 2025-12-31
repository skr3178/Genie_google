# Dataset Usage Guide

## Available Datasets

The `data/` directory contains the following H5 dataset files:

1. **`pong_frames.h5`** - ~14MB (smallest, good for quick testing)
2. **`pole_position_frames.h5`** - ~17MB (small, good for quick testing)
3. **`picodoom_frames.h5`** - 59,785 frames
4. **`sonic_frames.h5`** - 41,242 frames
5. **`zelda_frames.h5`** - 72,410 frames
6. **`coinrun_frames.h5`** - 10,000,000 frames (largest, full training)

## Using a Specific Dataset

All training scripts now support a `--dataset` parameter to use a specific dataset file instead of loading all datasets.

### Training Video Tokenizer

```bash
# Use pong dataset only
python scripts/train_tokenizer.py --dataset pong

# Use pole_position dataset only
python scripts/train_tokenizer.py --dataset pole_position

# Use all datasets (default)
python scripts/train_tokenizer.py
```

### Training LAM

```bash
# Use pong dataset only
python scripts/train_lam.py --dataset pong

# Use pole_position dataset only
python scripts/train_lam.py --dataset pole_position

# Use all datasets (default)
python scripts/train_lam.py
```

### Training Dynamics Model

```bash
# Use pong dataset only
python scripts/train_dynamics.py \
    --dataset pong \
    --tokenizer_path checkpoints/tokenizer/checkpoint.pth \
    --lam_path checkpoints/lam/checkpoint.pth

# Use pole_position dataset only
python scripts/train_dynamics.py \
    --dataset pole_position \
    --tokenizer_path checkpoints/tokenizer/checkpoint.pth \
    --lam_path checkpoints/lam/checkpoint.pth

# Use all datasets (default)
python scripts/train_dynamics.py \
    --tokenizer_path checkpoints/tokenizer/checkpoint.pth \
    --lam_path checkpoints/lam/checkpoint.pth
```

## Dataset Name Matching

The `--dataset` parameter uses pattern matching:
- `--dataset pong` matches `pong_frames.h5`
- `--dataset pole_position` matches `pole_position_frames.h5`
- `--dataset coinrun` matches `coinrun_frames.h5`
- etc.

## Recommendations

### For Quick Testing/Development
Use smaller datasets for faster iteration:
```bash
--dataset pong          # Smallest (~14MB)
--dataset pole_position # Small (~17MB)
```

### For Full Training
Use all datasets or the large CoinRun dataset:
```bash
# Use all datasets (default)
python scripts/train_tokenizer.py

# Or use only CoinRun (largest dataset)
python scripts/train_tokenizer.py --dataset coinrun
```

## Example: Quick Training Workflow

1. **Test with small dataset:**
   ```bash
   python scripts/train_tokenizer.py --dataset pong
   python scripts/train_lam.py --dataset pong
   python scripts/train_dynamics.py --dataset pong \
       --tokenizer_path checkpoints/tokenizer/checkpoint.pth \
       --lam_path checkpoints/lam/checkpoint.pth
   ```

2. **Full training with all datasets:**
   ```bash
   python scripts/train_tokenizer.py
   python scripts/train_lam.py
   python scripts/train_dynamics.py \
       --tokenizer_path checkpoints/tokenizer/checkpoint.pth \
       --lam_path checkpoints/lam/checkpoint.pth
   ```
