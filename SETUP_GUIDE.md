# Training Setup Guide for Rented Server

This guide helps you set up the Genie training environment on a rented server.

## Quick Start

1. **Run the setup script** (it will clone the repository automatically):
   ```bash
   # Clone and set up in default location ($HOME/Genie_google)
   bash <(curl -s https://raw.githubusercontent.com/skr3178/Genie_google/main/setup_training_env.sh)
   
   # OR download the script and run it
   wget https://raw.githubusercontent.com/skr3178/Genie_google/main/setup_training_env.sh
   chmod +x setup_training_env.sh
   ./setup_training_env.sh
   
   # OR specify a custom working directory
   ./setup_training_env.sh /path/to/your/workspace
   ```

The script will:
- Clone the repository from GitHub (if not already present)
- Install `uv` (fast Python package manager) if not present
- Create a Python 3.10 virtual environment using `uv`
- Install all required packages from `requirements.txt`
- Download the pong dataset
- Create necessary directories (checkpoints, logs)
- Verify the setup

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Clone repository
```bash
cd ~  # or your preferred directory
git clone https://github.com/skr3178/Genie_google.git
cd Genie_google
```

### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### 3. Create environment
```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
```

### 4. Install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download pong dataset
```bash
python download_dataset.py datasets --pattern "*pong*.h5" --out data
```

### 6. Create directories
```bash
mkdir -p data checkpoints/{tokenizer,lam,dynamics} logs/{tokenizer,lam,dynamics}
```

## Training Workflow

The training follows a 3-stage process:

### Stage 1: Train Video Tokenizer
```bash
source .venv/bin/activate
python scripts/train_tokenizer.py --dataset pong
```

Checkpoints will be saved to `checkpoints/tokenizer/`

### Stage 2: Train LAM (Latent Action Model)
```bash
python scripts/train_lam.py --dataset pong
```

Checkpoints will be saved to `checkpoints/lam/`

### Stage 3: Train Dynamics Model
```bash
python scripts/train_dynamics.py \
    --dataset pong \
    --tokenizer_path checkpoints/tokenizer/checkpoint_final.pt \
    --lam_path checkpoints/lam/checkpoint_final.pt
```

Checkpoints will be saved to `checkpoints/dynamics/`

## Configuration

Training parameters can be adjusted in the config files:
- `configs/tokenizer_config.yaml` - Video tokenizer settings
- `configs/lam_config.yaml` - LAM settings
- `configs/dynamics_config.yaml` - Dynamics model settings

Or override via command-line arguments:
```bash
python scripts/train_tokenizer.py \
    --dataset pong \
    --batch_size 2 \
    --max_steps 10000 \
    --save_every 1000
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Directory Structure

```
Genie_SKR/
├── data/                    # Dataset files (.h5)
├── checkpoints/
│   ├── tokenizer/           # Video tokenizer checkpoints
│   ├── lam/                 # LAM checkpoints
│   └── dynamics/            # Dynamics model checkpoints
├── logs/
│   ├── tokenizer/           # Training logs
│   ├── lam/
│   └── dynamics/
├── configs/                  # Configuration files
├── scripts/                  # Training scripts
└── src/                      # Source code
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config or via `--batch_size`
- Reduce `sequence_length` in config
- Enable gradient checkpointing (already enabled by default)

### Dataset Not Found
```bash
# Re-download the dataset
python download_dataset.py datasets --pattern "*pong*.h5" --out data
```

### Missing Packages
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Issues
```bash
# Recreate environment
rm -rf .venv
./setup_training_env.sh
```

## Server Recommendations

- **GPU**: NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for datasets and checkpoints
- **Python**: 3.10 (required)

## Next Steps

After training completes, you can:
1. Evaluate models using scripts in `scripts/`
2. Generate videos using `scripts/inference.py`
3. Compare checkpoints using `scripts/create_comparison_videos.py`

For more details, see the main `README.md`.
