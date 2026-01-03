#!/bin/bash
# Wrapper script for training LAM model with 100M parameter config

# Initialize conda (if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Set unbuffered Python output
export PYTHONUNBUFFERED=1

# Optional: Set CUDA memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to script directory
cd /media/skr/storage/robot_world/Genie/Genie_SKR || {
    echo "ERROR: Failed to change directory" >&2
    exit 1
}

# Print diagnostic info immediately
echo "========================================"
echo "Training LAM (100M config) started at $(date)"
echo "Working directory: $(pwd)"
echo "Config: configs/lam_config_100m.yaml"
echo "========================================"

# Flush output
sync

# Run training with conda - use exec to ensure proper output handling
exec conda run -n robot_wm python -u scripts/train_lam.py \
    --config configs/lam_config_100m.yaml \
    --data_dir data \
    "$@"
