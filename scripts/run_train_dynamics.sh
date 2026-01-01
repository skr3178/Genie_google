#!/bin/bash
# Wrapper script for training dynamics model with proper output handling

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
echo "Training script started at $(date)"
echo "Working directory: $(pwd)"
echo "Conda env: robot_wm"
echo "========================================"

# Flush output
sync

# Run training with conda - use exec to ensure proper output handling
exec conda run -n robot_wm python -u scripts/train_dynamics.py \
    --lam_path /media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/lam/checkpoint_step_5000.pt \
    --tokenizer_path /media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/tokenizer/checkpoint_step_2319.pt \
    --data_dir data \
    --dataset pong \
    --max_steps 2000
