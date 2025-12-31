#!/bin/bash
# Quick training script to create checkpoints for testing dynamics model
# This trains tokenizer and LAM for 1000 steps each and saves checkpoints

echo "Training tokenizer for 1000 steps..."
python scripts/train_tokenizer.py --dataset pong --max_steps 1000 --save_every 1000

echo ""
echo "Training LAM for 1000 steps..."
python scripts/train_lam.py --dataset pong --max_steps 1000 --save_every 1000

echo ""
echo "Checkpoints created:"
echo "  - checkpoints/tokenizer/checkpoint.pth"
echo "  - checkpoints/lam/checkpoint.pth"
echo ""
echo "You can now train dynamics with:"
echo "python scripts/train_dynamics.py --dataset pong \\"
echo "    --tokenizer_path checkpoints/tokenizer/checkpoint.pth \\"
echo "    --lam_path checkpoints/lam/checkpoint.pth"
