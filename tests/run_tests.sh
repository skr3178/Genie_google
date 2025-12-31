#!/bin/bash
# Convenience script to run all tests with conda environment

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robot_wm

# Change to project root
cd "$(dirname "$0")/.."

# Run all tests
echo "Running all tests with conda environment: robot_wm"
echo "=========================================="
python tests/run_all_tests.py
