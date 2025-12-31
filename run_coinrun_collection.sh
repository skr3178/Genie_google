#!/bin/bash
# Script to run CoinRun data collection in background using nohup
# This will collect 10M transitions (10,000 levels × 1,000 timesteps)

# Configuration
OUTPUT_FILE="data/coinrun_frames.h5"
LOG_FILE="coinrun_collection.log"
PID_FILE="coinrun_collection.pid"

# Create data directory if it doesn't exist
mkdir -p data

# Check if collection is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Collection is already running (PID: $PID)"
        echo "Check progress with: tail -f $LOG_FILE"
        exit 1
    else
        echo "Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Check if output file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Warning: Output file $OUTPUT_FILE already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm "$OUTPUT_FILE"
fi

echo "Starting CoinRun data collection..."
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
echo "This will collect 10,000,000 frames (10,000 levels × 1,000 timesteps)"
echo "Estimated time: 2-4 days"
echo "Estimated size: 100-200 GB (compressed)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo "  watch -n 60 'du -h $OUTPUT_FILE 2>/dev/null || echo \"File not created yet\"'"
echo ""

# Activate conda environment and run with nohup
# Using conda run to ensure the correct environment is used
nohup conda run -n robot_wm python collect_coinrun_data.py \
    --output "$OUTPUT_FILE" \
    --num-levels 10000 \
    --timesteps-per-level 1000 \
    --resolution 160x90 \
    --difficulty hard \
    --start-seed 0 \
    > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"

echo "Collection started in background (PID: $(cat $PID_FILE))"
echo "Log file: $LOG_FILE"
echo ""
echo "Useful commands:"
echo "  Check progress: tail -f $LOG_FILE"
echo "  Check file size: du -h $OUTPUT_FILE"
echo "  Check if running: ps -p \$(cat $PID_FILE)"
echo "  Stop collection: kill \$(cat $PID_FILE)"

