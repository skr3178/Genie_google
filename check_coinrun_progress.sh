#!/bin/bash
# Helper script to check CoinRun collection progress

OUTPUT_FILE="data/coinrun_frames.h5"
LOG_FILE="coinrun_collection.log"
PID_FILE="coinrun_collection.pid"

echo "CoinRun Data Collection Status"
echo "=============================="
echo ""

# Check if process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Process is running (PID: $PID)"
        echo ""
        
        # Check output file size
        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo "Output file: $OUTPUT_FILE"
            echo "File size: $FILE_SIZE"
            
            # Try to get frame count from H5 file
            if command -v python3 &> /dev/null; then
                FRAME_COUNT=$(python3 -c "import h5py; f = h5py.File('$OUTPUT_FILE', 'r'); print(f['frames'].shape[0])" 2>/dev/null)
                if [ ! -z "$FRAME_COUNT" ]; then
                    TOTAL_FRAMES=10000000
                    PERCENTAGE=$(echo "scale=2; $FRAME_COUNT * 100 / $TOTAL_FRAMES" | bc)
                    echo "Frames collected: $FRAME_COUNT / $TOTAL_FRAMES ($PERCENTAGE%)"
                fi
            fi
        else
            echo "Output file not created yet"
        fi
        
        echo ""
        echo "Last 10 lines of log:"
        echo "-------------------"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
    else
        echo "✗ Process is not running (stale PID file)"
        rm "$PID_FILE"
    fi
else
    echo "✗ No PID file found - collection may not be running"
fi

echo ""
echo "To see live updates:"
echo "  tail -f $LOG_FILE"

