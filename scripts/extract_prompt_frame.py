"""Extract a frame from the pong dataset to use as a prompt image for inference"""

import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def extract_frame(h5_path: str, frame_idx: int = 0, output_path: str = "prompt_frame.png"):
    """Extract a single frame from H5 dataset and save as PNG"""
    
    print(f"Loading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Try different possible keys
        if 'frames' in f:
            frames = f['frames']
        elif 'images' in f:
            frames = f['images']
        else:
            # Use first key
            first_key = list(f.keys())[0]
            frames = f[first_key]
        
        num_frames = frames.shape[0]
        print(f"  Found {num_frames} frames")
        
        if frame_idx >= num_frames:
            print(f"  Warning: frame_idx {frame_idx} >= {num_frames}, using frame 0")
            frame_idx = 0
        
        # Load frame
        frame = frames[frame_idx]
        
        # Convert to numpy if needed
        if isinstance(frame, h5py.Dataset):
            frame = frame[:]
        
        print(f"  Frame shape: {frame.shape}")
        
        # Handle different formats
        if frame.shape[0] == 3:  # (C, H, W)
            frame = frame.transpose(1, 2, 0)  # (H, W, C)
        elif len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)  # Convert to RGB
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Resize to training resolution (128, 72) if needed
        if frame.shape[:2] != (128, 72):
            print(f"  Resizing from {frame.shape[:2]} to (128, 72)")
            img = Image.fromarray(frame)
            img = img.resize((72, 128), Image.Resampling.LANCZOS)  # (W, H) for PIL
            frame = np.array(img)
        
        # Save as PNG
        img = Image.fromarray(frame)
        img.save(output_path)
        print(f"  ✓ Saved prompt frame to {output_path}")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]} (WxH)")
        print(f"  Use this image with: --prompt {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract a frame from pong dataset for inference")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to extract")
    parser.add_argument("--output", type=str, default="prompt_frame.png", help="Output image path")
    args = parser.parse_args()
    
    # Find pong dataset
    data_dir = Path(args.data_dir)
    pong_files = list(data_dir.glob("*pong*.h5"))
    
    if not pong_files:
        print(f"Error: No pong dataset found in {data_dir}")
        print(f"  Looking for files matching: *pong*.h5")
        return
    
    if len(pong_files) > 1:
        print(f"Found {len(pong_files)} pong files, using: {pong_files[0]}")
    
    extract_frame(str(pong_files[0]), args.frame_idx, args.output)


if __name__ == "__main__":
    main()
