"""Create MP4 comparison videos from original and reconstructed frames"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False


def load_frames_from_directory(frame_dir: Path, prefix: str = "original") -> list:
    """Load all frames from a directory, sorted by frame number"""
    frames = []
    frame_files = sorted(frame_dir.glob(f"{prefix}_*.png"))
    
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        frames.append(np.array(img))
    
    return frames


def create_side_by_side_video(original_frames: list, reconstructed_frames: list, 
                               output_path: Path, fps: float = 10.0):
    """Create a side-by-side comparison video"""
    
    if len(original_frames) != len(reconstructed_frames):
        raise ValueError(f"Mismatch in frame counts: {len(original_frames)} vs {len(reconstructed_frames)}")
    
    # Create side-by-side frames
    combined_frames = []
    for orig, recon in zip(original_frames, reconstructed_frames):
        # Ensure same height
        h = max(orig.shape[0], recon.shape[0])
        w_orig = orig.shape[1]
        w_recon = recon.shape[1]
        
        # Resize if needed to match height
        if orig.shape[0] != h:
            orig_img = Image.fromarray(orig)
            orig = np.array(orig_img.resize((w_orig, h), Image.Resampling.LANCZOS))
        if recon.shape[0] != h:
            recon_img = Image.fromarray(recon)
            recon = np.array(recon_img.resize((w_recon, h), Image.Resampling.LANCZOS))
        
        # Combine side by side
        combined = np.hstack([orig, recon])
        combined_frames.append(combined)
    
    # Save video
    video_saved = False
    
    if IMAGEIO_AVAILABLE:
        try:
            # Use imageio with H.264 codec for maximum compatibility
            imageio.mimwrite(
                str(output_path),
                combined_frames,
                fps=fps,
                codec='libx264',  # H.264 codec
                quality=8,  # High quality (0-10 scale, 10 is best)
                pixelformat='yuv420p'  # Ensures compatibility
            )
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"  ✓ Saved video to {output_path} using H.264 (libx264)")
                video_saved = True
            else:
                print(f"  Warning: Video file appears to be too small or missing")
        except Exception as e:
            print(f"  Warning: Error creating video with imageio: {e}")
            if CV2_AVAILABLE:
                print(f"  Falling back to OpenCV...")
    
    if not video_saved and CV2_AVAILABLE:
        # Fallback to OpenCV with H.264 settings
        try:
            H, W = combined_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264/AVC codec
            temp_path = str(output_path).replace('.mp4', '_temp.mp4')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            
            if out.isOpened():
                for frame in combined_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                # Check if file was written successfully
                if Path(temp_path).stat().st_size > 1000:
                    import shutil
                    shutil.move(temp_path, str(output_path))
                    print(f"  ✓ Saved video to {output_path} using OpenCV (H.264)")
                    video_saved = True
                else:
                    Path(temp_path).unlink()
        except Exception as e:
            print(f"  Warning: Error with OpenCV: {e}")
    
    if not video_saved:
        raise RuntimeError("Failed to create video with both imageio and OpenCV")
    
    return video_saved


def main():
    parser = argparse.ArgumentParser(description="Create MP4 comparison videos from tokenizer evaluation")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="evaluations/tokenizer",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all available)"
    )
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return
    
    print("=" * 70)
    print("Creating Comparison Videos")
    print("=" * 70)
    
    # Find all sample directories
    sample_dirs = sorted(eval_dir.glob("sample_*_original"))
    
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]
    
    if not sample_dirs:
        print(f"No sample directories found in {eval_dir}")
        return
    
    print(f"Found {len(sample_dirs)} samples to process\n")
    
    for sample_dir in sample_dirs:
        sample_num = sample_dir.name.split('_')[1]
        print(f"Processing sample {sample_num}...")
        
        # Get directories
        orig_dir = eval_dir / f"sample_{sample_num}_original"
        recon_dir = eval_dir / f"sample_{sample_num}_reconstructed"
        
        if not orig_dir.exists() or not recon_dir.exists():
            print(f"  ✗ Missing directories for sample {sample_num}")
            continue
        
        # Load frames
        try:
            original_frames = load_frames_from_directory(orig_dir, prefix="original")
            reconstructed_frames = load_frames_from_directory(recon_dir, prefix="reconstructed")
            
            if not original_frames or not reconstructed_frames:
                print(f"  ✗ No frames found for sample {sample_num}")
                continue
            
            print(f"  Loaded {len(original_frames)} frames")
            
            # Create output path
            output_path = eval_dir / f"comparison_sample_{sample_num}.mp4"
            
            # Create video
            create_side_by_side_video(
                original_frames,
                reconstructed_frames,
                output_path,
                fps=args.fps
            )
            
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_num}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Video creation complete!")
    print("=" * 70)
    print(f"Videos saved to: {eval_dir}/")
    print(f"  - comparison_sample_*.mp4")


if __name__ == "__main__":
    main()
