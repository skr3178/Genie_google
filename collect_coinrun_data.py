#!/usr/bin/env python3
"""
Collect CoinRun dataset following Genie paper specifications:
- CoinRun environment from Procgen benchmark
- "hard" mode
- Random policy with no action repeats
- Level seeds: 0 to 10,000
- 1,000 timesteps per level
- Total: 10M transitions
- Resolution: 160x90x3 (as per Genie paper)
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import gym
import signal
import sys
try:
    import procgen
    PROCGEN_AVAILABLE = True
except ImportError:
    PROCGEN_AVAILABLE = False
    print("Warning: procgen not available. Install with: pip install procgen")


def collect_coinrun_data(
    output_path: str,
    num_levels: int = 10000,
    timesteps_per_level: int = 1000,
    resolution: tuple = (160, 90),
    difficulty: str = "hard",
    start_seed: int = 0,
    verbose: bool = True,
    save_interval: int = 100  # Save every N levels
):
    """
    Collect CoinRun data following Genie paper specifications.
    
    Args:
        output_path: Path to save the H5 file
        num_levels: Number of levels to collect (default: 10,000)
        timesteps_per_level: Timesteps per level (default: 1,000)
        resolution: (width, height) resolution (default: (160, 90))
        difficulty: Difficulty mode (default: "hard")
        start_seed: Starting seed for levels (default: 0)
        num_workers: Number of parallel workers (default: 1)
        verbose: Print progress (default: True)
    """
    if not PROCGEN_AVAILABLE:
        raise ImportError(
            "procgen is required. Install with: pip install procgen"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total frames
    total_frames = num_levels * timesteps_per_level
    width, height = resolution
    
    # Check if output file exists and load existing data
    existing_frames = 0
    h5_file = None
    dataset = None
    resume_mode = False
    
    if output_path.exists():
        print(f"Found existing file: {output_path}")
        with h5py.File(output_path, 'r') as f:
            if 'frames' in f:
                existing_frames = f['frames'].shape[0]
                print(f"  Existing frames: {existing_frames:,}")
                if existing_frames >= total_frames:
                    print("  Collection already complete!")
                    return
                resume_mode = True
    
    # Calculate starting level based on existing frames
    start_level = 0
    frames_in_partial_level = 0
    if resume_mode and existing_frames > 0:
        start_level = existing_frames // timesteps_per_level
        frames_in_partial_level = existing_frames % timesteps_per_level
        print(f"Resuming from level {start_level} (frame {existing_frames:,})")
        if frames_in_partial_level > 0:
            print(f"  Note: Level {start_level} was partially completed ({frames_in_partial_level} frames)")
            print(f"  Will truncate to start of level {start_level} and restart from there")
            # Truncate to start of the incomplete level
            existing_frames = start_level * timesteps_per_level
            # Open file and truncate dataset
            h5_file = h5py.File(output_path, 'a')
            dataset = h5_file['frames']
            if dataset.shape[0] > existing_frames:
                dataset.resize((existing_frames, height, width, 3))
                h5_file.flush()
                print(f"  Truncated dataset to {existing_frames:,} frames")
        else:
            # Open in append mode
            h5_file = h5py.File(output_path, 'a')
            dataset = h5_file['frames']
    
    # Ensure dataset is properly sized
    if h5_file is None:
        # Will be created when first batch is saved
        pass
    else:
        # Resize dataset if needed for total frames
        if dataset.shape[0] < total_frames:
            dataset.resize((total_frames, height, width, 3))
    
    # Initialize environment using gym interface
    # Procgen uses "hard" difficulty by default, but we specify it explicitly
    # Procgen will cycle through levels automatically on each reset
    env = gym.make(
        "procgen-coinrun-v0",
        distribution_mode=difficulty,
        num_levels=num_levels,
        start_level=start_seed
    )
    
    # Get action space
    action_space = env.action_space
    num_actions = action_space.n
    
    # Storage for frames (batch for periodic saving)
    frames_batch = []
    current_level = start_level
    frames_collected = existing_frames
    last_action = None
    
    # Skip to the correct level if resuming
    if resume_mode and start_level > 0:
        print(f"Fast-forwarding to level {start_level}...")
        for _ in range(start_level):
            env.reset()
    
    if verbose:
        pbar = tqdm(total=total_frames, initial=frames_collected, desc="Collecting frames")
    
    # Helper function to save frames batch
    def save_frames_batch(batch, h5f, ds, start_idx):
        """Save a batch of frames to HDF5 file"""
        if not batch:
            return h5f, ds
        if h5f is None or ds is None:
            # Create new file
            h5f = h5py.File(output_path, 'w')
            ds = h5f.create_dataset('frames', 
                                   shape=(total_frames, height, width, 3),
                                   dtype=np.uint8,
                                   compression='gzip', 
                                   compression_opts=9,
                                   maxshape=(total_frames, height, width, 3))
        if len(batch) > 0:
            frames_array = np.array(batch, dtype=np.uint8)
            end_idx = start_idx + len(batch)
            ds[start_idx:end_idx] = frames_array
            h5f.flush()
        return h5f, ds
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal! Saving collected frames...")
        nonlocal h5_file, dataset
        h5_file, dataset = save_frames_batch(frames_batch, h5_file, dataset, frames_collected - len(frames_batch))
        if h5_file:
            if dataset and dataset.shape[0] > frames_collected:
                dataset.resize((frames_collected, height, width, 3))
            h5_file.close()
        env.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while current_level < num_levels:
            # Reset environment for new level
            # Procgen automatically cycles through levels on each reset
            obs = env.reset()
            
            # Resize observation to target resolution if needed
            # obs shape is [H, W, 3] for single env
            if obs.shape[:2] != (height, width):
                from PIL import Image
                img = Image.fromarray(obs)
                img = img.resize((width, height), Image.LANCZOS)
                obs = np.array(img)
            
            # Store initial frame
            frames_batch.append(obs)
            frames_collected += 1
            if verbose:
                pbar.update(1)
            
            # Collect timesteps for this level
            for step in range(1, timesteps_per_level):  # Start from 1 since we already stored first frame
                # Random policy with no action repeats
                # To avoid repeats, we track last action and ensure it's different
                if last_action is None:
                    action = action_space.sample()
                else:
                    # Sample until we get a different action
                    new_action = action_space.sample()
                    while new_action == last_action and num_actions > 1:
                        new_action = action_space.sample()
                    action = new_action
                
                last_action = action
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                # Resize if needed
                if obs.shape[:2] != (height, width):
                    from PIL import Image
                    img = Image.fromarray(obs)
                    img = img.resize((width, height), Image.LANCZOS)
                    obs = np.array(img)
                
                # Store frame
                frames_batch.append(obs)
                frames_collected += 1
                
                if verbose:
                    pbar.update(1)
                
                # If done early, break and move to next level
                if done:
                    break
            
            current_level += 1
            last_action = None  # Reset for next level
            
            # Periodic save to avoid data loss
            if current_level % save_interval == 0:
                print(f"\n[Saving checkpoint at level {current_level}...]")
                h5_file, dataset = save_frames_batch(frames_batch, h5_file, dataset, frames_collected - len(frames_batch))
                frames_batch = []  # Clear batch after saving
                print(f"[Checkpoint saved. Total frames: {frames_collected:,}]")
        
        # Save remaining frames
        if frames_batch:
            print(f"\nSaving final batch...")
            h5_file, dataset = save_frames_batch(frames_batch, h5_file, dataset, frames_collected - len(frames_batch))
        
        if verbose:
            pbar.close()
        
        # Finalize file
        if h5_file:
            # Trim dataset to actual size if needed
            if dataset.shape[0] > frames_collected:
                dataset.resize((frames_collected, height, width, 3))
            h5_file.close()
        
        print(f"\n✓ Collection complete!")
        print(f"  Total frames: {frames_collected:,}")
        print(f"  Saved to: {output_path}")
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024**3)
            print(f"  File size: {file_size:.2f} GB")
        
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("Saving collected frames...")
        if frames_batch:
            h5_file, dataset = save_frames_batch(frames_batch, h5_file, dataset, frames_collected - len(frames_batch))
        if h5_file:
            if dataset and dataset.shape[0] > frames_collected:
                dataset.resize((frames_collected, height, width, 3))
            h5_file.close()
        raise
    finally:
        if h5_file:
            h5_file.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect CoinRun dataset for Genie training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/coinrun_frames.h5",
        help="Output H5 file path (default: data/coinrun_frames.h5)"
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=10000,
        help="Number of levels to collect (default: 10000)"
    )
    parser.add_argument(
        "--timesteps-per-level",
        type=int,
        default=1000,
        help="Timesteps per level (default: 1000)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="160x90",
        help="Resolution as WIDTHxHEIGHT (default: 160x90)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=["easy", "hard", "extreme", "memory"],
        help="Difficulty mode (default: hard)"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed for levels (default: 0)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: collect only 10 levels with 100 timesteps each"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N levels (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Test mode
    if args.test:
        print("Running in TEST mode (10 levels, 100 timesteps each)")
        args.num_levels = 10
        args.timesteps_per_level = 100
    
    print(f"CoinRun Data Collection")
    print(f"========================")
    print(f"Output: {args.output}")
    print(f"Levels: {args.num_levels}")
    print(f"Timesteps per level: {args.timesteps_per_level}")
    print(f"Total frames: {args.num_levels * args.timesteps_per_level:,}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Start seed: {args.start_seed}")
    print()
    
    collect_coinrun_data(
        output_path=args.output,
        num_levels=args.num_levels,
        timesteps_per_level=args.timesteps_per_level,
        resolution=resolution,
        difficulty=args.difficulty,
        start_seed=args.start_seed,
        verbose=True,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()

