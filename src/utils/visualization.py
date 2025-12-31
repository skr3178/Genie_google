"""Visualization utilities"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path


def visualize_video(
    video: torch.Tensor,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize video frames.
    
    Args:
        video: Video tensor of shape (T, C, H, W) or (B, T, C, H, W)
        save_path: Optional path to save visualization
    
    Returns:
        Video as numpy array
    """
    # Handle batch dimension
    if video.dim() == 5:
        video = video[0]  # Take first batch
    
    # Convert to numpy
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    
    # Ensure values are in [0, 1]
    if video.max() > 1.0:
        video = video / 255.0
    
    # Convert to uint8
    video = (video * 255).astype(np.uint8)
    
    # Transpose if needed: (T, C, H, W) -> (T, H, W, C)
    if video.shape[1] == 3:
        video = video.transpose(0, 2, 3, 1)
    
    if save_path:
        # Save as individual frames or video
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Implementation would save frames or use video writer
    
    return video


def save_video_grid(
    videos: torch.Tensor,
    save_path: str,
    nrow: int = 4,
):
    """
    Save a grid of videos.
    
    Args:
        videos: Video tensor of shape (B, T, C, H, W)
        save_path: Path to save grid
        nrow: Number of videos per row
    """
    # Placeholder implementation
    # Would create a grid visualization of multiple videos
    pass
