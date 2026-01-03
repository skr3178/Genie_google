"""Visual evaluation script for Latent Action Model (LAM)

This script creates visual comparisons showing:
1. Past frames (input context)
2. Ground truth next frame
3. Reconstructed next frame (from LAM)
4. Action visualizations (which actions were selected)
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

from src.models.lam import LAM
from src.utils.config import load_config


def load_lam_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load LAM model from checkpoint"""
    # Load to CPU first to avoid OOM, then move to device
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = load_config("configs/lam_config.yaml")
    
    model = LAM(
        encoder_config=config['model']['encoder'],
        decoder_config=config['model']['decoder'],
        codebook_config=config['model']['codebook'],
        patch_size=config['model']['patch_size'],
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model, config, checkpoint


def load_video_sequence(h5_path: str, start_idx: int, sequence_length: int, resolution: tuple = (128, 72)):
    """
    Load a sequence of frames from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Starting frame index
        sequence_length: Number of frames to load
        resolution: (H, W) target resolution
    
    Returns:
        Video tensor of shape (T, C, H, W) normalized to [0, 1]
    """
    with h5py.File(h5_path, 'r') as f:
        if 'frames' in f:
            frames = f['frames']
        else:
            first_key = list(f.keys())[0]
            frames = f[first_key]
        
        num_frames = frames.shape[0]
        
        # Check bounds
        if start_idx + sequence_length > num_frames:
            start_idx = max(0, num_frames - sequence_length)
        
        # Load frames
        frame_data = frames[start_idx:start_idx + sequence_length]  # (T, H, W, 3)
        
        # Convert to numpy if needed
        if isinstance(frame_data, h5py.Dataset):
            frame_data = frame_data[:]
        
        # Resize if needed
        if frame_data.shape[1:3] != resolution:
            resized_frames = []
            for frame in frame_data:
                img = Image.fromarray(frame)
                img = img.resize((resolution[1], resolution[0]), Image.Resampling.LANCZOS)  # (W, H)
                resized_frames.append(np.array(img))
            frame_data = np.array(resized_frames)
        
        # Convert to (T, C, H, W) and normalize to [0, 1]
        frame_data = frame_data.transpose(0, 3, 1, 2)  # (T, C, H, W)
        frame_data = frame_data.astype(np.float32) / 255.0
        
        return torch.from_numpy(frame_data).float()


def visualize_actions(actions: torch.Tensor, patch_size: int = 16, num_codes: int = 8, show_labels: bool = True):
    """
    Visualize action indices as a colored heatmap with action numbers.
    Redesigned for clarity and simplicity.
    
    Args:
        actions: Action tensor of shape (H_patches, W_patches) or (B, T, H_patches, W_patches)
        patch_size: Size of each patch
        num_codes: Number of action codes
        show_labels: Whether to show action numbers on patches
    
    Returns:
        Visualization as numpy array (H, W, 3) in [0, 255]
    """
    # Handle batch and time dimensions
    if actions.dim() == 4:
        # Take last time step and first batch
        actions = actions[0, -1]  # (H_patches, W_patches)
    elif actions.dim() == 3:
        # Take last time step
        actions = actions[-1]  # (H_patches, W_patches)
    
    actions_np = actions.cpu().numpy().astype(np.int32)
    H_patches, W_patches = actions_np.shape
    
    # Create distinct colors for each action code
    # Use a more distinct color palette
    if num_codes == 2:
        # For 2 actions: Red and Blue
        colors = np.array([
            [255, 0, 0],    # Red for action 1
            [0, 0, 255],    # Blue for action 2
        ], dtype=np.uint8)
    elif num_codes == 3:
        # For 3 actions: Red, Green, Blue
        colors = np.array([
            [255, 0, 0],    # Red for action 1
            [0, 255, 0],    # Green for action 2
            [0, 0, 255],    # Blue for action 3
        ], dtype=np.uint8)
    else:
        # For more actions, use a colormap but ensure distinct colors
        colors = plt.cm.Set3(np.linspace(0, 1, num_codes))[:, :3]  # Set3 has more distinct colors
        colors = (colors * 255).astype(np.uint8)
    
    # Create visualization
    H, W = H_patches * patch_size, W_patches * patch_size
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Fill patches with colors
    for h in range(H_patches):
        for w in range(W_patches):
            action_idx = actions_np[h, w]
            action_idx = action_idx % num_codes  # Ensure valid index
            color = colors[action_idx]
            h_start = h * patch_size
            h_end = h_start + patch_size
            w_start = w * patch_size
            w_end = w_start + patch_size
            vis[h_start:h_end, w_start:w_end] = color
    
    # Add text labels showing action numbers on each patch
    if show_labels:
        from PIL import Image, ImageDraw, ImageFont
        vis_pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis_pil)
        try:
            # Use larger, clearer font
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, patch_size // 2))
        except:
            label_font = ImageFont.load_default()
        
        for h in range(H_patches):
            for w in range(W_patches):
                action_idx = actions_np[h, w]
                action_idx = action_idx % num_codes
                action_num = action_idx + 1  # Display as 1-based (1, 2, 3, ...)
                
                # Center of the patch
                center_x = w * patch_size + patch_size // 2
                center_y = h * patch_size + patch_size // 2
                
                # Draw text with outline for visibility
                text = str(action_num)
                # Get text size for centering
                bbox = draw.textbbox((0, 0), text, font=label_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = center_x - text_w // 2
                text_y = center_y - text_h // 2
                
                # Draw thick outline (black) for better visibility
                for adj in [(-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
                           (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                           (0,-2), (0,-1), (0,1), (0,2),
                           (1,-2), (1,-1), (1,0), (1,1), (1,2),
                           (2,-2), (2,-1), (2,0), (2,1), (2,2)]:
                    draw.text((text_x + adj[0], text_y + adj[1]), text, fill=(0, 0, 0), font=label_font)
                
                # Draw main text (white)
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=label_font)
        
        vis = np.array(vis_pil)
    
    return vis


def add_gap(image: np.ndarray, gap_width: int = 5, gap_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Add a vertical gap (black stripe) to the right of an image.
    
    Args:
        image: Image array of shape (H, W, C)
        gap_width: Width of the gap in pixels
        gap_color: RGB color of the gap (default: black)
    
    Returns:
        Image with gap appended
    """
    H, W, C = image.shape
    gap = np.full((H, gap_width, C), gap_color, dtype=image.dtype)
    return np.hstack([image, gap])


def add_border(image: np.ndarray, border_width: int = 3, border_color: tuple = (255, 0, 0)) -> np.ndarray:
    """
    Add a colored border around an image.
    
    Args:
        image: Image array of shape (H, W, C)
        border_width: Width of the border in pixels
        border_color: RGB color of the border (default: red)
    
    Returns:
        Image with border added
    """
    H, W, C = image.shape
    # Create border color array
    border_color_arr = np.array(border_color, dtype=image.dtype)
    
    # Create new image with border
    new_H = H + 2 * border_width
    new_W = W + 2 * border_width
    bordered = np.full((new_H, new_W, C), border_color_arr, dtype=image.dtype)
    
    # Place original image in the center
    bordered[border_width:border_width+H, border_width:border_width+W] = image
    
    return bordered


def create_comparison_frame(
    past_frames: np.ndarray,
    ground_truth: np.ndarray,
    reconstructed: np.ndarray,
    actions_vis: np.ndarray,
    show_past: bool = True,
    show_actions: bool = True,
    gap_width: int = 5,
    border_width: int = 3,
    num_codes: int = 8,
    frame_diff: np.ndarray = None,
    action_stats: dict = None,
):
    """
    Create a single comparison frame showing all components.
    
    Args:
        past_frames: Past frames (T, H, W, 3) in [0, 255]
        ground_truth: Ground truth next frame (H, W, 3) in [0, 255]
        reconstructed: Reconstructed next frame (H, W, 3) in [0, 255]
        actions_vis: Action visualization (H, W, 3) in [0, 255]
        show_past: Whether to show past frames
        show_actions: Whether to show action visualization
    
    Returns:
        Combined frame as numpy array
    """
    H, W = ground_truth.shape[:2]
    
    # Create grid layout
    if show_past and show_actions:
        # 2x3 grid: past frames | GT | Recon | Actions
        # Show only 1-2 past frames to avoid confusion
        num_past = min(2, past_frames.shape[0])  # Show up to 2 past frames
        past_vis = past_frames[-num_past:] if num_past < past_frames.shape[0] else past_frames
        
        # Show each past frame as a separate section with its own border
        # Resize each past frame to match GT size (full size, not split)
        past_frames_bordered = []
        for pf in past_vis:
            # Resize to match GT dimensions
            if pf.shape[:2] != (H, W):
                img = Image.fromarray(pf)
                img = img.resize((W, H), Image.Resampling.LANCZOS)
                pf_resized = np.array(img)
            else:
                pf_resized = pf
            
            # Add border to each past frame separately
            pf_bordered = add_border(pf_resized, border_width=border_width)
            past_frames_bordered.append(pf_bordered)
        
        # Combine past frames with gaps between them
        if len(past_frames_bordered) == 1:
            past_combined = past_frames_bordered[0]
        else:
            # Combine with gaps between each frame
            past_combined = past_frames_bordered[0]
            for i in range(1, len(past_frames_bordered)):
                # Create gap
                gap = np.full((past_combined.shape[0], gap_width, 3), (0, 0, 0), dtype=past_combined.dtype)
                # Combine: existing + gap + next frame
                past_combined = np.hstack([past_combined, gap, past_frames_bordered[i]])
        
        past_bordered = past_combined
        gt_bordered = add_border(ground_truth, border_width=border_width)
        recon_bordered = add_border(reconstructed, border_width=border_width)
        
        # Add frame difference visualization if available
        if frame_diff is not None:
            # Resize frame_diff to match GT size
            if frame_diff.shape[:2] != ground_truth.shape[:2]:
                frame_diff_img = Image.fromarray(frame_diff)
                frame_diff_img = frame_diff_img.resize((ground_truth.shape[1], ground_truth.shape[0]), Image.Resampling.LANCZOS)
                frame_diff = np.array(frame_diff_img)
            diff_bordered = add_border(frame_diff, border_width=border_width)
            diff_with_gap = add_gap(diff_bordered, gap_width=gap_width)
            # Layout: [Past1 | Past2 | GT | Diff | Recon] (past frames already have gaps between them)
            past_with_gap = add_gap(past_bordered, gap_width=gap_width)
            gt_with_gap = add_gap(gt_bordered, gap_width=gap_width)
            top_row = np.hstack([past_with_gap, gt_with_gap, diff_with_gap, recon_bordered])
        else:
            # Combine: [Past1 | Past2 | GT | Recon] (past frames already have gaps between them)
            past_with_gap = add_gap(past_bordered, gap_width=gap_width)
            gt_with_gap = add_gap(gt_bordered, gap_width=gap_width)
            top_row = np.hstack([past_with_gap, gt_with_gap, recon_bordered])
        top_width = top_row.shape[1]
        
        # Resize actions to match top row width (accounting for borders)
        # Top row width includes borders, so we need to subtract 2*border_width from each side
        actions_target_width = top_width - 2 * border_width
        actions_resized = Image.fromarray(actions_vis)
        actions_resized = actions_resized.resize((actions_target_width, H), Image.Resampling.LANCZOS)
        actions_resized = np.array(actions_resized)
        
        # Add red border around actions (this will make it match top_width)
        actions_bordered = add_border(actions_resized, border_width=border_width)
        bottom_row = actions_bordered
        
        # Add text labels with smaller font and better positioning
        from PIL import ImageDraw, ImageFont
        top_pil = Image.fromarray(top_row)
        draw = ImageDraw.Draw(top_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
        except:
            try:
                # Try smaller default font
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate positions for labels (accounting for gaps and borders)
        past1_x = border_width + 3
        if len(past_frames_bordered) > 1:
            past2_x = past_frames_bordered[0].shape[1] + gap_width + border_width + 3
            # GT comes after all past frames
            gt_x = past_bordered.shape[1] + gap_width + border_width + 3
        else:
            past2_x = None
            gt_x = past_bordered.shape[1] + gap_width + border_width + 3
        
        # Account for GT + gap + border on diff/recon
        if frame_diff is not None:
            diff_x = gt_x + gt_bordered.shape[1] + gap_width + border_width + 3
            recon_x = diff_x + diff_bordered.shape[1] + gap_width + border_width + 3
        else:
            recon_x = gt_x + gt_bordered.shape[1] + gap_width + border_width + 3
        
        text_y = border_width + 3
        
        # Add labels with smaller font and outline for visibility
        def draw_text_with_outline(draw_obj, x, y, text, font_obj=None):
            """Draw text with black outline for better visibility"""
            if font_obj:
                # Draw outline (black)
                for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    draw_obj.text((x + adj[0], y + adj[1]), text, fill=(0, 0, 0), font=font_obj)
                # Draw main text (white)
                draw_obj.text((x, y), text, fill=(255, 255, 255), font=font_obj)
            else:
                # Draw outline (black)
                for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    draw_obj.text((x + adj[0], y + adj[1]), text, fill=(0, 0, 0))
                # Draw main text (white)
                draw_obj.text((x, y), text, fill=(255, 255, 255))
        
        # Label each past frame separately
        draw_text_with_outline(draw, past1_x, text_y, "Past1", font)
        if past2_x is not None:
            draw_text_with_outline(draw, past2_x, text_y, "Past2", font)
        draw_text_with_outline(draw, gt_x, text_y, "GT", font)
        if frame_diff is not None:
            draw_text_with_outline(draw, diff_x, text_y, "Diff", font)
        draw_text_with_outline(draw, recon_x, text_y, "Recon", font)
        
        bottom_pil = Image.fromarray(bottom_row)
        draw_bottom = ImageDraw.Draw(bottom_pil)
        # Add label for Actions section with outline for visibility
        actions_label_x = border_width + 3
        actions_label_y = border_width + 3
        if font:
            # Draw outline (black)
            for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                draw_bottom.text((actions_label_x + adj[0], actions_label_y + adj[1]), "Actions", fill=(0, 0, 0), font=font)
            # Draw main text (white)
            draw_bottom.text((actions_label_x, actions_label_y), "Actions", fill=(255, 255, 255), font=font)
        else:
            # Draw outline (black)
            for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                draw_bottom.text((actions_label_x + adj[0], actions_label_y + adj[1]), "Actions", fill=(0, 0, 0))
            # Draw main text (white)
            draw_bottom.text((actions_label_x, actions_label_y), "Actions", fill=(255, 255, 255))
        
        # Add clean legend showing action codes 1 to num_codes
        legend_start_x = actions_label_x + 60  # Start legend after "Actions" label
        legend_y = border_width + 3
        legend_box_size = 16  # Slightly larger boxes
        legend_spacing = 25  # More spacing between items
        
        # Get colors used in visualization (same as in visualize_actions)
        if num_codes == 2:
            colors = np.array([
                [255, 0, 0],    # Red
                [0, 0, 255],    # Blue
            ], dtype=np.uint8)
        elif num_codes == 3:
            colors = np.array([
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
            ], dtype=np.uint8)
        else:
            import matplotlib.pyplot as plt
            colors = plt.cm.Set3(np.linspace(0, 1, num_codes))[:, :3]
            colors = (colors * 255).astype(np.uint8)
        
        # Show all actions from 1 to num_codes
        for action_num in range(1, num_codes + 1):
            action_idx = action_num - 1  # Convert 1-based to 0-based
            color = tuple(colors[action_idx].tolist())
            
            # Draw color box with white border
            box_x = legend_start_x + (action_num - 1) * legend_spacing
            draw_bottom.rectangle(
                [box_x, legend_y, box_x + legend_box_size, legend_y + legend_box_size],
                fill=color,
                outline=(255, 255, 255),
                width=2
            )
            
            # Draw action number next to box with outline
            text_x = box_x + legend_box_size + 4
            text_y = legend_y
            if font:
                # Draw outline
                for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    draw_bottom.text((text_x + adj[0], text_y + adj[1]), str(action_num), fill=(0, 0, 0), font=font)
                # Draw main text
                draw_bottom.text((text_x, text_y), str(action_num), fill=(255, 255, 255), font=font)
            else:
                draw_bottom.text((text_x, text_y), str(action_num), fill=(255, 255, 255))
        
        # Add action statistics if available
        if action_stats:
            stats_x = legend_start_x + num_codes * legend_spacing + 20
            most_common = action_stats.get('most_common', 'N/A')
            stats_text = f"Most used: {most_common}"
            if font:
                # Draw outline
                for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    draw_bottom.text((stats_x + adj[0], legend_y + adj[1]), stats_text, fill=(0, 0, 0), font=font)
                # Draw main text
                draw_bottom.text((stats_x, legend_y), stats_text, fill=(255, 255, 255), font=font)
            else:
                draw_bottom.text((stats_x, legend_y), stats_text, fill=(255, 255, 255))
        
        # Add horizontal gap between top and bottom rows
        top_array = np.array(top_pil)
        gap_horizontal = np.full((gap_width, top_width, 3), (0, 0, 0), dtype=top_array.dtype)
        combined = np.vstack([top_array, gap_horizontal, np.array(bottom_pil)])
        
    elif show_past:
        # 1x3: Past | GT | Recon with gaps and borders
        num_past = min(2, past_frames.shape[0])
        past_vis = past_frames[-num_past:]
        past_resized = np.hstack([np.array(Image.fromarray(pf).resize((W // num_past, H), Image.Resampling.LANCZOS)) 
                                  for pf in past_vis])
        past_bordered = add_border(past_resized, border_width=border_width)
        gt_bordered = add_border(ground_truth, border_width=border_width)
        recon_bordered = add_border(reconstructed, border_width=border_width)
        past_with_gap = add_gap(past_bordered, gap_width=gap_width)
        gt_with_gap = add_gap(gt_bordered, gap_width=gap_width)
        combined = np.hstack([past_with_gap, gt_with_gap, recon_bordered])
    else:
        # Simple side-by-side: GT | Recon with gap and borders
        gt_bordered = add_border(ground_truth, border_width=border_width)
        recon_bordered = add_border(reconstructed, border_width=border_width)
        gt_with_gap = add_gap(gt_bordered, gap_width=gap_width)
        combined = np.hstack([gt_with_gap, recon_bordered])
    
    return combined


def create_comparison_video(
    past_frames_list: list,
    ground_truth_list: list,
    reconstructed_list: list,
    actions_list: list,
    output_path: Path,
    fps: float = 10.0,
    patch_size: int = 16,
    num_codes: int = 8,
    show_past: bool = True,
    show_actions: bool = True,
    gap_width: int = 5,
    border_width: int = 3,
):
    """Create a comparison video from multiple sequences"""
    
    all_frames = []
    
    for past_frames, gt, recon, actions in zip(past_frames_list, ground_truth_list, reconstructed_list, actions_list):
        # Convert to numpy if needed (keep as float for difference calculation)
        if isinstance(past_frames, torch.Tensor):
            past_frames_float = past_frames.cpu().numpy()
        else:
            past_frames_float = past_frames.astype(np.float32) if past_frames.dtype != np.float32 else past_frames
        
        if isinstance(gt, torch.Tensor):
            gt_float = gt.cpu().numpy()
        else:
            gt_float = gt.astype(np.float32) if gt.dtype != np.float32 else gt
            
        if isinstance(recon, torch.Tensor):
            recon_float = recon.cpu().numpy()
        else:
            recon_float = recon.astype(np.float32) if recon.dtype != np.float32 else recon
        
        # Normalize to [0, 1] if needed (for float values)
        if past_frames_float.max() > 1.0:
            past_frames_float = past_frames_float / 255.0
        if gt_float.max() > 1.0:
            gt_float = gt_float / 255.0
        if recon_float.max() > 1.0:
            recon_float = recon_float / 255.0
        
        # Normalize reconstruction to full [0, 1] range for better visibility
        # This helps if the model output is dimmer than expected
        recon_min = recon_float.min()
        recon_max = recon_float.max()
        if recon_max > recon_min:
            # Stretch to full range
            recon_float = (recon_float - recon_min) / (recon_max - recon_min)
            print(f"      Reconstructed range: [{recon_min:.3f}, {recon_max:.3f}] -> normalized to [0, 1]")
        else:
            print(f"      Warning: Reconstructed values are constant ({recon_min:.3f})")
        
        # Calculate frame difference BEFORE converting to uint8
        frame_diff = None
        if past_frames_float.ndim == 4 and past_frames_float.shape[0] >= 1:
            last_past = past_frames_float[-1]  # Last past frame
            if last_past.ndim == 3 and last_past.shape[0] == 3:  # (C, H, W)
                last_past = last_past.transpose(1, 2, 0)  # (H, W, C)
            if last_past.shape == gt_float.shape:
                frame_diff = np.abs(gt_float - last_past)
                # Normalize to [0, 255] for visualization
                if frame_diff.max() > 0:
                    frame_diff = (frame_diff / frame_diff.max() * 255).astype(np.uint8)
                else:
                    frame_diff = (frame_diff * 255).astype(np.uint8)
        
        # Now convert to uint8 for display
        past_frames = (past_frames_float * 255).astype(np.uint8)
        gt = (gt_float * 255).astype(np.uint8)
        recon = (recon_float * 255).astype(np.uint8)
        
        # Handle tensor shapes - past_frames should be (T, C, H, W) from model
        if past_frames.ndim == 4:
            if past_frames.shape[1] == 3:  # (T, C, H, W) - standard case
                past_frames = past_frames.transpose(0, 2, 3, 1)  # (T, H, W, C)
            elif past_frames.shape[0] == 3:  # (C, H, W) - single frame, add time dim
                past_frames = past_frames.transpose(1, 2, 0)[np.newaxis, :, :, :]  # (1, H, W, C)
            else:  # Already (T, H, W, C) or need transpose
                # Check if last dim is 3 (channels)
                if past_frames.shape[-1] == 3:
                    # Already in (T, H, W, C) format
                    pass
                else:
                    # Assume (T, C, H, W) and transpose
                    past_frames = past_frames.transpose(0, 2, 3, 1)  # (T, H, W, C)
        elif past_frames.ndim == 3:
            # Single frame (C, H, W) or (H, W, C)
            if past_frames.shape[0] == 3:  # (C, H, W)
                past_frames = past_frames.transpose(1, 2, 0)[np.newaxis, :, :, :]  # (1, H, W, C)
            else:  # (H, W, C)
                past_frames = past_frames[np.newaxis, :, :, :]  # (1, H, W, C)
        
        if gt.ndim == 3 and gt.shape[0] == 3:  # (C, H, W)
            gt = gt.transpose(1, 2, 0)  # (H, W, C)
        
        # Ensure reconstructed is a single frame (H, W, C)
        if recon.ndim == 4:
            # If it has 4 dims, take the first/last frame
            if recon.shape[0] == 3:  # (C, H, W, ?) - shouldn't happen
                recon = recon[:, :, :, 0].transpose(1, 2, 0)  # Take first and transpose
            elif recon.shape[-1] == 3:  # (T, H, W, C) - take last frame
                recon = recon[-1]  # (H, W, C)
            else:  # (C, H, W, T) - shouldn't happen
                recon = recon[:, :, :, -1].transpose(1, 2, 0)  # Take last and transpose
        elif recon.ndim == 3 and recon.shape[0] == 3:  # (C, H, W)
            recon = recon.transpose(1, 2, 0)  # (H, W, C)
        elif recon.ndim == 2:
            # If it's 2D, something is wrong - skip or handle
            print(f"  Warning: reconstructed has unexpected shape: {recon.shape}")
            continue
        
        # Final check: recon should be (H, W, C)
        if recon.ndim != 3 or recon.shape[2] != 3:
            print(f"  Warning: reconstructed final shape is unexpected: {recon.shape}, skipping")
            continue
        
        # Visualize actions
        actions_vis = visualize_actions(actions, patch_size=patch_size, num_codes=num_codes)
        
        # Calculate action statistics (from the last time step actions that correspond to the transition)
        actions_np = actions.cpu().numpy().astype(np.int32)
        if actions_np.ndim == 4:
            actions_flat = actions_np[0, -1].flatten()  # Last time step, first batch
        elif actions_np.ndim == 3:
            actions_flat = actions_np[-1].flatten()  # Last time step
        else:
            actions_flat = actions_np.flatten()
        
        # Count action usage
        from collections import Counter
        action_counts = Counter(actions_flat)
        if len(action_counts) > 0:
            most_common_action = action_counts.most_common(1)[0][0] + 1  # Convert 0-7 to 1-8
            action_stats = {
                'most_common': most_common_action,
                'counts': dict(action_counts)
            }
        else:
            action_stats = None
        
        # Create comparison frame
        comp_frame = create_comparison_frame(
            past_frames,
            gt,
            recon,
            actions_vis,
            show_past=show_past,
            show_actions=show_actions,
            gap_width=gap_width,
            border_width=border_width,
            num_codes=num_codes,
            frame_diff=frame_diff,
            action_stats=action_stats,
        )
        
        all_frames.append(comp_frame)
    
    # Save video
    video_saved = False
    
    if IMAGEIO_AVAILABLE:
        try:
            imageio.mimwrite(
                str(output_path),
                all_frames,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"  ✓ Saved video to {output_path} using imageio")
                video_saved = True
        except Exception as e:
            print(f"  Warning: Error with imageio: {e}")
    
    if not video_saved and CV2_AVAILABLE:
        try:
            H, W = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            temp_path = str(output_path).replace('.mp4', '_temp.mp4')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            
            if out.isOpened():
                for frame in all_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                if Path(temp_path).stat().st_size > 1000:
                    import shutil
                    shutil.move(temp_path, str(output_path))
                    print(f"  ✓ Saved video to {output_path} using OpenCV")
                    video_saved = True
                else:
                    Path(temp_path).unlink()
        except Exception as e:
            print(f"  Warning: Error with OpenCV: {e}")
    
    if not video_saved:
        raise RuntimeError("Failed to create video")
    
    return video_saved


def main():
    parser = argparse.ArgumentParser(description="Visual evaluation of LAM checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/skr/storage/robot_world/Genie/Genie_SKR/checkpoints/lam/run_20260102_113307/checkpoint_step_10000.pt",
        help="Path to LAM checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset HDF5 file (if None, uses random synthetic data)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lam_config.yaml",
        help="LAM config file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output video path"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting frame index in dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--no-past",
        dest="show_past",
        action="store_false",
        help="Don't show past frames in visualization"
    )
    parser.add_argument(
        "--no-actions",
        dest="show_actions",
        action="store_false",
        help="Don't show action visualization"
    )
    parser.add_argument(
        "--gap-width",
        type=int,
        default=5,
        help="Width of gap (in pixels) between sections (default: 5)"
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=3,
        help="Width of red border (in pixels) around each section (default: 3)"
    )
    parser.set_defaults(show_past=True, show_actions=True)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LAM Visual Evaluation")
    print("=" * 70)
    
    # Load model first to get the correct config from checkpoint
    print("\n1. Loading LAM model...")
    model, config, checkpoint = load_lam_from_checkpoint(args.checkpoint, args.device)
    
    # Get config values from the loaded model's config
    sequence_length = config['data']['sequence_length']
    resolution = tuple(config['data']['resolution'][:2])  # (H, W)
    patch_size = config['model']['patch_size']
    num_codes = config['model']['codebook']['num_codes']
    
    print(f"   Sequence length: {sequence_length}")
    print(f"   Resolution: {resolution}")
    print(f"   Patch size: {patch_size}")
    print(f"   Number of action codes: {num_codes}")
    step = checkpoint.get('global_step', 'unknown')
    print(f"   ✓ Model loaded (step: {step})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    print("\n3. Preparing data...")
    use_real_data = args.data_path is not None and Path(args.data_path).exists()
    
    if use_real_data:
        print(f"   Using real data from: {args.data_path}")
        with h5py.File(args.data_path, 'r') as f:
            if 'frames' in f:
                total_frames = f['frames'].shape[0]
            else:
                first_key = list(f.keys())[0]
                total_frames = f[first_key].shape[0]
        print(f"   Total frames in dataset: {total_frames}")
    else:
        print("   Using synthetic random data")
    
    # Collect samples
    print(f"\n3. Running inference on {args.num_samples} sample(s)...")
    past_frames_list = []
    ground_truth_list = []
    reconstructed_list = []
    actions_list = []
    
    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            print(f"   Sample {sample_idx + 1}/{args.num_samples}...")
            
            if use_real_data:
                # Load from dataset
                start_frame = args.start_idx + sample_idx * (sequence_length + 1)
                if start_frame + sequence_length >= total_frames:
                    start_frame = max(0, total_frames - sequence_length - 1)
                
                video = load_video_sequence(
                    args.data_path,
                    start_frame,
                    sequence_length + 1,  # Need one extra for next frame
                    resolution=resolution
                )  # (T+1, C, H, W)
                
                past_frames = video[:-1]  # (T, C, H, W)
                next_frame = video[-1]  # (C, H, W)
            else:
                # Generate random synthetic data
                C, H, W = 3, resolution[0], resolution[1]
                past_frames = torch.rand(sequence_length, C, H, W, device=args.device)
                next_frame = torch.rand(C, H, W, device=args.device)
            
            # Add batch dimension
            past_frames_batch = past_frames.unsqueeze(0).to(args.device)  # (1, T, C, H, W)
            next_frame_batch = next_frame.unsqueeze(0).to(args.device)  # (1, C, H, W)
            
            # Run LAM
            reconstructed, actions, vq_loss_dict = model(past_frames_batch, next_frame_batch)
            
            # Apply sigmoid to convert logits to [0, 1] range
            # (LAM outputs logits, sigmoid is applied in loss function)
            reconstructed = torch.sigmoid(reconstructed)
            
            # Remove batch dimension
            past_frames = past_frames_batch[0].cpu()  # (T, C, H, W)
            next_frame = next_frame_batch[0].cpu()  # (C, H, W)
            reconstructed = reconstructed[0].cpu()  # Should be (C, H, W) - single frame
            actions = actions[0].cpu()  # (T, H_patches, W_patches)
            
            # Store
            past_frames_list.append(past_frames)
            ground_truth_list.append(next_frame)
            reconstructed_list.append(reconstructed)
            actions_list.append(actions)
            
            # Print metrics
            mse = F.mse_loss(reconstructed, next_frame).item()
            print(f"      MSE: {mse:.6f}")
            print(f"      Actions shape: {actions.shape}")
            print(f"      Unique actions: {len(torch.unique(actions))}")
    
    # Create output path
    if args.output_path is None:
        output_dir = Path("evaluations/lam")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"visual_eval_step_{step}.mp4"
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    print(f"\n4. Creating comparison video...")
    create_comparison_video(
        past_frames_list,
        ground_truth_list,
        reconstructed_list,
        actions_list,
        output_path,
        fps=args.fps,
        patch_size=patch_size,
        num_codes=num_codes,
        show_past=args.show_past,
        show_actions=args.show_actions,
        gap_width=args.gap_width,
        border_width=args.border_width,
    )
    
    print("\n" + "=" * 70)
    print("Visual evaluation complete!")
    print("=" * 70)
    print(f"Comparison video saved to: {output_path}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - FPS: {args.fps}")


if __name__ == "__main__":
    main()
