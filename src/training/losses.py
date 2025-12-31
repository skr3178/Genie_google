"""Loss functions for Genie models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute reconstruction loss.
    
    Args:
        pred: Predicted frames of shape (B, T, C, H, W) or (B, C, H, W)
        target: Target frames of same shape
        loss_type: Type of loss ("mse" or "bce")
    
    Returns:
        Reconstruction loss
    """
    # Clamp predictions to valid range to prevent NaN
    pred = torch.clamp(pred, min=1e-8, max=1.0 - 1e-8)
    
    if loss_type == "mse":
        loss = F.mse_loss(pred, target)
    elif loss_type == "bce":
        loss = F.binary_cross_entropy(pred, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Check for NaN and replace with a small value
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)
    
    return loss


def vq_loss(
    vq_loss_dict: Dict[str, torch.Tensor],
    commitment_weight: float = 0.25,
    codebook_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute VQ losses from loss dictionary.
    
    Args:
        vq_loss_dict: Dictionary with 'vq_loss', 'commitment_loss', 'codebook_loss'
        commitment_weight: Weight for commitment loss
        codebook_weight: Weight for codebook loss
    
    Returns:
        Dictionary with weighted losses
    """
    # Get losses with default values and ensure they're valid
    commitment_loss = vq_loss_dict.get('commitment_loss', torch.tensor(0.0))
    codebook_loss = vq_loss_dict.get('codebook_loss', torch.tensor(0.0))
    
    # Ensure tensors have proper device/dtype
    if not isinstance(commitment_loss, torch.Tensor):
        commitment_loss = torch.tensor(float(commitment_loss))
    if not isinstance(codebook_loss, torch.Tensor):
        codebook_loss = torch.tensor(float(codebook_loss))
    
    # Check for NaN/Inf and replace with small values
    if torch.isnan(commitment_loss) or torch.isinf(commitment_loss):
        device = commitment_loss.device if isinstance(commitment_loss, torch.Tensor) else 'cpu'
        dtype = commitment_loss.dtype if isinstance(commitment_loss, torch.Tensor) else torch.float32
        commitment_loss = torch.tensor(0.0, device=device, dtype=dtype)
    if torch.isnan(codebook_loss) or torch.isinf(codebook_loss):
        device = codebook_loss.device if isinstance(codebook_loss, torch.Tensor) else 'cpu'
        dtype = codebook_loss.dtype if isinstance(codebook_loss, torch.Tensor) else torch.float32
        codebook_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    total_loss = (
        commitment_weight * commitment_loss +
        codebook_weight * codebook_loss
    )
    
    # Final check on total loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
    
    return {
        'vq_loss': total_loss,
        'commitment_loss': commitment_loss,
        'codebook_loss': codebook_loss,
        'perplexity': vq_loss_dict.get('perplexity', torch.tensor(0.0)),
    }


def maskgit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute MaskGIT loss (cross-entropy on masked tokens).
    
    Args:
        logits: Predicted token logits of shape (B, T, H_patches, W_patches, vocab_size)
        targets: Target token indices of shape (B, T, H_patches, W_patches)
        mask: Binary mask of shape (B, T, H_patches, W_patches) (1 = masked)
    
    Returns:
        Dictionary with loss components
    """
    B, T, H_patches, W_patches, vocab_size = logits.shape
    
    # Flatten
    logits_flat = logits.view(B * T * H_patches * W_patches, vocab_size)
    targets_flat = targets.view(B * T * H_patches * W_patches)
    mask_flat = mask.view(B * T * H_patches * W_patches)
    
    # Compute cross-entropy only on masked tokens
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    masked_loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    
    # Accuracy on masked tokens
    pred = logits_flat.argmax(dim=-1)
    correct = (pred == targets_flat).float() * mask_flat
    accuracy = correct.sum() / (mask_flat.sum() + 1e-8)
    
    return {
        'maskgit_loss': masked_loss,
        'accuracy': accuracy,
    }


def lam_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vq_loss_dict: Dict[str, torch.Tensor],
    reconstruction_weight: float = 1.0,
    commitment_weight: float = 0.25,
    codebook_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute LAM loss (sigmoid cross-entropy on pixels + VQ losses).
    
    Args:
        pred: Predicted next frame logits of shape (B, C, H, W) (before sigmoid)
        target: Target next frame of shape (B, C, H, W) (normalized 0-1)
        vq_loss_dict: VQ loss dictionary
        reconstruction_weight: Weight for reconstruction loss
        commitment_weight: Weight for commitment loss
        codebook_weight: Weight for codebook loss
    
    Returns:
        Dictionary with loss components
    """
    # Binary cross-entropy with logits (combines sigmoid + BCE, numerically stable, autocast-safe)
    recon_loss = F.binary_cross_entropy_with_logits(pred, target)
    
    # VQ losses
    vq_losses = vq_loss(vq_loss_dict, commitment_weight, codebook_weight)
    
    # Total loss
    total_loss = (
        reconstruction_weight * recon_loss +
        vq_losses['vq_loss']
    )
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': recon_loss,
        **vq_losses,
    }
