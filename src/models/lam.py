"""Latent Action Model (LAM) implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .st_transformer import STTransformer
from .vq import VectorQuantizer


class LAMEncoder(nn.Module):
    """ST-Transformer encoder for LAM (processes past frames + next frame)"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        patch_size: int = 16,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch embedding for raw pixels
        self.patch_embedding = nn.Conv2d(
            3,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        self.transformer = STTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            k_q_size=k_q_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            causal=True,  # Causal masking for autoregressive generation
            qk_normalization=False,
        )
        
        # Output projection to latent action dimension
        self.action_proj = nn.Linear(d_model, d_model)
    
    def forward(self, past_frames: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_frames: Past frames of shape (B, T, C, H, W)
            next_frame: Next frame of shape (B, C, H, W)
        
        Returns:
            Latent actions of shape (B, T, H_patches, W_patches, d_model)
        """
        B, T, C, H, W = past_frames.shape
        
        # Concatenate past frames and next frame
        # next_frame: (B, C, H, W) -> (B, 1, C, H, W)
        next_frame = next_frame.unsqueeze(1)
        frames = torch.cat([past_frames, next_frame], dim=1)  # (B, T+1, C, H, W)
        
        # Patch embedding
        frames_flat = frames.view(B * (T + 1), C, H, W)
        patches = self.patch_embedding(frames_flat)  # (B*(T+1), d_model, H_patches, W_patches)
        
        # Reshape: (B*(T+1), d_model, H_patches, W_patches) -> (B, T+1, H_patches, W_patches, d_model)
        _, d, h_patches, w_patches = patches.shape
        patches = patches.view(B, T + 1, d, h_patches, w_patches)
        patches = patches.permute(0, 1, 3, 4, 2)
        
        # ST-Transformer encoding
        encoded = self.transformer(patches)  # (B, T+1, H_patches, W_patches, d_model)
        
        # Extract latent actions (from transitions between frames)
        # Use difference between consecutive frames as latent actions
        latent_actions = encoded[:, 1:] - encoded[:, :-1]  # (B, T, H_patches, W_patches, d_model)
        
        # Project to action space
        latent_actions = self.action_proj(latent_actions)
        
        return latent_actions


class LAMDecoder(nn.Module):
    """ST-Transformer decoder for LAM (reconstructs next frame from history + action)"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        patch_size: int = 16,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.transformer = STTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            k_q_size=k_q_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            causal=True,  # Causal masking
            qk_normalization=False,
        )
        
        # Output projection to reconstruct frames
        # Calculate output_padding to handle non-divisible dimensions
        # For H=128, W=72 with patch_size=16:
        # H_patches = 128//16 = 8 (exact, no padding needed)
        # W_patches = 72//16 = 4, but 4*16=64, need 8 more pixels
        # output_padding formula: target_size - (input_size - 1) * stride - kernel_size + 2 * padding
        # For width: 72 - (4 - 1) * 16 - 16 + 0 = 72 - 48 - 16 = 8
        self.output_proj = nn.ConvTranspose2d(
            d_model,
            3,
            kernel_size=patch_size,
            stride=patch_size,
            output_padding=(0, 8),  # (H_padding, W_padding) - 8 pixels for width
        )
    
    def forward(
        self,
        history: torch.Tensor,
        quantized_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            history: History features of shape (B, T, H_patches, W_patches, d_model)
            quantized_action: Quantized action of shape (B, H_patches, W_patches, d_model)
        
        Returns:
            Reconstructed next frame of shape (B, 3, H, W)
        """
        B, T, H_patches, W_patches, d = history.shape
        
        # Add action to last frame in history
        last_frame = history[:, -1:]  # (B, 1, H_patches, W_patches, d_model)
        action_expanded = quantized_action.unsqueeze(1)  # (B, 1, H_patches, W_patches, d_model)
        next_frame_feat = last_frame + action_expanded
        
        # Concatenate with history
        x = torch.cat([history, next_frame_feat], dim=1)  # (B, T+1, H_patches, W_patches, d_model)
        
        # ST-Transformer decoding
        x = self.transformer(x)
        
        # Extract next frame
        next_frame = x[:, -1]  # (B, H_patches, W_patches, d_model)
        
        # Reshape for convolution: (B, H_patches, W_patches, d_model) -> (B, d_model, H_patches, W_patches)
        next_frame = next_frame.permute(0, 3, 1, 2)
        
        # Reconstruct frame
        next_frame = self.output_proj(next_frame)  # (B, 3, H, W)
        
        return next_frame


class LAM(nn.Module):
    """Full Latent Action Model with encoder, small codebook, and decoder"""
    
    def __init__(
        self,
        encoder_config: Dict,
        decoder_config: Dict,
        codebook_config: Dict,
        patch_size: int = 16,
    ):
        """
        Args:
            encoder_config: Configuration for encoder
            decoder_config: Configuration for decoder
            codebook_config: Configuration for codebook (8 codes)
            patch_size: Patch size
        """
        super().__init__()
        self.patch_size = patch_size
        
        # Encoder
        self.encoder = LAMEncoder(
            num_layers=encoder_config['num_layers'],
            d_model=encoder_config['d_model'],
            num_heads=encoder_config['num_heads'],
            k_q_size=encoder_config['k_q_size'],
            dim_feedforward=encoder_config['dim_feedforward'],
            patch_size=patch_size,
            dropout=encoder_config.get('dropout', 0.1),
            activation=encoder_config.get('activation', 'gelu'),
        )
        
        # Projection layers for quantizer (d_model -> latent_dim -> d_model)
        self.pre_quantizer_proj = nn.Linear(
            encoder_config['d_model'],
            codebook_config['latent_dim']
        )
        self.post_quantizer_proj = nn.Linear(
            codebook_config['latent_dim'],
            encoder_config['d_model']
        )
        
        # Vector Quantizer (small codebook: 8 codes)
        self.quantizer = VectorQuantizer(
            num_codes=codebook_config['num_codes'],
            latent_dim=codebook_config['latent_dim'],
            commitment_cost=codebook_config.get('commitment_cost', 0.25),
            decay=codebook_config.get('decay', 0.99),
            epsilon=codebook_config.get('epsilon', 1e-5),
        )
        
        # Decoder
        self.decoder = LAMDecoder(
            num_layers=decoder_config['num_layers'],
            d_model=decoder_config['d_model'],
            num_heads=decoder_config['num_heads'],
            k_q_size=decoder_config['k_q_size'],
            dim_feedforward=decoder_config['dim_feedforward'],
            patch_size=patch_size,
            dropout=decoder_config.get('dropout', 0.1),
            activation=decoder_config.get('activation', 'gelu'),
        )
    
    def forward(
        self,
        past_frames: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            past_frames: Past frames of shape (B, T, C, H, W)
            next_frame: Next frame of shape (B, C, H, W)
        
        Returns:
            reconstructed: Reconstructed next frame of shape (B, C, H, W)
            actions: Quantized action indices of shape (B, T, H_patches, W_patches)
            loss_dict: Dictionary with loss components
        """
        # Encode to latent actions
        latent_actions = self.encoder(past_frames, next_frame)  # (B, T, H_patches, W_patches, d_model)
        
        # Quantize actions
        B, T, H_patches, W_patches, d = latent_actions.shape
        # Project to quantizer's latent_dim before quantization
        latent_actions_flat = latent_actions.view(B * T * H_patches * W_patches, d)
        latent_actions_proj = self.pre_quantizer_proj(latent_actions_flat)  # (B*T*H*W, latent_dim)
        quantized, actions, vq_loss_dict = self.quantizer(latent_actions_proj)
        # Project back to d_model after quantization
        quantized = self.post_quantizer_proj(quantized)  # (B*T*H*W, d_model)
        quantized = quantized.view(B, T, H_patches, W_patches, d)
        actions = actions.view(B, T, H_patches, W_patches)
        
        # Get history features (from past frames)
        # Re-encode past frames to get history
        past_frames_reshaped = past_frames.view(B * T, *past_frames.shape[2:])
        past_patches = self.encoder.patch_embedding(past_frames_reshaped)
        past_patches_reshaped = past_patches.view(B, T, H_patches, W_patches, d)
        history = self.encoder.transformer(past_patches_reshaped)[:, :-1]  # Remove last frame
        
        # Decode using history and quantized action
        # Use the last quantized action
        last_action = quantized[:, -1]  # (B, H_patches, W_patches, d_model)
        reconstructed = self.decoder(history, last_action)  # (B, 3, H, W)
        
        # Note: sigmoid is applied in the loss function using binary_cross_entropy_with_logits
        # This is more numerically stable and works with autocast
        
        return reconstructed, actions, vq_loss_dict
