"""Video Tokenizer (ST-ViViT) implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .st_transformer import STTransformer
from .vq import VectorQuantizer


class PatchEmbedding(nn.Module):
    """Convert video frames to patches"""
    
    def __init__(self, patch_size: int, in_channels: int = 3, d_model: int = 512):
        """
        Args:
            patch_size: Size of each patch
            in_channels: Number of input channels
            d_model: Model dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames of shape (B, T, C, H, W)
        
        Returns:
            Patches of shape (B, T, H_patches, W_patches, d_model)
        """
        B, T, C, H, W = x.shape
        
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Extract patches
        x = self.proj(x)  # (B*T, d_model, H_patches, W_patches)
        
        # Reshape: (B*T, d_model, H_patches, W_patches) -> (B, T, H_patches, W_patches, d_model)
        _, d, h_patches, w_patches = x.shape
        x = x.view(B, T, d, h_patches, w_patches)
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H_patches, W_patches, d_model)
        
        return x


class VideoTokenizerEncoder(nn.Module):
    """ST-Transformer encoder for video tokenizer"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        patch_size: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels=3, d_model=d_model)
        
        self.transformer = STTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            k_q_size=k_q_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            causal=False,  # Encoder doesn't need causal masking
            qk_normalization=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames of shape (B, T, C, H, W)
        
        Returns:
            Encoded features of shape (B, T, H_patches, W_patches, d_model)
        """
        # Patch embedding
        x = self.patch_embedding(x)  # (B, T, H_patches, W_patches, d_model)
        
        # ST-Transformer encoding
        x = self.transformer(x)
        
        return x


class VideoTokenizerDecoder(nn.Module):
    """ST-Transformer decoder for video tokenizer"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        patch_size: int = 4,
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
            causal=False,  # Decoder doesn't need causal masking for reconstruction
            qk_normalization=False,
        )
        
        # Output projection to reconstruct frames
        self.output_proj = nn.ConvTranspose2d(
            d_model,
            3,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features of shape (B, T, H_patches, W_patches, d_model)
        
        Returns:
            Reconstructed frames of shape (B, T, 3, H, W)
        """
        B, T, H_patches, W_patches, d = x.shape
        
        # ST-Transformer decoding
        x = self.transformer(x)
        
        # Reshape for convolution: (B, T, H_patches, W_patches, d_model) -> (B*T, d_model, H_patches, W_patches)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * T, d, H_patches, W_patches)
        
        # Reconstruct frames
        x = self.output_proj(x)  # (B*T, 3, H, W)
        
        # Reshape back: (B*T, 3, H, W) -> (B, T, 3, H, W)
        x = x.view(B, T, 3, x.shape[2], x.shape[3])
        
        return x


class VideoTokenizer(nn.Module):
    """Full Video Tokenizer with encoder, VQ, and decoder"""
    
    def __init__(
        self,
        encoder_config: Dict,
        decoder_config: Dict,
        codebook_config: Dict,
        patch_size: int = 4,
    ):
        """
        Args:
            encoder_config: Configuration for encoder
            decoder_config: Configuration for decoder
            codebook_config: Configuration for codebook
            patch_size: Patch size
        """
        super().__init__()
        self.patch_size = patch_size
        self.d_encoder = encoder_config['d_model']
        self.latent_dim = codebook_config['latent_dim']
        self.d_decoder = decoder_config['d_model']
        
        # Encoder
        self.encoder = VideoTokenizerEncoder(
            num_layers=encoder_config['num_layers'],
            d_model=encoder_config['d_model'],
            num_heads=encoder_config['num_heads'],
            k_q_size=encoder_config['k_q_size'],
            dim_feedforward=encoder_config['dim_feedforward'],
            patch_size=patch_size,
            dropout=encoder_config.get('dropout', 0.1),
            activation=encoder_config.get('activation', 'gelu'),
        )
        
        # Projection from encoder d_model to latent_dim for quantization
        self.pre_quantizer_proj = nn.Linear(
            encoder_config['d_model'],
            codebook_config['latent_dim']
        )
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_codes=codebook_config['num_codes'],
            latent_dim=codebook_config['latent_dim'],
            commitment_cost=codebook_config.get('commitment_cost', 0.25),
            decay=codebook_config.get('decay', 0.99),
            epsilon=codebook_config.get('epsilon', 1e-5),
        )
        
        # Projection from latent_dim to decoder d_model
        self.post_quantizer_proj = nn.Linear(
            codebook_config['latent_dim'],
            decoder_config['d_model']
        )
        
        # Decoder
        self.decoder = VideoTokenizerDecoder(
            num_layers=decoder_config['num_layers'],
            d_model=decoder_config['d_model'],
            num_heads=decoder_config['num_heads'],
            k_q_size=decoder_config['k_q_size'],
            dim_feedforward=decoder_config['dim_feedforward'],
            patch_size=patch_size,
            dropout=decoder_config.get('dropout', 0.1),
            activation=decoder_config.get('activation', 'gelu'),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: Input frames of shape (B, T, C, H, W)
        
        Returns:
            reconstructed: Reconstructed frames of shape (B, T, C, H, W)
            tokens: Quantized token indices of shape (B, T, H_patches, W_patches)
            loss_dict: Dictionary with loss components
        """
        # Encode
        encoded = self.encoder(x)  # (B, T, H_patches, W_patches, d_model)
        
        # Project to latent_dim for quantization
        B, T, H_patches, W_patches, d_encoder = encoded.shape
        encoded_flat = encoded.view(B * T * H_patches * W_patches, d_encoder)
        latent_flat = self.pre_quantizer_proj(encoded_flat)  # (N, latent_dim)
        
        # Quantize
        quantized_latent, tokens, vq_loss_dict = self.quantizer(latent_flat)
        
        # Project back to decoder d_model
        quantized = self.post_quantizer_proj(quantized_latent)  # (N, d_decoder)
        quantized = quantized.view(B, T, H_patches, W_patches, self.d_decoder)
        tokens = tokens.view(B, T, H_patches, W_patches)
        
        # Decode
        reconstructed = self.decoder(quantized)  # (B, T, C, H, W)
        
        # Apply sigmoid to output (as per paper)
        reconstructed = torch.sigmoid(reconstructed)
        
        return reconstructed, tokens, vq_loss_dict
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to tokens"""
        encoded = self.encoder(x)
        B, T, H_patches, W_patches, d_encoder = encoded.shape
        encoded_flat = encoded.view(B * T * H_patches * W_patches, d_encoder)
        latent_flat = self.pre_quantizer_proj(encoded_flat)
        _, tokens, _ = self.quantizer(latent_flat)
        tokens = tokens.view(B, T, H_patches, W_patches)
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to frames"""
        # Get quantized vectors from codebook
        B, T, H_patches, W_patches = tokens.shape
        tokens_flat = tokens.view(-1)
        quantized_latent = self.quantizer.codebook[tokens_flat]  # (B*T*H_patches*W_patches, latent_dim)
        # Project to decoder d_model
        quantized = self.post_quantizer_proj(quantized_latent)  # (B*T*H_patches*W_patches, d_decoder)
        quantized = quantized.view(B, T, H_patches, W_patches, -1)
        
        # Decode
        reconstructed = self.decoder(quantized)
        reconstructed = torch.sigmoid(reconstructed)
        return reconstructed
