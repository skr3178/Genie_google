"""Positional embeddings for spatiotemporal transformers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SpatialPositionalEmbedding(nn.Module):
    """2D learnable positional embeddings for spatial positions"""
    
    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 128):
        """
        Args:
            d_model: Embedding dimension
            max_h: Maximum height
            max_w: Maximum width
        """
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # Learnable embeddings for each spatial position
        self.embedding = nn.Parameter(torch.randn(1, max_h, max_w, d_model) * 0.02)
    
    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Args:
            h: Height
            w: Width
        
        Returns:
            Positional embeddings of shape (1, h, w, d_model)
        """
        if h > self.max_h or w > self.max_w:
            # Interpolate if needed
            emb = F.interpolate(
                self.embedding.permute(0, 3, 1, 2),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            emb = self.embedding[:, :h, :w, :]
        
        return emb


class TemporalPositionalEmbedding(nn.Module):
    """1D learnable positional embeddings for time steps"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable embeddings for each time step
        self.embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
    
    def forward(self, t: int) -> torch.Tensor:
        """
        Args:
            t: Sequence length
        
        Returns:
            Positional embeddings of shape (t, d_model)
        """
        if t > self.max_len:
            # Use last embedding for longer sequences
            emb = self.embedding[-1:].repeat(t, 1)
        else:
            emb = self.embedding[:t]
        
        return emb


class CombinedPositionalEmbedding(nn.Module):
    """Combine spatial and temporal positional embeddings"""
    
    def __init__(
        self,
        d_model: int,
        max_h: int = 128,
        max_w: int = 128,
        max_t: int = 1000,
    ):
        """
        Args:
            d_model: Embedding dimension
            max_h: Maximum height
            max_w: Maximum width
            max_t: Maximum sequence length
        """
        super().__init__()
        self.spatial_emb = SpatialPositionalEmbedding(d_model, max_h, max_w)
        self.temporal_emb = TemporalPositionalEmbedding(d_model, max_t)
    
    def forward(self, t: int, h: int, w: int) -> torch.Tensor:
        """
        Args:
            t: Sequence length
            h: Height
            w: Width
        
        Returns:
            Combined positional embeddings of shape (t, h, w, d_model)
        """
        spatial = self.spatial_emb(h, w)  # (1, h, w, d_model)
        temporal = self.temporal_emb(t)  # (t, d_model)
        
        # Add spatial and temporal embeddings
        # Expand spatial: (1, h, w, d_model) -> (t, h, w, d_model)
        spatial = spatial.expand(t, -1, -1, -1)
        # Expand temporal: (t, d_model) -> (t, h, w, d_model)
        temporal = temporal.unsqueeze(1).unsqueeze(2).expand(-1, h, w, -1)
        
        # Combine
        combined = spatial + temporal
        
        return combined
