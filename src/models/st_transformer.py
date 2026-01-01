"""ST-Transformer backbone with factored spatial/temporal attention"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Tuple
from .embeddings import CombinedPositionalEmbedding


class SpatialAttentionBlock(nn.Module):
    """Multi-head attention over spatial tokens (H×W) per frame"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dropout: float = 0.1,
        qk_normalization: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            k_q_size: Dimension of key/query projections
            dropout: Dropout probability
            qk_normalization: Whether to apply QK normalization
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_q_size = k_q_size
        self.head_dim = k_q_size // num_heads
        self.qk_normalization = qk_normalization
        
        assert k_q_size % num_heads == 0, "k_q_size must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, k_q_size)
        self.k_proj = nn.Linear(d_model, k_q_size)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, H, W, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, T, H, W, d_model)
        """
        B, T, H, W, d = x.shape
        
        # Reshape: (B, T, H, W, d_model) -> (B*T, H*W, d_model)
        x_flat = x.view(B * T, H * W, d)
        
        # Compute Q, K, V
        Q = self.q_proj(x_flat)  # (B*T, H*W, k_q_size)
        K = self.k_proj(x_flat)  # (B*T, H*W, k_q_size)
        V = self.v_proj(x_flat)  # (B*T, H*W, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(B * T, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, H * W, self.num_heads, d // self.num_heads).transpose(1, 2)
        
        # QK normalization (if enabled)
        if self.qk_normalization:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B*T, num_heads, H*W, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B * T, H * W, d)
        
        # Output projection
        out = self.out_proj(out)
        
        # Reshape back: (B*T, H*W, d_model) -> (B, T, H, W, d_model)
        out = out.view(B, T, H, W, d)
        
        return out


class TemporalAttentionBlock(nn.Module):
    """Multi-head attention over time (T) per spatial position with causal masking"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dropout: float = 0.1,
        causal: bool = True,
        qk_normalization: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            k_q_size: Dimension of key/query projections
            dropout: Dropout probability
            causal: Whether to use causal masking
            qk_normalization: Whether to apply QK normalization
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_q_size = k_q_size
        self.head_dim = k_q_size // num_heads
        self.causal = causal
        self.qk_normalization = qk_normalization
        
        assert k_q_size % num_heads == 0, "k_q_size must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, k_q_size)
        self.k_proj = nn.Linear(d_model, k_q_size)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, H, W, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, T, H, W, d_model)
        """
        B, T, H, W, d = x.shape
        
        # Reshape: (B, T, H, W, d_model) -> (B*H*W, T, d_model)
        # Process each spatial position independently
        x_reshaped = x.permute(0, 2, 3, 1, 4).contiguous()
        x_flat = x_reshaped.view(B * H * W, T, d)
        
        # Compute Q, K, V
        Q = self.q_proj(x_flat)  # (B*H*W, T, k_q_size)
        K = self.k_proj(x_flat)  # (B*H*W, T, k_q_size)
        V = self.v_proj(x_flat)  # (B*H*W, T, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * H * W, T, self.num_heads, d // self.num_heads).transpose(1, 2)
        
        # QK normalization (if enabled)
        if self.qk_normalization:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Causal mask (upper triangular)
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B*H*W, num_heads, T, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B * H * W, T, d)
        
        # Output projection
        out = self.out_proj(out)
        
        # Reshape back: (B*H*W, T, d_model) -> (B, T, H, W, d_model)
        out = out.view(B, H, W, T, d).permute(0, 3, 1, 2, 4)
        
        return out


class STTransformerBlock(nn.Module):
    """Combined spatial and temporal attention block with residual connections"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True,
        qk_normalization: bool = False,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            k_q_size: Dimension of key/query projections
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            activation: Activation function
            causal: Whether to use causal masking in temporal attention
            qk_normalization: Whether to apply QK normalization
        """
        super().__init__()
        self.spatial_attn = SpatialAttentionBlock(
            d_model, num_heads, k_q_size, dropout, qk_normalization
        )
        self.temporal_attn = TemporalAttentionBlock(
            d_model, num_heads, k_q_size, dropout, causal, qk_normalization
        )
        
        # Feed-forward network (applied once after both attention layers)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, H, W, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, T, H, W, d_model)
        """
        # Spatial attention with residual
        x = x + self.dropout(self.spatial_attn(self.norm1(x), mask))
        
        # Temporal attention with residual
        x = x + self.dropout(self.temporal_attn(self.norm2(x), mask))
        
        # FFN with residual
        # Reshape for FFN: (B, T, H, W, d_model) -> (B*T*H*W, d_model)
        B, T, H, W, d = x.shape
        x_flat = x.reshape(B * T * H * W, d)
        x_flat = x_flat + self.ffn(self.norm3(x_flat))
        x = x_flat.reshape(B, T, H, W, d)
        
        return x


class STTransformer(nn.Module):
    """Stacked ST-Transformer blocks with positional embeddings"""
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True,
        qk_normalization: bool = False,
        max_h: int = 128,
        max_w: int = 128,
        max_t: int = 1000,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            num_layers: Number of transformer blocks
            d_model: Model dimension
            num_heads: Number of attention heads
            k_q_size: Dimension of key/query projections
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            activation: Activation function
            causal: Whether to use causal masking
            qk_normalization: Whether to apply QK normalization
            max_h: Maximum height for positional embeddings
            max_w: Maximum width for positional embeddings
            max_t: Maximum sequence length for positional embeddings
            use_gradient_checkpointing: Whether to use gradient checkpointing to save memory
        """
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Positional embeddings
        self.pos_embedding = CombinedPositionalEmbedding(
            d_model, max_h, max_w, max_t
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            STTransformerBlock(
                d_model,
                num_heads,
                k_q_size,
                dim_feedforward,
                dropout,
                activation,
                causal,
                qk_normalization,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        self.use_gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.use_gradient_checkpointing = False
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, H, W, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, T, H, W, d_model)
        """
        B, T, H, W, d = x.shape
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(T, H, W)  # (T, H, W, d_model)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1, -1)
        x = x + self.dropout(pos_emb)
        
        # Apply transformer blocks with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            for layer in self.layers:
                x = checkpoint(layer, x, mask, use_reentrant=False)
        else:
            # Normal forward pass
            for layer in self.layers:
                x = layer(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        return x
