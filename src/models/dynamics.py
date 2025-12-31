"""Dynamics Model with MaskGIT decoder and action embeddings"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .st_transformer import STTransformer


class TokenEmbedding(nn.Module):
    """Embed video tokens"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: Size of token vocabulary (from video tokenizer)
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Token indices of shape (B, T, H_patches, W_patches)
        
        Returns:
            Token embeddings of shape (B, T, H_patches, W_patches, embedding_dim)
        """
        return self.embedding(tokens)


class ActionEmbedding(nn.Module):
    """Embed discrete actions (spatial action maps from LAM)"""
    
    def __init__(self, num_actions: int, embedding_dim: int):
        """
        Args:
            num_actions: Number of discrete actions (8)
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: Action indices of shape (B, T, H_patches, W_patches) or (B, T) or (B,)
        
        Returns:
            Action embeddings of shape (B, T, H_patches, W_patches, embedding_dim) 
            or (B, T, embedding_dim) or (B, embedding_dim)
        """
        # Handle different input shapes
        if actions.dim() == 4:
            # Spatial action map: (B, T, H_patches, W_patches)
            B, T, H_patches, W_patches = actions.shape
            actions_flat = actions.view(B * T * H_patches * W_patches)
            emb_flat = self.embedding(actions_flat)  # (B*T*H*W, embedding_dim)
            return emb_flat.view(B, T, H_patches, W_patches, -1)
        elif actions.dim() == 2:
            # Temporal actions: (B, T)
            return self.embedding(actions)
        elif actions.dim() == 1:
            # Single action: (B,)
            return self.embedding(actions)
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")


class MaskGITDecoder(nn.Module):
    """
    MaskGIT Decoder: Bidirectional transformer for masked token prediction.
    
    Based on: https://github.com/google-research/maskgit
    
    Key features:
    - Bidirectional attention (not causal) during training
    - Learns to predict randomly masked tokens
    - Uses learnable mask token for masked positions
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        k_q_size: int,
        dim_feedforward: int,
        vocab_size: int,
        num_actions: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        qk_normalization: bool = True,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            d_model: Model dimension
            num_heads: Number of attention heads
            k_q_size: Dimension of key/query projections
            dim_feedforward: FFN dimension
            vocab_size: Size of token vocabulary
            num_actions: Number of discrete actions
            dropout: Dropout probability
            activation: Activation function
            qk_normalization: Whether to apply QK normalization
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.action_embedding = ActionEmbedding(num_actions, d_model)
        
        # Learnable mask token (as per MaskGIT)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model) * 0.02)
        
        # ST-Transformer with BIDIRECTIONAL attention (not causal)
        # MaskGIT uses bidirectional transformer to attend to all tokens
        self.transformer = STTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            k_q_size=k_q_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            causal=False,  # Bidirectional attention for MaskGIT (key difference!)
            qk_normalization=qk_normalization,
        )
        
        # Output head: predict token logits
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MaskGIT training/inference.
        
        Uses additive embeddings: token embeddings + action embeddings (as per Genie paper).
        
        Args:
            tokens: Token indices of shape (B, T, H_patches, W_patches) from video tokenizer
            actions: Action indices of shape (B, T, H_patches, W_patches) from LAM
                     (spatial action maps) or (B, T) or (B,) for single action per timestep
            mask: Binary mask of shape (B, T, H_patches, W_patches) where 1 = masked
        
        Returns:
            Token logits of shape (B, T, H_patches, W_patches, vocab_size)
        """
        B, T, H_patches, W_patches = tokens.shape
        
        # Embed tokens
        token_emb = self.token_embedding(tokens)  # (B, T, H_patches, W_patches, d_model)
        
        # Embed actions (spatial action maps from LAM)
        # Actions should be shape (B, T, H_patches, W_patches) from LAM
        # If single action per timestep, expand to spatial dimensions
        if actions.dim() == 1:
            # (B,) -> expand to (B, T, H_patches, W_patches)
            actions = actions.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, T, H_patches, W_patches)
        elif actions.dim() == 2:
            # (B, T) -> expand to (B, T, H_patches, W_patches)
            actions = actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_patches, W_patches)
        elif actions.dim() == 4:
            # Actions are already (B, T, H_lam, W_lam) but may have different spatial dims
            _, T_actions, H_lam, W_lam = actions.shape
            # Need to interpolate/upsample actions to match token spatial dimensions
            if H_lam != H_patches or W_lam != W_patches:
                # Reshape for interpolation: (B, T, H, W) -> (B*T, 1, H, W) for interpolate
                actions_reshaped = actions.view(B * T, 1, H_lam, W_lam).float()
                actions_interp = F.interpolate(
                    actions_reshaped, 
                    size=(H_patches, W_patches), 
                    mode='nearest'
                )
                actions = actions_interp.view(B, T, H_patches, W_patches).long()
        # If already (B, T, H_patches, W_patches), use as is
        
        action_emb = self.action_embedding(actions)  # (B, T, H_patches, W_patches, d_model)
        
        # Additive embeddings: Add action embeddings to token embeddings
        # Stopgrad on actions as per Genie paper (actions are from frozen LAM)
        action_emb = action_emb.detach()
        x = token_emb + action_emb
        
        # Apply mask if provided (for MaskGIT training)
        # Replace masked tokens with learnable mask token
        if mask is not None:
            # Convert bool mask to float if needed
            if mask.dtype == torch.bool:
                mask_float = mask.float()
            else:
                mask_float = mask
            mask_expanded = mask_float.unsqueeze(-1).expand_as(x)  # (B, T, H_patches, W_patches, d_model)
            # Expand mask token to match spatial dimensions
            mask_token_expanded = self.mask_token.expand(B, T, H_patches, W_patches, -1)
            # Replace masked positions with mask token
            x = x * (1 - mask_expanded) + mask_token_expanded * mask_expanded
        
        # ST-Transformer with bidirectional attention
        x = self.transformer(x)  # (B, T, H_patches, W_patches, d_model)
        
        # Predict token logits
        logits = self.output_head(x)  # (B, T, H_patches, W_patches, vocab_size)
        
        return logits


class DynamicsModel(nn.Module):
    """Full Dynamics Model with MaskGIT training"""
    
    def __init__(
        self,
        architecture_config: Dict,
        token_embedding_config: Dict,
        action_embedding_config: Dict,
    ):
        """
        Args:
            architecture_config: Configuration for transformer architecture
            token_embedding_config: Configuration for token embeddings
            action_embedding_config: Configuration for action embeddings
        """
        super().__init__()
        
        self.decoder = MaskGITDecoder(
            num_layers=architecture_config['num_layers'],
            d_model=architecture_config['d_model'],
            num_heads=architecture_config['num_heads'],
            k_q_size=architecture_config['k_q_size'],
            dim_feedforward=architecture_config['dim_feedforward'],
            vocab_size=token_embedding_config['vocab_size'],
            num_actions=action_embedding_config['num_actions'],
            dropout=architecture_config.get('dropout', 0.1),
            activation=architecture_config.get('activation', 'gelu'),
            qk_normalization=architecture_config.get('qk_normalization', True),
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: Token indices of shape (B, T, H_patches, W_patches)
            actions: Action indices of shape (B, T) or (B, T-1)
            mask: Optional mask for MaskGIT training
        
        Returns:
            Token logits of shape (B, T, H_patches, W_patches, vocab_size)
        """
        return self.decoder(tokens, actions, mask)
    
    def generate_mask(
        self,
        shape: Tuple[int, ...],
        mask_prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random mask for MaskGIT training.
        
        As per MaskGIT paper: randomly mask tokens during training.
        
        Args:
            shape: Shape of tokens (B, T, H_patches, W_patches)
            mask_prob: Probability of masking each token (typically 0.5-1.0)
            device: Device to create mask on
        
        Returns:
            Binary mask of same shape (1 = masked, 0 = not masked)
        """
        mask = torch.rand(shape, device=device) < mask_prob
        return mask.float()
    
    def iterative_refinement(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        steps: int = 8,
        temperature: float = 1.0,
        r: float = 0.5,
    ) -> torch.Tensor:
        """
        MaskGIT iterative refinement for inference.
        
        Based on: https://github.com/google-research/maskgit
        
        Process:
        1. Start with all tokens masked
        2. Iteratively predict and unmask tokens based on confidence
        3. Use scheduling function to determine how many tokens to unmask each step
        
        Args:
            tokens: Initial token predictions (can be random or from previous step)
            actions: Action indices of shape (B, T)
            steps: Number of refinement steps (typically 8-12)
            temperature: Temperature for sampling
            r: Scheduling ratio (determines unmasking schedule)
        
        Returns:
            Refined tokens of shape (B, T, H_patches, W_patches)
        """
        B, T, H_patches, W_patches = tokens.shape
        num_tokens = T * H_patches * W_patches
        
        # Ensure input tokens are Long type for embedding lookup
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        
        # Start with all tokens masked
        mask = torch.ones(B, T, H_patches, W_patches, device=tokens.device, dtype=torch.bool)
        
        # Initialize with random tokens (or use provided tokens)
        # Ensure tokens are Long type for embedding lookup
        current_tokens = tokens.clone().long()
        
        for step in range(steps):
            # Get logits for all tokens
            logits = self.forward(current_tokens, actions, mask)  # (B, T, H_patches, W_patches, vocab_size)
            
            # Compute confidence scores (max probability)
            probs = F.softmax(logits / temperature, dim=-1)
            confidence = probs.max(dim=-1)[0]  # (B, T, H_patches, W_patches)
            
            # Determine how many tokens to unmask this step
            # Scheduling: cosine schedule as per MaskGIT
            ratio = 1.0 - (step + 1) / steps
            num_to_unmask = int(num_tokens * (1 - ratio ** r))
            
            # Unmask tokens with highest confidence
            # Flatten for selection
            confidence_flat = confidence.view(B, -1)  # (B, num_tokens)
            mask_flat = mask.view(B, -1)  # (B, num_tokens)
            
            # Only consider currently masked tokens
            masked_confidence = confidence_flat * mask_flat.float()
            
            # Get top-k masked tokens to unmask
            _, top_indices = torch.topk(masked_confidence, num_to_unmask, dim=-1)
            
            # Create batch indices
            batch_indices = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, num_to_unmask)
            
            # Unmask selected tokens (set to False)
            mask_flat[batch_indices, top_indices] = False
            mask = mask_flat.view(B, T, H_patches, W_patches)
            
            # Sample new tokens for unmasked positions
            # Flatten to (B*T*H*W, vocab_size) for multinomial, then reshape back
            probs_flat = probs.view(-1, self.decoder.vocab_size)  # (B*T*H*W, vocab_size)
            sampled_flat = torch.multinomial(probs_flat, 1)  # (B*T*H*W, 1)
            sampled_tokens = sampled_flat.view(B, T, H_patches, W_patches).long()  # (B, T, H_patches, W_patches)
            
            # Update tokens: keep masked tokens, use sampled for unmasked positions
            # Use torch.where to preserve Long dtype
            current_tokens = torch.where(mask, current_tokens, sampled_tokens)
        
        return current_tokens