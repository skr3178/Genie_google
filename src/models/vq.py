"""Vector Quantization module with codebook and straight-through estimator"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """VQ-VAE style vector quantization with straight-through estimator"""
    
    def __init__(
        self,
        num_codes: int,
        latent_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            num_codes: Number of codes in the codebook
            latent_dim: Dimension of latent vectors
            commitment_cost: Weight for commitment loss
            decay: Exponential moving average decay for codebook updates
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(num_codes, latent_dim))
        self.codebook.data.normal_() / latent_dim
        
        # Exponential moving average for codebook updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_w', self.codebook.clone())
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Quantize input vectors using the codebook.
        
        Args:
            inputs: Input tensor of shape (..., latent_dim)
        
        Returns:
            quantized: Quantized vectors (same shape as inputs)
            encodings: Code indices (shape: inputs.shape[:-1])
            loss_dict: Dictionary with loss components
        """
        # Flatten input: (..., latent_dim) -> (N, latent_dim)
        input_shape = inputs.shape
        # #region agent log
        import json
        with open('/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"vq.py:56","message":"VQ forward - input shape","data":{"input_shape":list(input_shape),"latent_dim":self.latent_dim},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        flat_input = inputs.view(-1, self.latent_dim)
        num_vectors = flat_input.shape[0]
        # #region agent log
        with open('/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"vq.py:58","message":"After flatten - flat_input shape and num_vectors","data":{"flat_input_shape":list(flat_input.shape),"num_vectors":num_vectors,"input_numel":inputs.numel()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        # Calculate distances to codebook
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook ** 2, dim=1) -
            2 * torch.matmul(flat_input, self.codebook.t())
        )
        # #region agent log
        with open('/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"vq.py:65","message":"After distance calculation - distances shape","data":{"distances_shape":list(distances.shape),"codebook_shape":list(self.codebook.shape)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        # Find closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)  # (num_vectors,)
        # #region agent log
        with open('/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"vq.py:68","message":"After argmin - encoding_indices shape","data":{"encoding_indices_shape":list(encoding_indices.shape),"encoding_indices_numel":encoding_indices.numel(),"expected_numel":num_vectors},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        encodings = torch.zeros(
            num_vectors,
            self.num_codes,
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.codebook)
        quantized = quantized.view(input_shape)
        
        # Straight-through estimator: use quantized in forward, gradients flow to inputs
        quantized_st = inputs + (quantized - inputs).detach()
        
        # Losses
        e_latent_loss = F.mse_loss(quantized_st.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * e_latent_loss
        
        # Codebook loss
        codebook_loss = q_latent_loss
        
        # Total loss
        vq_loss = commitment_loss + codebook_loss
        
        # Update codebook using exponential moving average (during training)
        if self.training:
            # Update EMA cluster size
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )
            
            # Update EMA codebook
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_codes * self.epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Update codebook
            self.codebook.data.copy_(
                self.ema_w / cluster_size.unsqueeze(1)
            )
        
        loss_dict = {
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity': self._perplexity(encodings),
        }
        
        # Reshape encoding_indices to match input shape without the last dimension
        # If input was already 2D (N, latent_dim), return 1D (N,)
        # Otherwise, reshape to match all dimensions except the last
        if len(input_shape) == 2:
            tokens = encoding_indices  # Already 1D
        else:
            tokens_shape = input_shape[:-1]
            tokens = encoding_indices.reshape(tokens_shape)
        # #region agent log
        with open('/media/skr/storage/robot_world/Genie/Genie_SKR/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"vq.py:131","message":"Before return - tokens shape","data":{"tokens_shape":list(tokens.shape),"tokens_numel":tokens.numel(),"input_shape":list(input_shape),"input_shape_len":len(input_shape)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        return quantized_st, tokens, loss_dict
    
    def _perplexity(self, encodings: torch.Tensor) -> torch.Tensor:
        """Calculate perplexity of codebook usage"""
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity
    
    def get_codebook(self) -> torch.Tensor:
        """Get the current codebook"""
        return self.codebook
    
    def quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        """Quantize inputs without computing gradients"""
        with torch.no_grad():
            input_shape = inputs.shape
            flat_input = inputs.view(-1, self.latent_dim)
            
            distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                torch.sum(self.codebook ** 2, dim=1) -
                2 * torch.matmul(flat_input, self.codebook.t())
            )
            
            encoding_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[encoding_indices]
            quantized = quantized.view(input_shape)
            
            return quantized
