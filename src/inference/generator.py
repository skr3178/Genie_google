"""Inference and generation module for Genie"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from pathlib import Path


class GenieGenerator:
    """Generator for end-to-end video generation"""
    
    def __init__(
        self,
        tokenizer,
        lam,
        dynamics_model,
        device: str = "cuda",
        maskgit_steps: int = 25,
        temperature: float = 2.0,
    ):
        """
        Args:
            tokenizer: Trained video tokenizer
            lam: Trained LAM model
            dynamics_model: Trained dynamics model
            device: Device to run on
            maskgit_steps: Number of MaskGIT steps for inference
            temperature: Temperature for sampling
        """
        self.tokenizer = tokenizer.to(device).eval()
        self.lam = lam.to(device).eval()
        self.dynamics_model = dynamics_model.to(device).eval()
        self.device = device
        self.maskgit_steps = maskgit_steps
        self.temperature = temperature
    
    @torch.no_grad()
    def generate(
        self,
        prompt_frame: torch.Tensor,
        actions: List[int],
        num_frames: int = 16,
    ) -> torch.Tensor:
        """
        Generate video from prompt frame and actions.
        
        Args:
            prompt_frame: Initial frame of shape (1, C, H, W) or (C, H, W)
            actions: List of action indices (8 discrete actions)
            num_frames: Number of frames to generate
        
        Returns:
            Generated video of shape (num_frames, C, H, W)
        """
        # Ensure prompt_frame is on device and has batch dimension
        if prompt_frame.dim() == 3:
            prompt_frame = prompt_frame.unsqueeze(0)
        prompt_frame = prompt_frame.to(self.device)
        
        # Normalize to [0, 1] if needed
        if prompt_frame.max() > 1.0:
            prompt_frame = prompt_frame / 255.0
        
        # Tokenize initial frame
        prompt_tokens = self.tokenizer.encode(prompt_frame.unsqueeze(1))  # (1, 1, H_patches, W_patches)
        
        # Initialize token sequence
        tokens = [prompt_tokens]
        
        # Generate frames autoregressively
        for t in range(len(actions)):
            if t >= num_frames - 1:
                break
            
            # Get action
            action = torch.tensor([actions[t]], device=self.device)
            
            # Predict next tokens using dynamics model with MaskGIT
            next_tokens = self._maskgit_sample(
                tokens[-1],
                action,
                self.maskgit_steps,
                self.temperature,
            )
            
            tokens.append(next_tokens)
        
        # Decode tokens to frames
        frames = []
        for token_seq in tokens:
            # Stack tokens if needed
            if isinstance(token_seq, list):
                token_tensor = torch.cat(token_seq, dim=1)
            else:
                token_tensor = token_seq
            
            # Decode
            frame = self.tokenizer.decode(token_tensor)  # (1, T, C, H, W)
            frames.append(frame.squeeze(0))  # (T, C, H, W)
        
        # Concatenate all frames
        video = torch.cat(frames, dim=0)  # (total_frames, C, H, W)
        
        return video[:num_frames]
    
    @torch.no_grad()
    def _maskgit_sample(
        self,
        tokens: torch.Tensor,
        action: torch.Tensor,
        steps: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Sample next tokens using MaskGIT iterative refinement.
        
        Uses the official MaskGIT iterative refinement process:
        https://github.com/google-research/maskgit
        
        Args:
            tokens: Current tokens of shape (B, T, H_patches, W_patches)
            action: Action index of shape (B,)
            steps: Number of MaskGIT refinement steps (typically 8-12)
            temperature: Temperature for sampling
        
        Returns:
            Next tokens of shape (B, 1, H_patches, W_patches)
        """
        B, T, H_patches, W_patches = tokens.shape
        
        # Initialize next frame tokens (random or from previous prediction)
        next_tokens = torch.randint(
            0,
            self.tokenizer.quantizer.num_codes,
            (B, 1, H_patches, W_patches),
            device=self.device,
        )
        
        # Use MaskGIT iterative refinement from the decoder
        # Concatenate tokens for processing
        all_tokens = torch.cat([tokens, next_tokens], dim=1)  # (B, T+1, H_patches, W_patches)
        
        # Expand action for all tokens
        actions_expanded = action.unsqueeze(1).expand(-1, T + 1)
        
        # Apply MaskGIT iterative refinement
        refined_tokens = self.dynamics_model.decoder.iterative_refinement(
            all_tokens,
            actions_expanded,
            steps=steps,
            temperature=temperature,
            r=0.5,  # Scheduling ratio
        )
        
        # Extract next frame tokens
        next_tokens = refined_tokens[:, -1:]  # (B, 1, H_patches, W_patches)
        
        return next_tokens
    
    def load_models(
        self,
        tokenizer_path: str,
        lam_path: str,
        dynamics_path: str,
    ):
        """Load trained models from checkpoints"""
        # Load tokenizer
        tokenizer_checkpoint = torch.load(tokenizer_path, map_location=self.device)
        self.tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
        
        # Load LAM
        lam_checkpoint = torch.load(lam_path, map_location=self.device)
        self.lam.load_state_dict(lam_checkpoint['model_state_dict'])
        
        # Load dynamics model
        dynamics_checkpoint = torch.load(dynamics_path, map_location=self.device)
        self.dynamics_model.load_state_dict(dynamics_checkpoint['model_state_dict'])
        
        print(f"Loaded models from {tokenizer_path}, {lam_path}, {dynamics_path}")
