"""
Multimodal Projection Layers

This module implements projection layers that align embeddings from different
modalities (vision, audio, etc.) to the text embedding space.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MultiModalProjector(nn.Module):
    """
    Multi-layer perceptron projector for aligning multimodal embeddings.
    
    This projector takes embeddings from different modalities and projects them
    to the text embedding dimension using modality-specific MLPs.
    """
    
    def __init__(
        self,
        text_hidden_size: int,
        modality_configs: Dict[str, Dict[str, Union[int, float]]],
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Initialize the multimodal projector.
        
        Args:
            text_hidden_size: Dimension of text embeddings (target dimension)
            modality_configs: Configuration for each modality containing:
                - "input_dim": Input embedding dimension for this modality
                - "hidden_dim": Hidden layer dimension (optional, defaults to text_hidden_size)
            activation: Activation function ("relu", "gelu", "tanh")
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.text_hidden_size = text_hidden_size
        self.modality_configs = modality_configs
        self.activation_fn = self._get_activation_fn(activation)
        
        # Create modality-specific projectors
        self.projectors = nn.ModuleDict()
        
        for modality_name, config in modality_configs.items():
            input_dim = config["input_dim"]
            hidden_dim = config.get("hidden_dim", text_hidden_size)
            
            # Build MLP layers
            layers = []
            
            # Input projection
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            
            # Output projection
            layers.append(nn.Linear(hidden_dim, text_hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(text_hidden_size))
            
            self.projectors[modality_name] = nn.Sequential(*layers)
            
        logger.info(f"Initialized MultiModalProjector with modalities: {list(modality_configs.keys())}")
    
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
            
        return activation_map[activation.lower()]
    
    def forward(
        self,
        multimodal_embeddings: Dict[str, List[torch.Tensor]],
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Project multimodal embeddings to text embedding space.
        
        Args:
            multimodal_embeddings: Dictionary mapping modality names to lists of tensors
            return_attention_mask: Whether to return attention masks for projected embeddings
            
        Returns:
            Dictionary containing:
                - "projected_embeddings": Concatenated projected embeddings [batch_size, seq_len, hidden_size]
                - "attention_mask": Attention mask for projected embeddings (if requested)
                - "modality_offsets": Start and end indices for each modality
        """
        device = next(self.parameters()).device
        batch_projected_embeddings = []
        batch_attention_masks = []
        batch_modality_offsets = []
        
        # Process each item in the batch
        if not multimodal_embeddings:
            return {
                "projected_embeddings": torch.empty(0, 0, self.text_hidden_size, device=device),
                "attention_mask": torch.empty(0, 0, device=device) if return_attention_mask else None,
                "modality_offsets": [],
            }
        
        # Get batch size from the first modality
        first_modality_embeddings = list(multimodal_embeddings.values())[0]
        if not first_modality_embeddings:
            return {
                "projected_embeddings": torch.empty(0, 0, self.text_hidden_size, device=device),
                "attention_mask": torch.empty(0, 0, device=device) if return_attention_mask else None,
                "modality_offsets": [],
            }
        
        batch_size = len(first_modality_embeddings)
        for batch_idx in range(batch_size):
            projected_embeddings = []
            attention_mask = []
            modality_offsets = {}
            current_offset = 0
            
            # Process each modality
            for modality_name, embedding_lists in multimodal_embeddings.items():
                if modality_name not in self.projectors:
                    logger.warning(f"No projector found for modality: {modality_name}")
                    continue
                
                # Get embeddings for current batch item
                modality_embeddings = embedding_lists[batch_idx] if batch_idx < len(embedding_lists) else []
                
                if not modality_embeddings:
                    continue
                
                # Stack embeddings if multiple tensors per modality
                if isinstance(modality_embeddings, list):
                    # Handle list of tensors
                    stacked_embeddings = torch.stack(modality_embeddings).to(device)
                else:
                    # Handle single tensor
                    stacked_embeddings = modality_embeddings.unsqueeze(0).to(device)
                
                # Project embeddings
                projector = self.projectors[modality_name]
                projected = projector(stacked_embeddings)  # [num_embeddings, hidden_size]
                
                # Store modality offset information
                start_idx = current_offset
                end_idx = current_offset + projected.shape[0]
                modality_offsets[modality_name] = (start_idx, end_idx)
                current_offset = end_idx
                
                # Collect projected embeddings and attention mask
                projected_embeddings.append(projected)
                attention_mask.extend([1] * projected.shape[0])
            
            if projected_embeddings:
                # Concatenate all modality embeddings
                batch_item_embeddings = torch.cat(projected_embeddings, dim=0)
                batch_projected_embeddings.append(batch_item_embeddings)
                batch_attention_masks.append(torch.tensor(attention_mask, device=device))
                batch_modality_offsets.append(modality_offsets)
            else:
                # No multimodal embeddings for this batch item
                batch_projected_embeddings.append(torch.empty(0, self.text_hidden_size, device=device))
                batch_attention_masks.append(torch.empty(0, device=device))
                batch_modality_offsets.append({})
        
        # Pad sequences to same length for batching
        if batch_projected_embeddings:
            max_seq_len = max(emb.shape[0] for emb in batch_projected_embeddings)
            
            padded_embeddings = []
            padded_masks = []
            
            for embeddings, mask in zip(batch_projected_embeddings, batch_attention_masks):
                if embeddings.shape[0] < max_seq_len:
                    # Pad embeddings
                    padding_size = max_seq_len - embeddings.shape[0]
                    padding = torch.zeros(padding_size, self.text_hidden_size, device=device)
                    padded_embeddings.append(torch.cat([embeddings, padding], dim=0))
                    
                    # Pad attention mask
                    mask_padding = torch.zeros(padding_size, device=device)
                    padded_masks.append(torch.cat([mask, mask_padding], dim=0))
                else:
                    padded_embeddings.append(embeddings)
                    padded_masks.append(mask)
            
            result = {
                "projected_embeddings": torch.stack(padded_embeddings),
                "modality_offsets": batch_modality_offsets,
            }
            
            if return_attention_mask:
                result["attention_mask"] = torch.stack(padded_masks)
                
        else:
            # No embeddings to process
            result = {
                "projected_embeddings": torch.empty(0, 0, self.text_hidden_size, device=device),
                "modality_offsets": [],
            }
            
            if return_attention_mask:
                result["attention_mask"] = torch.empty(0, 0, device=device)
        
        return result
    
    def get_modality_info(self) -> Dict[str, Dict[str, int]]:
        """Get information about supported modalities."""
        return {
            modality: {
                "input_dim": config["input_dim"],
                "output_dim": self.text_hidden_size,
            }
            for modality, config in self.modality_configs.items()
        }


class AdaptiveMultiModalProjector(MultiModalProjector):
    """
    Adaptive projector that can handle variable input dimensions.
    
    This projector can adapt to different input dimensions at runtime,
    useful for handling embeddings from different models or configurations.
    """
    
    def __init__(
        self,
        text_hidden_size: int,
        max_input_dim: int = 4096,
        adaptive_pooling: str = "mean",
        **kwargs
    ):
        """
        Initialize adaptive projector.
        
        Args:
            text_hidden_size: Target embedding dimension
            max_input_dim: Maximum expected input dimension
            adaptive_pooling: Pooling strategy for dimension adaptation ("mean", "max", "attention")
        """
        # Create a generic config for adaptive mode
        modality_configs = {
            "adaptive": {
                "input_dim": max_input_dim,
                "hidden_dim": text_hidden_size,
            }
        }
        
        super().__init__(text_hidden_size, modality_configs, **kwargs)
        
        self.max_input_dim = max_input_dim
        self.adaptive_pooling = adaptive_pooling
        
        # Add adaptive pooling layer if needed
        if adaptive_pooling == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=max_input_dim,
                num_heads=8,
                batch_first=True
            )
    
    def _adapt_input_dimension(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Adapt input embeddings to expected dimension."""
        current_dim = embeddings.shape[-1]
        
        if current_dim == self.max_input_dim:
            return embeddings
        elif current_dim < self.max_input_dim:
            # Pad with zeros
            padding_size = self.max_input_dim - current_dim
            padding = torch.zeros(*embeddings.shape[:-1], padding_size, device=embeddings.device)
            return torch.cat([embeddings, padding], dim=-1)
        else:
            # Reduce dimension using pooling
            if self.adaptive_pooling == "mean":
                # Reshape and average pool
                reshaped = embeddings.view(*embeddings.shape[:-1], -1, self.max_input_dim)
                return reshaped.mean(dim=-2)
            elif self.adaptive_pooling == "max":
                # Reshape and max pool
                reshaped = embeddings.view(*embeddings.shape[:-1], -1, self.max_input_dim)
                return reshaped.max(dim=-2)[0]
            elif self.adaptive_pooling == "attention":
                # Use attention pooling
                query = embeddings.mean(dim=-2, keepdim=True)  # Global average as query
                pooled, _ = self.attention_pool(query, embeddings, embeddings)
                return pooled.squeeze(-2)
            else:
                # Simple truncation
                return embeddings[..., :self.max_input_dim]
    
    def forward(
        self,
        multimodal_embeddings: Dict[str, List[torch.Tensor]],
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive dimension handling."""
        # Adapt all embeddings to expected dimension
        adapted_embeddings = {}
        
        for modality_name, embedding_lists in multimodal_embeddings.items():
            adapted_lists = []
            for embedding_list in embedding_lists:
                if isinstance(embedding_list, list):
                    adapted_list = [self._adapt_input_dimension(emb) for emb in embedding_list]
                else:
                    adapted_list = self._adapt_input_dimension(embedding_list)
                adapted_lists.append(adapted_list)
            adapted_embeddings[modality_name] = adapted_lists
        
        # Use "adaptive" modality for all inputs
        generic_embeddings = {"adaptive": []}
        for modality_embeddings in adapted_embeddings.values():
            generic_embeddings["adaptive"].extend(modality_embeddings)
        
        return super().forward(generic_embeddings, return_attention_mask) 