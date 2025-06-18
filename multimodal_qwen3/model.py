"""
Multimodal Qwen3 Model

This module implements the core multimodal model that extends Qwen3 to handle
multimodal inputs by integrating text and arbitrary embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .projector import MultiModalProjector
from .processor import MultimodalProcessor

logger = logging.getLogger(__name__)


class MultimodalQwen3Config(PretrainedConfig):
    """
    Configuration class for MultimodalQwen3Model.
    """
    
    model_type = "multimodal_qwen3"
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        modality_configs: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
        projector_type: str = "mlp",
        projector_hidden_act: str = "relu",
        projector_dropout: float = 0.1,
        use_layer_norm: bool = True,
        fusion_strategy: str = "early",
        multimodal_token: str = "<|multimodal|>",
        max_multimodal_length: int = 256,
        **kwargs
    ):
        """
        Initialize MultimodalQwen3Config.
        
        Args:
            base_model_name: Base Qwen model to extend
            modality_configs: Configuration for each modality
            projector_type: Type of projector ("mlp", "adaptive")
            projector_hidden_act: Activation function for projector
            projector_dropout: Dropout rate for projector
            use_layer_norm: Whether to use layer normalization
            fusion_strategy: Strategy for fusing modalities ("early", "late")
            multimodal_token: Special token for multimodal content
            max_multimodal_length: Maximum length for multimodal sequences
        """
        super().__init__(**kwargs)
        
        self.base_model_name = base_model_name
        self.modality_configs = modality_configs or {}
        self.projector_type = projector_type
        self.projector_hidden_act = projector_hidden_act
        self.projector_dropout = projector_dropout
        self.use_layer_norm = use_layer_norm
        self.fusion_strategy = fusion_strategy
        self.multimodal_token = multimodal_token
        self.max_multimodal_length = max_multimodal_length


class MultimodalQwen3Model(PreTrainedModel):
    """
    Multimodal extension of Qwen3 model supporting arbitrary embeddings.
    
    This model combines text and multimodal embeddings through projection layers
    and generates text conditioned on both modalities.
    """
    
    config_class = MultimodalQwen3Config
    
    def __init__(self, config: MultimodalQwen3Config):
        """
        Initialize the multimodal model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.config = config
        
        # Load base language model
        logger.info(f"Loading base model: {config.base_model_name}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Get model dimensions
        self.hidden_size = self.language_model.config.hidden_size
        self.vocab_size = self.language_model.config.vocab_size
        
        # Initialize tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.processor = MultimodalProcessor(
            tokenizer=self.tokenizer,
            multimodal_token=config.multimodal_token,
            device=self.language_model.device,
        )
        
        # Initialize multimodal projector
        if config.modality_configs:
            self.multimodal_projector = MultiModalProjector(
                text_hidden_size=self.hidden_size,
                modality_configs=config.modality_configs,
                activation=config.projector_hidden_act,
                dropout_rate=config.projector_dropout,
                use_layer_norm=config.use_layer_norm,
            )
        else:
            logger.warning("No modality configs provided, multimodal projector not initialized")
            self.multimodal_projector = None
        
        # Extend vocabulary if needed
        if config.multimodal_token not in self.tokenizer.get_vocab():
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Initialized MultimodalQwen3Model with {len(config.modality_configs)} modalities")
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings from the language model."""
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module):
        """Set input embeddings for the language model."""
        self.language_model.set_input_embeddings(value)
    
    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings from the language model."""
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings: nn.Module):
        """Set output embeddings for the language model."""
        self.language_model.set_output_embeddings(new_embeddings)
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        multimodal_embeddings: Optional[Dict[str, List[torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.
        
        This method is called by the generate() method to prepare inputs
        for each generation step.
        """
        # If we have past_key_values, we only need the last input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # Don't pass multimodal_embeddings for subsequent steps
            multimodal_embeddings = None
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "multimodal_embeddings": multimodal_embeddings,
        }
        
        return model_inputs
    
    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: Dict[str, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge multimodal embeddings with text embeddings.
        
        Args:
            input_ids: Input token IDs
            inputs_embeds: Text embeddings
            multimodal_embeddings: Multimodal embeddings to merge
            attention_mask: Attention mask for the sequence
            
        Returns:
            Tuple of (merged_embeddings, merged_attention_mask)
        """
        if not multimodal_embeddings or self.multimodal_projector is None:
            return inputs_embeds, attention_mask
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Project multimodal embeddings
        projected_data = self.multimodal_projector(
            multimodal_embeddings, 
            return_attention_mask=True
        )
        
        projected_embeddings = projected_data["projected_embeddings"]  # [batch_size, mm_seq_len, hidden_size]
        multimodal_attention_mask = projected_data["attention_mask"]   # [batch_size, mm_seq_len]
        
        if projected_embeddings.numel() == 0:
            # No multimodal embeddings to merge
            return inputs_embeds, attention_mask
        
        # Find multimodal token positions
        multimodal_token_id = self.processor.get_multimodal_token_id()
        multimodal_positions = (input_ids == multimodal_token_id).nonzero(as_tuple=True)
        
        if len(multimodal_positions[0]) == 0:
            # No multimodal tokens found, concatenate at the beginning
            merged_embeddings = torch.cat([projected_embeddings, inputs_embeds], dim=1)
            
            if attention_mask is not None:
                merged_attention_mask = torch.cat([multimodal_attention_mask, attention_mask], dim=1)
            else:
                merged_attention_mask = torch.cat([
                    multimodal_attention_mask,
                    torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
                ], dim=1)
        else:
            # Replace multimodal tokens with projected embeddings
            merged_embeddings = inputs_embeds.clone()
            merged_attention_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
            
            # Simple replacement strategy: replace first N multimodal tokens
            mm_seq_len = projected_embeddings.shape[1]
            for batch_idx in range(batch_size):
                batch_positions = multimodal_positions[1][multimodal_positions[0] == batch_idx]
                
                if len(batch_positions) > 0 and batch_idx < projected_embeddings.shape[0]:
                    # Replace tokens with projected embeddings
                    replace_count = min(len(batch_positions), mm_seq_len)
                    
                    for i in range(replace_count):
                        pos = batch_positions[i]
                        if pos < seq_len and i < projected_embeddings.shape[1]:
                            merged_embeddings[batch_idx, pos] = projected_embeddings[batch_idx, i]
        
        return merged_embeddings, merged_attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        multimodal_embeddings: Optional[Dict[str, List[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the multimodal model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            multimodal_embeddings: Multimodal embeddings to integrate
            labels: Labels for training
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either input_ids or inputs_embeds")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Merge multimodal embeddings
        if multimodal_embeddings is not None and past_key_values is None:
            # Only merge multimodal embeddings on the first forward pass
            inputs_embeds, attention_mask = self._merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, attention_mask
            )
            
            # Update input_ids to None since we're using inputs_embeds
            input_ids = None
        
        # Forward pass through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with multimodal conditioning.
        
        Args:
            inputs: Dictionary containing text and multimodal data
            max_length: Maximum total length
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs
        """
        if inputs is None:
            raise ValueError("inputs cannot be None for multimodal generation")
        
        # Process inputs
        if isinstance(inputs, dict):
            processed_inputs = self.processor.create_multimodal_input(
                text=inputs.get("text", ""),
                multimodal_data=inputs.get("multimodal_data", None),
            )
        else:
            processed_inputs = inputs
        
        # Set default values
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Prepare generation arguments
        generation_kwargs = {
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": num_beams,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            **kwargs
        }
        
        # Remove None values
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        # Generate
        with torch.no_grad():
            outputs = super().generate(
                input_ids=processed_inputs["input_ids"],
                attention_mask=processed_inputs["attention_mask"],
                multimodal_embeddings=processed_inputs.get("multimodal_embeddings"),
                **generation_kwargs
            )
        
        return outputs
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Chat interface for conversational multimodal interaction.
        
        Args:
            messages: List of message dictionaries
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response text
        """
        # Format conversation
        conversation_text = ""
        all_multimodal_data = {}
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            multimodal_data = message.get("multimodal_data", {})
            
            conversation_text += f"{role}: {content}\n"
            
            # Merge multimodal data
            if "multimodal_embeddings" in multimodal_data:
                for modality, embeddings in multimodal_data["multimodal_embeddings"].items():
                    if modality not in all_multimodal_data:
                        all_multimodal_data[modality] = []
                    all_multimodal_data[modality].extend(embeddings)
        
        conversation_text += "assistant: "
        
        # Generate response
        inputs = {
            "text": conversation_text,
            "multimodal_data": {"multimodal_embeddings": all_multimodal_data} if all_multimodal_data else None,
        }
        
        outputs = self.generate(
            inputs=inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant: " in response:
            response = response.split("assistant: ")[-1]
        
        return response.strip()
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        # Save the configuration
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        super().save_pretrained(save_directory, **kwargs)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[MultimodalQwen3Config] = None,
        **kwargs
    ):
        """Load a pretrained model."""
        if config is None:
            config = MultimodalQwen3Config.from_pretrained(pretrained_model_name_or_path)
        
        model = cls(config)
        
        # Load model weights if available
        try:
            model.load_state_dict(
                torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location="cpu"),
                strict=False
            )
            logger.info(f"Loaded model weights from {pretrained_model_name_or_path}")
        except FileNotFoundError:
            logger.warning(f"No model weights found at {pretrained_model_name_or_path}, using base model weights")
        
        return model


# Register the model for vLLM integration
def register_multimodal_qwen3():
    """Register the model with vLLM's model registry."""
    try:
        from vllm.model_executor.models import ModelRegistry
        ModelRegistry.register_model("multimodal_qwen3", MultimodalQwen3Model)
        logger.info("Registered MultimodalQwen3Model with vLLM")
    except ImportError:
        logger.warning("vLLM not available, skipping model registration")


# Auto-register on import
register_multimodal_qwen3() 