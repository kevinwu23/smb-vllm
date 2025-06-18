"""
Multimodal Pipeline

This module provides a high-level pipeline interface for easy usage of the
multimodal Qwen3 model.
"""

import torch
from typing import Dict, List, Union, Any, Optional
import logging
from pathlib import Path

from .model import MultimodalQwen3Model, MultimodalQwen3Config
from .processor import MultimodalProcessor

logger = logging.getLogger(__name__)


class MultimodalQwen3Pipeline:
    """
    High-level pipeline for multimodal text generation.
    
    This pipeline provides an easy-to-use interface for loading and using
    the multimodal Qwen3 model for text generation conditioned on arbitrary embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        modality_configs: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        **model_kwargs
    ):
        """
        Initialize the multimodal pipeline.
        
        Args:
            model_name: Base model name or path
            modality_configs: Configuration for multimodal projectors
            device: Device to load model on
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision  
            trust_remote_code: Whether to trust remote code
            **model_kwargs: Additional model arguments
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default modality configs if none provided
        if modality_configs is None:
            modality_configs = {
                "vision": {"input_dim": 768, "hidden_dim": 4096},
                "audio": {"input_dim": 512, "hidden_dim": 4096},
                "generic": {"input_dim": 1024, "hidden_dim": 4096},
            }
        
        # Create model configuration
        self.config = MultimodalQwen3Config(
            base_model_name=model_name,
            modality_configs=modality_configs,
            **model_kwargs
        )
        
        # Initialize model
        logger.info(f"Loading multimodal model: {model_name}")
        self.model = MultimodalQwen3Model(self.config)
        
        # Move to device
        if self.device != "auto":
            self.model = self.model.to(self.device)
        
        logger.info(f"Pipeline initialized with {len(modality_configs)} modalities on {self.device}")
    
    def __call__(
        self,
        inputs: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from multimodal inputs.
        
        Args:
            inputs: Input data (text string, input dict, or list of dicts)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text(s)
        """
        return self.generate(
            inputs=inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
    
    def generate(
        self,
        inputs: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        max_new_tokens: int = 256,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with multimodal conditioning.
        
        Args:
            inputs: Input data
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text(s)
        """
        # Handle different input formats
        if isinstance(inputs, str):
            # Simple text input
            formatted_inputs = {"text": inputs}
        elif isinstance(inputs, dict):
            # Single multimodal input
            formatted_inputs = inputs
        elif isinstance(inputs, list):
            # Batch of inputs - for now, process the first one
            # TODO: Implement proper batch processing
            formatted_inputs = inputs[0] if inputs else {"text": ""}
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Generate with the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=formatted_inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode outputs
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 2 and outputs.shape[0] > 1:
                # Multiple sequences
                results = []
                for i in range(outputs.shape[0]):
                    text = self.model.tokenizer.decode(outputs[i], skip_special_tokens=True)
                    # Remove the input text from the generated text
                    if "text" in formatted_inputs:
                        input_text = formatted_inputs["text"]
                        if text.startswith(input_text):
                            text = text[len(input_text):].strip()
                    results.append(text)
                return results
            else:
                # Single sequence
                text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the input text from the generated text
                if "text" in formatted_inputs:
                    input_text = formatted_inputs["text"]
                    if text.startswith(input_text):
                        text = text[len(input_text):].strip()
                return text
        else:
            return str(outputs)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Chat interface for conversational interaction.
        
        Args:
            messages: List of message dictionaries with role, content, and optional multimodal_data
            max_new_tokens: Maximum tokens for response
            **kwargs: Additional generation arguments
            
        Returns:
            Generated response
        """
        return self.model.chat(
            messages=messages,
            max_length=max_new_tokens,
            **kwargs
        )
    
    def process_embeddings(
        self,
        embeddings: Dict[str, List[torch.Tensor]],
        normalize: bool = True,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Process and validate multimodal embeddings.
        
        Args:
            embeddings: Dictionary of embeddings by modality
            normalize: Whether to normalize embeddings
            
        Returns:
            Processed embeddings
        """
        processed = {}
        
        for modality, embedding_list in embeddings.items():
            processed_list = []
            
            for emb in embedding_list:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                
                # Move to device
                emb = emb.to(self.device)
                
                # Normalize if requested
                if normalize and emb.numel() > 0:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                
                processed_list.append(emb)
            
            processed[modality] = processed_list
        
        return processed
    
    def add_modality(
        self,
        modality_name: str,
        input_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Add a new modality to the model.
        
        Args:
            modality_name: Name of the modality
            input_dim: Input dimension for the modality
            hidden_dim: Hidden dimension (defaults to text hidden size)
        """
        if hidden_dim is None:
            hidden_dim = self.model.hidden_size
        
        # Add to config
        self.config.modality_configs[modality_name] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
        }
        
        # Reinitialize projector with new modality
        if self.model.multimodal_projector is not None:
            from .projector import MultiModalProjector
            self.model.multimodal_projector = MultiModalProjector(
                text_hidden_size=self.model.hidden_size,
                modality_configs=self.config.modality_configs,
                activation=self.config.projector_hidden_act,
                dropout_rate=self.config.projector_dropout,
                use_layer_norm=self.config.use_layer_norm,
            ).to(self.device)
        
        logger.info(f"Added modality '{modality_name}' with input_dim={input_dim}")
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save the pipeline to a directory.
        
        Args:
            save_directory: Directory to save to
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_directory))
        
        logger.info(f"Pipeline saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "MultimodalQwen3Pipeline":
        """
        Load a pretrained pipeline.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            **kwargs: Additional arguments
            
        Returns:
            Loaded pipeline
        """
        # Load config
        config = MultimodalQwen3Config.from_pretrained(pretrained_model_name_or_path)
        
        # Create pipeline
        pipeline = cls(
            model_name=config.base_model_name,
            modality_configs=config.modality_configs,
            **kwargs
        )
        
        # Load model weights
        pipeline.model = MultimodalQwen3Model.from_pretrained(
            pretrained_model_name_or_path,
            config=config
        )
        
        logger.info(f"Loaded pipeline from {pretrained_model_name_or_path}")
        return pipeline
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        info = {
            "base_model": self.model_name,
            "device": self.device,
            "hidden_size": self.model.hidden_size,
            "vocab_size": self.model.vocab_size,
            "modalities": list(self.config.modality_configs.keys()),
            "modality_configs": self.config.modality_configs,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        return info
    
    def benchmark(
        self,
        test_inputs: List[Dict[str, Any]],
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """
        Benchmark the pipeline performance.
        
        Args:
            test_inputs: List of test inputs
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        results = {
            "avg_latency": 0.0,
            "avg_throughput": 0.0,
            "total_tokens": 0,
        }
        
        total_time = 0.0
        total_tokens = 0
        
        logger.info(f"Running benchmark with {len(test_inputs)} inputs for {num_runs} runs")
        
        for run in range(num_runs):
            start_time = time.time()
            
            for test_input in test_inputs:
                with torch.no_grad():
                    outputs = self.generate(test_input, max_new_tokens=50)
                    if isinstance(outputs, str):
                        total_tokens += len(self.model.tokenizer.encode(outputs))
                    elif isinstance(outputs, list):
                        for output in outputs:
                            total_tokens += len(self.model.tokenizer.encode(output))
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        results["avg_latency"] = total_time / (num_runs * len(test_inputs))
        results["avg_throughput"] = total_tokens / total_time
        results["total_tokens"] = total_tokens
        
        logger.info(f"Benchmark results: {results}")
        return results


def create_example_config() -> Dict[str, Dict[str, int]]:
    """
    Create example modality configurations.
    
    Returns:
        Example modality configurations
    """
    return {
        # Visual modality (e.g., CLIP embeddings)
        "vision": {
            "input_dim": 768,  # CLIP ViT-B/32 dimension
            "hidden_dim": 4096,
        },
        
        # Audio modality (e.g., Wav2Vec2 embeddings)
        "audio": {
            "input_dim": 512,  # Common audio embedding dimension
            "hidden_dim": 4096,
        },
        
        # Generic embedding modality
        "embedding": {
            "input_dim": 1024,  # Generic embedding dimension
            "hidden_dim": 4096,
        },
        
        # Code embeddings (e.g., CodeBERT)
        "code": {
            "input_dim": 768,  # CodeBERT dimension
            "hidden_dim": 4096,
        },
    }


def create_example_inputs() -> List[Dict[str, Any]]:
    """
    Create example inputs for testing.
    
    Returns:
        List of example input dictionaries
    """
    import torch
    
    return [
        {
            "text": "Describe this scene.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768), torch.randn(768)],
                    "audio": [torch.randn(512)],
                }
            }
        },
        {
            "text": "What is happening in this image?",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768)],
                }
            }
        },
        {
            "text": "Analyze this code and audio.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "code": [torch.randn(768)],
                    "audio": [torch.randn(512), torch.randn(512)],
                }
            }
        },
    ] 