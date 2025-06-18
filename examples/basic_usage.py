#!/usr/bin/env python3
"""
Basic Usage Example

This script demonstrates basic usage of the multimodal Qwen3 extension
according to the original requirements.
"""

import torch
import logging
from multimodal_qwen3 import MultimodalQwen3Pipeline
from multimodal_qwen3.pipeline import create_example_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function demonstrating the core functionality."""
    logger.info("Starting multimodal Qwen3 basic usage example")
    
    # Configure modalities as per requirements
    modality_configs = {
        "m1": {"input_dim": 768, "hidden_dim": 4096},   # e.g., vision embeddings (CLIP)
        "m2": {"input_dim": 512, "hidden_dim": 4096},   # e.g., audio embeddings (Wav2Vec2)
    }
    
    logger.info(f"Using modality configs: {list(modality_configs.keys())}")
    
    # Initialize the pipeline
    pipeline = MultimodalQwen3Pipeline(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        modality_configs=modality_configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    logger.info("Pipeline initialized successfully")
    
    # Example 1: Text-only generation (baseline)
    logger.info("\n=== Example 1: Text-only Generation ===")
    text_only_input = "What is artificial intelligence?"
    response = pipeline.generate(text_only_input, max_new_tokens=50)
    print(f"Input: {text_only_input}")
    print(f"Response: {response}")
    
    # Example 2: Single modality input (as per requirements)
    logger.info("\n=== Example 2: Single Modality Input ===")
    single_modal_input = {
        "text": "Describe this scene.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [torch.randn(768), torch.randn(768)]  # Simulated vision embeddings
            }
        }
    }
    
    response = pipeline.generate(single_modal_input, max_new_tokens=100)
    print(f"Input: {single_modal_input['text']}")
    print(f"Modality: m1 (vision-like embeddings)")
    print(f"Response: {response}")
    
    # Example 3: Multiple modalities (as per requirements)
    logger.info("\n=== Example 3: Multiple Modalities ===")
    multi_modal_input = {
        "text": "Describe this scene.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [torch.randn(768)],        # Vision embedding
                "m2": [torch.randn(512), torch.randn(512)]  # Audio embeddings
            }
        }
    }
    
    response = pipeline.generate(multi_modal_input, max_new_tokens=100)
    print(f"Input: {multi_modal_input['text']}")
    print(f"Modalities: {list(multi_modal_input['multimodal_data']['multimodal_embeddings'].keys())}")
    print(f"Response: {response}")
    
    # Example 4: Demonstrate the exact format from requirements
    logger.info("\n=== Example 4: Requirements Format ===")
    requirements_input = {
        "text": "Describe this scene.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [torch.randn(768)],  # tensor() 
                "m2": [torch.randn(512)]   # tensor()
            }
        }
    }
    
    response = pipeline.generate(requirements_input, max_new_tokens=50)
    print(f"Input format (as per requirements):")
    print(f'  text: "{requirements_input["text"]}"')
    print(f'  multimodal_data: {{"multimodal_embeddings": {{"m1": [tensor()], "m2": [tensor()]}}}}')
    print(f"Output: \"{response}\"")
    
    # Example 5: Model information
    logger.info("\n=== Example 5: Model Information ===")
    model_info = pipeline.get_model_info()
    print("Model Configuration:")
    print(f"  Base model: {model_info['base_model']}")
    print(f"  Device: {model_info['device']}")
    print(f"  Hidden size: {model_info['hidden_size']}")
    print(f"  Supported modalities: {model_info['modalities']}")
    print(f"  Modality configurations:")
    for modality, config in model_info['modality_configs'].items():
        print(f"    {modality}: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main() 