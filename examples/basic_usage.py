#!/usr/bin/env python3
"""
Basic Usage Example

This script demonstrates basic usage of the multimodal Qwen3 extension.
"""

import torch
import logging
from multimodal_qwen3 import MultimodalQwen3Pipeline, create_example_config, create_example_inputs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting multimodal Qwen3 basic usage example")
    
    # Create modality configurations
    modality_configs = create_example_config()
    logger.info(f"Using modality configs: {list(modality_configs.keys())}")
    
    # Initialize the pipeline
    pipeline = MultimodalQwen3Pipeline(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # You can change this to a smaller model for testing
        modality_configs=modality_configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    logger.info("Pipeline initialized successfully")
    
    # Example 1: Simple text generation
    logger.info("\n=== Example 1: Simple Text Generation ===")
    simple_text = "What is artificial intelligence?"
    response = pipeline(simple_text, max_new_tokens=100)
    print(f"Input: {simple_text}")
    print(f"Response: {response}")
    
    # Example 2: Multimodal input with m1 embeddings
    logger.info("\n=== Example 2: Multimodal Input (m1 - e.g., Vision) ===")
    multimodal_input = {
        "text": "Describe this scene.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [
                    torch.randn(768),  # Simulated visual embedding 1
                    torch.randn(768),  # Simulated visual embedding 2
                ]
            }
        }
    }
    
    response = pipeline(multimodal_input, max_new_tokens=150, temperature=0.7)
    print(f"Input: {multimodal_input['text']}")
    print(f"m1 embeddings: {len(multimodal_input['multimodal_data']['multimodal_embeddings']['m1'])} embeddings")
    print(f"Response: {response}")
    
    # Example 3: Multiple modalities
    logger.info("\n=== Example 3: Multiple Modalities ===")
    multi_modal_input = {
        "text": "Analyze the visual scene and audio content.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [torch.randn(768)],  # Visual embedding
                "m2": [torch.randn(512), torch.randn(512)],  # Audio embeddings
            }
        }
    }
    
    response = pipeline(multi_modal_input, max_new_tokens=200, temperature=0.8)
    print(f"Input: {multi_modal_input['text']}")
    print(f"Modalities: {list(multi_modal_input['multimodal_data']['multimodal_embeddings'].keys())}")
    print(f"Response: {response}")
    
    # Example 4: Chat interface
    logger.info("\n=== Example 4: Chat Interface ===")
    messages = [
        {
            "role": "user",
            "content": "I have an image and some audio. Can you help me understand them?",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "m1": [torch.randn(768)],
                    "m2": [torch.randn(512)],
                }
            }
        }
    ]
    
    chat_response = pipeline.chat(messages, max_new_tokens=100)
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {chat_response}")
    
    # Example 5: Adding a new modality dynamically
    logger.info("\n=== Example 5: Dynamic Modality Addition ===")
    
    # Add a new modality for text embeddings
    pipeline.add_modality("m5", input_dim=384, hidden_dim=4096)
    
    # Use the new modality
    new_modal_input = {
        "text": "Process this text embedding along with visual content.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "m1": [torch.randn(768)],
                "m5": [torch.randn(384)],  # New modality
            }
        }
    }
    
    response = pipeline(new_modal_input, max_new_tokens=100)
    print(f"Input: {new_modal_input['text']}")
    print(f"New modalities: {list(new_modal_input['multimodal_data']['multimodal_embeddings'].keys())}")
    print(f"Response: {response}")
    
    # Example 6: Model information
    logger.info("\n=== Example 6: Model Information ===")
    model_info = pipeline.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        if key == "modality_configs":
            print(f"  {key}:")
            for modality, config in value.items():
                print(f"    {modality}: {config}")
        else:
            print(f"  {key}: {value}")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main() 