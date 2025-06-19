#!/usr/bin/env python3
"""
CLIP Integration Example

This script demonstrates using real CLIP embeddings with the multimodal Qwen3 extension.
It loads a CLIP model, processes a real image, and uses the embeddings for text generation.
"""

import torch
import logging
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel  # type: ignore
from multimodal_qwen3 import MultimodalQwen3Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_clip_model():
    """Load CLIP model and processor."""
    logger.info("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor


def load_sample_image():
    """Load the local image.jpeg file."""
    image_path = "examples/image.jpeg"
    
    try:
        logger.info(f"Loading local image from: {image_path}")
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Image loaded successfully. Size: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        logger.info("Please make sure image.jpeg exists in the examples/ directory")
        # Fallback: create a simple synthetic image
        logger.info("Creating fallback synthetic image...")
        image = Image.new('RGB', (224, 224), color=(0, 0, 255))  # Blue color
        return image


def extract_clip_embeddings(image, clip_model, clip_processor):
    """Extract CLIP embeddings from an image."""
    logger.info("Extracting CLIP embeddings...")
    
    # Process the image
    inputs = clip_processor(images=image, return_tensors="pt")
    
    # Extract image embeddings
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        # Normalize the features (standard practice for CLIP)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    logger.info(f"CLIP embeddings extracted. Shape: {image_features.shape}")
    return image_features.squeeze(0)  # Remove batch dimension


def main():
    """Main function demonstrating CLIP integration."""
    logger.info("Starting CLIP integration example")
    
    # Load CLIP model
    clip_model, clip_processor = load_clip_model()
    
    # Load sample image
    image = load_sample_image()
    
    # Extract CLIP embeddings
    clip_embeddings = extract_clip_embeddings(image, clip_model, clip_processor)
    
    # Configure multimodal pipeline
    # CLIP ViT-Base outputs 512-dimensional embeddings
    modality_configs = {
        "vision": {"input_dim": 512, "hidden_dim": 4096},  # CLIP embedding dimension
    }
    
    logger.info("Initializing multimodal pipeline...")
    pipeline = MultimodalQwen3Pipeline(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        modality_configs=modality_configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Test with the specific dog counting question
    test_cases = [
        "How many dogs are in this image?",
        "Describe what you see in this image.",
        "What is the main subject of this image?",
        "Are there any animals in this image?",
        "What kind of scene is shown?",
    ]
    
    logger.info("\n" + "="*60)
    logger.info("TESTING WITH REAL CLIP EMBEDDINGS")
    logger.info("="*60)
    
    for i, prompt in enumerate(test_cases, 1):
        logger.info(f"\n--- Test {i} ---")
        
        # Create multimodal input with real CLIP embeddings
        multimodal_input = {
            "text": prompt,
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [clip_embeddings]  # Real CLIP embeddings
                }
            }
        }
        
        # Generate response
        response = pipeline.generate(multimodal_input, max_new_tokens=100, temperature=0.7)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 40)
    
    # Compare with text-only baseline
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: TEXT-ONLY BASELINE")
    logger.info("="*60)
    
    text_only_prompt = "Describe a typical outdoor scene."
    text_only_response = pipeline.generate(text_only_prompt, max_new_tokens=100)
    print(f"Text-only prompt: {text_only_prompt}")
    print(f"Text-only response: {text_only_response}")
    
    # Show embedding statistics
    logger.info("\n" + "="*60)
    logger.info("CLIP EMBEDDING STATISTICS")
    logger.info("="*60)
    print(f"Embedding shape: {clip_embeddings.shape}")
    print(f"Embedding norm: {clip_embeddings.norm().item():.4f}")
    print(f"Min value: {clip_embeddings.min().item():.4f}")
    print(f"Max value: {clip_embeddings.max().item():.4f}")
    print(f"Mean value: {clip_embeddings.mean().item():.4f}")
    print(f"Std value: {clip_embeddings.std().item():.4f}")
    
    logger.info("CLIP integration example completed!")


if __name__ == "__main__":
    # Check dependencies
    try:
        import requests
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install requests pillow transformers")
        exit(1)
    
    main() 