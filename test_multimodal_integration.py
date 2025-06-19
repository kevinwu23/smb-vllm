#!/usr/bin/env python3
"""
Test Multimodal Integration

This script tests that multiple modalities are actually being integrated
by comparing outputs with different modality combinations.
"""

import torch
import logging
from multimodal_qwen3 import MultimodalQwen3Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_embeddings():
    """Create distinct test embeddings for different modalities."""
    return {
        "vision": [torch.randn(512) * 0.1 + torch.tensor([1.0] * 512)],      # Positive bias
        "audio": [torch.randn(256) * 0.1 + torch.tensor([-1.0] * 256)],     # Negative bias  
        "text_features": [torch.randn(768) * 0.1],                          # Zero-centered
    }


def test_multimodal_integration():
    """Test that different modality combinations produce different outputs."""
    
    # Configure pipeline with multiple modalities
    modality_configs = {
        "vision": {"input_dim": 512, "hidden_dim": 4096},
        "audio": {"input_dim": 256, "hidden_dim": 4096},
        "text_features": {"input_dim": 768, "hidden_dim": 4096},
    }
    
    pipeline = MultimodalQwen3Pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model for faster testing
        modality_configs=modality_configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    embeddings = create_test_embeddings()
    base_prompt = "Describe what you perceive from the multimodal input."
    
    # Test cases with different modality combinations
    test_cases = [
        ("Text only", {}),
        ("Vision only", {"vision": embeddings["vision"]}),
        ("Audio only", {"audio": embeddings["audio"]}),
        ("Vision + Audio", {"vision": embeddings["vision"], "audio": embeddings["audio"]}),
        ("All modalities", embeddings),
    ]
    
    results = {}
    
    logger.info("Testing multimodal integration...")
    print("\n" + "="*60)
    print("MULTIMODAL INTEGRATION TEST")
    print("="*60)
    
    for test_name, modality_data in test_cases:
        input_data = {
            "text": base_prompt,
            "multimodal_data": {"multimodal_embeddings": modality_data} if modality_data else None,
        }
        
        # Generate response
        response = pipeline.generate(input_data, max_new_tokens=50, temperature=0.1)  # Low temp for consistency
        
        # Store first 100 chars for comparison
        results[test_name] = response[:100]
        
        print(f"\n--- {test_name} ---")
        print(f"Modalities: {list(modality_data.keys()) if modality_data else 'None'}")
        print(f"Response: {response[:150]}...")
    
    # Analyze results
    print("\n" + "="*60)
    print("INTEGRATION ANALYSIS")
    print("="*60)
    
    unique_responses = set(results.values())
    
    print(f"Total test cases: {len(test_cases)}")
    print(f"Unique responses: {len(unique_responses)}")
    
    # Check if multimodal combinations produce different outputs
    text_only = results["Text only"]
    differences = []
    
    for test_name, response in results.items():
        if test_name != "Text only":
            similarity = response == text_only
            differences.append((test_name, not similarity))
            print(f"{test_name:15} | Different from text-only: {not similarity}")
    
    # Summary
    multimodal_different = sum(1 for _, is_different in differences if is_different)
    
    print(f"\nSUMMARY:")
    print(f"✅ Multimodal cases with different outputs: {multimodal_different}/{len(differences)}")
    
    if multimodal_different > 0:
        print("✅ PASS: Multiple modalities are producing different outputs")
        print("   This suggests multimodal integration is working")
    else:
        print("❌ FAIL: All outputs are identical")
        print("   This suggests multimodal data may not be integrated")
    
    return multimodal_different > 0


if __name__ == "__main__":
    try:
        success = test_multimodal_integration()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        exit(1) 