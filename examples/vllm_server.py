#!/usr/bin/env python3
"""
vLLM Server Integration Example

This script demonstrates how to integrate the multimodal Qwen3 model with vLLM.
Note: This uses our custom pipeline rather than direct vLLM integration since our
current implementation doesn't follow vLLM's official multimodal pattern.
"""

import torch
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from multimodal_qwen3 import MultimodalQwen3Pipeline, create_example_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalVLLMServer:
    """
    Server wrapper for multimodal Qwen3 model that can work with or without vLLM.
    
    IMPORTANT: Our current MultimodalQwen3Model doesn't follow vLLM's official 
    multimodal integration pattern, so we use our pipeline instead.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        modality_configs: Dict[str, Dict[str, int]] = None,
        use_vllm_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize the server.
        
        Args:
            model_name: Base model name
            modality_configs: Modality configurations
            use_vllm_fallback: Whether to use vLLM for text-only generation
            **kwargs: Additional arguments for pipeline
        """
        self.model_name = model_name
        self.modality_configs = modality_configs or create_example_config()
        self.use_vllm_fallback = use_vllm_fallback
        
        # Initialize our multimodal pipeline
        logger.info("Initializing multimodal pipeline...")
        self.pipeline = MultimodalQwen3Pipeline(
            model_name=model_name,
            modality_configs=self.modality_configs,
            **kwargs
        )
        
        # Try to initialize vLLM for text-only fallback
        self.vllm_engine = None
        self.SamplingParams = None
        
        if use_vllm_fallback:
            self._init_vllm_fallback()
        
        logger.info(f"Server initialized with {len(self.modality_configs)} modalities")
    
    def _init_vllm_fallback(self):
        """Initialize vLLM for text-only generation as fallback."""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize vLLM WITHOUT custom_model parameter (which doesn't exist)
            self.vllm_engine = LLM(
                model=self.model_name,
                trust_remote_code=True,
                # Remove any invalid parameters like custom_model
            )
            
            self.SamplingParams = SamplingParams
            logger.info("vLLM fallback engine initialized for text-only generation")
            
        except ImportError:
            logger.warning("vLLM not available. Using pipeline for all generation.")
        except Exception as e:
            logger.warning(f"vLLM initialization failed: {e}. Using pipeline for all generation.")
    
    async def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate response asynchronously.
        
        Args:
            inputs: Input dictionary with text and optional multimodal data
            sampling_params: Sampling parameters
            
        Returns:
            Generated response
        """
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256,
            }
        
        # Check if this is a multimodal request
        has_multimodal = (
            "multimodal_data" in inputs and 
            "multimodal_embeddings" in inputs["multimodal_data"] and
            inputs["multimodal_data"]["multimodal_embeddings"]
        )
        
        if has_multimodal:
            # Use our multimodal pipeline for multimodal inputs
            return await self._generate_with_pipeline(inputs, sampling_params)
        elif self.vllm_engine is not None:
            # Use vLLM for text-only inputs (faster)
            return await self._generate_with_vllm(inputs, sampling_params)
        else:
            # Use pipeline as fallback
            return await self._generate_with_pipeline(inputs, sampling_params)
    
    async def _generate_with_pipeline(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate using our multimodal pipeline."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate():
                with torch.no_grad():
                    return self.pipeline.generate(
                        inputs=inputs,
                        max_new_tokens=sampling_params.get("max_tokens", 256),
                        temperature=sampling_params.get("temperature", 0.7),
                        top_p=sampling_params.get("top_p", 0.9),
                        do_sample=True,
                    )
            
            # Run generation in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                text = await loop.run_in_executor(executor, _generate)
            
            return {
                "text": text,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 0,  # Would need to calculate
                    "completion_tokens": len(self.pipeline.model.tokenizer.encode(text)) if text else 0,
                }
            }
                
        except Exception as e:
            logger.error(f"Pipeline generation error: {e}")
            return {"text": "", "finish_reason": "error", "error": str(e)}
    
    async def _generate_with_vllm(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate using vLLM for text-only inputs."""
        try:
            # Create sampling params
            vllm_sampling_params = self.SamplingParams(
                temperature=sampling_params.get("temperature", 0.7),
                top_p=sampling_params.get("top_p", 0.9),
                max_tokens=sampling_params.get("max_tokens", 256),
            )
            
            # Extract text input
            text_input = inputs.get("text", "")
            
            # Generate with vLLM
            outputs = self.vllm_engine.generate([text_input], vllm_sampling_params)
            
            if outputs:
                output = outputs[0].outputs[0].text
                return {
                    "text": output,
                    "finish_reason": outputs[0].outputs[0].finish_reason or "stop",
                    "usage": {
                        "prompt_tokens": len(outputs[0].prompt_token_ids),
                        "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    }
                }
            else:
                return {"text": "", "finish_reason": "error"}
                
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            return {"text": "", "finish_reason": "error", "error": str(e)}
    
    def generate_sync(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generation."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate_async(inputs, sampling_params))


def create_openai_compatible_response(
    result: Dict[str, Any],
    model: str = "multimodal-qwen3",
) -> Dict[str, Any]:
    """Create OpenAI-compatible response format."""
    import time
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("text", ""),
                },
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": result.get("usage", {}),
    }


# CORRECTED EXAMPLE: How to properly use our model with vLLM
def correct_usage_example():
    """
    Shows the CORRECT way to use our multimodal model.
    
    DO NOT use:
        llm = LLM(custom_model=MultimodalQwen3Model)  # custom_model doesn't exist!
    
    Instead, use our server wrapper or pipeline directly.
    """
    
    # Method 1: Use our server wrapper (recommended)
    server = MultimodalVLLMServer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Use smaller model for testing
        use_vllm_fallback=True,
    )
    
    # Method 2: Use our pipeline directly
    from multimodal_qwen3 import MultimodalQwen3Pipeline
    
    pipeline = MultimodalQwen3Pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        modality_configs={
            "vision": {"input_dim": 768, "hidden_dim": 4096},
            "audio": {"input_dim": 512, "hidden_dim": 4096},
        }
    )
    
    # Example usage
    inputs = {
        "text": "Describe this scene.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "vision": [torch.randn(768)],
                "audio": [torch.randn(512)],
            }
        }
    }
    
    # Generate with server
    result = server.generate_sync(inputs, {"max_tokens": 100})
    print(f"Server result: {result['text']}")
    
    # Generate with pipeline
    response = pipeline.generate(inputs, max_tokens=100)
    print(f"Pipeline result: {response}")


async def main():
    """Main example function."""
    logger.info("Starting vLLM server integration example")
    
    # Initialize server
    server = MultimodalVLLMServer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Use smaller model
        use_vllm_fallback=True,
    )
    
    # Example inputs
    test_inputs = [
        # Multimodal input
        {
            "text": "Describe this image.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768)],
                }
            }
        },
        # Text-only input (can use vLLM fallback)
        {
            "text": "What is the capital of France?",
        },
        # Complex multimodal input
        {
            "text": "Analyze this multimodal content.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768)],
                    "audio": [torch.randn(512)],
                }
            }
        }
    ]
    
    # Test generation
    logger.info("\n=== Testing Generation ===")
    
    for i, test_input in enumerate(test_inputs):
        logger.info(f"\nTest {i+1}:")
        print(f"Input: {test_input['text']}")
        
        if "multimodal_data" in test_input:
            modalities = list(test_input["multimodal_data"]["multimodal_embeddings"].keys())
            print(f"Modalities: {modalities}")
        else:
            print("Text-only input")
        
        # Generate response
        result = await server.generate_async(
            inputs=test_input,
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
            }
        )
        
        print(f"Response: {result['text']}")
        print(f"Finish reason: {result['finish_reason']}")
        
        # Create OpenAI-compatible response
        openai_response = create_openai_compatible_response(result)
        print(f"OpenAI format available")
    
    logger.info("Example completed!")


if __name__ == "__main__":
    print("=== IMPORTANT NOTE ===")
    print("This example shows how to use our multimodal model correctly.")
    print("Our current implementation does NOT follow vLLM's official multimodal pattern.")
    print("For true vLLM integration, the model would need to be rewritten to follow")
    print("vLLM's multimodal model interface (SupportsMultiModal, etc.)")
    print("======================\n")
    
    # Show correct usage
    print("For immediate usage, call correct_usage_example() or run main()")
    
    # Run async main
    asyncio.run(main()) 