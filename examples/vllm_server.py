#!/usr/bin/env python3
"""
vLLM Server Integration Example

This script demonstrates how to integrate the multimodal Qwen3 model with vLLM
for high-performance serving and inference.
"""

import torch
import asyncio
import json
import logging
from typing import Dict, List, Any
from multimodal_qwen3 import MultimodalQwen3Model, MultimodalQwen3Config, create_example_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalVLLMServer:
    """
    vLLM server wrapper for multimodal Qwen3 model.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        modality_configs: Dict[str, Dict[str, int]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        """
        Initialize the vLLM server.
        
        Args:
            model_name: Base model name
            modality_configs: Modality configurations
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model sequence length
        """
        self.model_name = model_name
        self.modality_configs = modality_configs or create_example_config()
        
        # Initialize model configuration
        self.config = MultimodalQwen3Config(
            base_model_name=model_name,
            modality_configs=self.modality_configs,
        )
        
        # Initialize vLLM engine
        self._init_vllm_engine(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        
        logger.info(f"vLLM server initialized with {len(self.modality_configs)} modalities")
    
    def _init_vllm_engine(
        self,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        """Initialize vLLM engine with multimodal model."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=True,  # For custom models
            )
            
            # Initialize async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Store sampling params class
            self.SamplingParams = SamplingParams
            
            logger.info("vLLM engine initialized successfully")
            
        except ImportError:
            logger.error("vLLM not available. Installing fallback implementation.")
            self._init_fallback_engine()
    
    def _init_fallback_engine(self):
        """Initialize fallback engine when vLLM is not available."""
        # Create multimodal model directly
        self.model = MultimodalQwen3Model(self.config)
        self.engine = None
        
        # Mock sampling params
        class MockSamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        self.SamplingParams = MockSamplingParams
        logger.info("Fallback engine initialized")
    
    async def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate response asynchronously.
        
        Args:
            inputs: Input dictionary with text and multimodal data
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
        
        if self.engine is not None:
            # Use vLLM engine
            return await self._generate_with_vllm(inputs, sampling_params)
        else:
            # Use fallback model
            return await self._generate_with_fallback(inputs, sampling_params)
    
    async def _generate_with_vllm(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate using vLLM engine."""
        try:
            # Create sampling params
            vllm_sampling_params = self.SamplingParams(**sampling_params)
            
            # For now, we'll process multimodal inputs by converting them to text
            # In a full implementation, this would integrate with vLLM's multimodal handling
            text_input = self._convert_multimodal_to_text(inputs)
            
            # Generate with vLLM
            results = await self.engine.generate(text_input, vllm_sampling_params)
            
            # Process results
            if results:
                output = results[0].outputs[0].text
                return {
                    "text": output,
                    "finish_reason": results[0].outputs[0].finish_reason,
                    "usage": {
                        "prompt_tokens": len(results[0].prompt_token_ids),
                        "completion_tokens": len(results[0].outputs[0].token_ids),
                    }
                }
            else:
                return {"text": "", "finish_reason": "error"}
                
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            return {"text": "", "finish_reason": "error", "error": str(e)}
    
    async def _generate_with_fallback(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate using fallback model."""
        try:
            # Use the multimodal model directly
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=inputs,
                    max_new_tokens=sampling_params.get("max_tokens", 256),
                    temperature=sampling_params.get("temperature", 0.7),
                    top_p=sampling_params.get("top_p", 0.9),
                    do_sample=True,
                )
            
            # Decode output
            if isinstance(outputs, torch.Tensor):
                text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove input text
                if "text" in inputs:
                    input_text = inputs["text"]
                    if text.startswith(input_text):
                        text = text[len(input_text):].strip()
                
                return {
                    "text": text,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 0,  # Would need to calculate
                        "completion_tokens": len(self.model.tokenizer.encode(text)),
                    }
                }
            else:
                return {"text": str(outputs), "finish_reason": "stop"}
                
        except Exception as e:
            logger.error(f"Fallback generation error: {e}")
            return {"text": "", "finish_reason": "error", "error": str(e)}
    
    def _convert_multimodal_to_text(self, inputs: Dict[str, Any]) -> str:
        """
        Convert multimodal inputs to text representation.
        
        This is a simplified approach for demonstration.
        In practice, you'd want to integrate multimodal embeddings properly.
        """
        text = inputs.get("text", "")
        
        if "multimodal_data" in inputs:
            multimodal_data = inputs["multimodal_data"]
            if "multimodal_embeddings" in multimodal_data:
                modalities = list(multimodal_data["multimodal_embeddings"].keys())
                if modalities:
                    text += f" [Multimodal content includes: {', '.join(modalities)}]"
        
        return text
    
    def generate_sync(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generation.
        
        Args:
            inputs: Input dictionary
            sampling_params: Sampling parameters
            
        Returns:
            Generated response
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate_async(inputs, sampling_params))


def create_openai_compatible_response(
    result: Dict[str, Any],
    model: str = "multimodal-qwen3",
) -> Dict[str, Any]:
    """
    Create OpenAI-compatible response format.
    
    Args:
        result: Generation result
        model: Model name
        
    Returns:
        OpenAI-compatible response
    """
    return {
        "id": "chatcmpl-multimodal",
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
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


async def main():
    """Main example function."""
    logger.info("Starting vLLM server integration example")
    
    # Initialize server
    server = MultimodalVLLMServer(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
    )
    
    # Example inputs
    test_inputs = [
        {
            "text": "Describe this image.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768)],
                }
            }
        },
        {
            "text": "What do you hear in this audio?",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "audio": [torch.randn(512), torch.randn(512)],
                }
            }
        },
        {
            "text": "Analyze this multimodal content.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(768)],
                    "audio": [torch.randn(512)],
                    "code": [torch.randn(768)],
                }
            }
        }
    ]
    
    # Test generation
    logger.info("\n=== Testing Multimodal Generation ===")
    
    for i, test_input in enumerate(test_inputs):
        logger.info(f"\nTest {i+1}:")
        print(f"Input: {test_input['text']}")
        
        if "multimodal_data" in test_input:
            modalities = list(test_input["multimodal_data"]["multimodal_embeddings"].keys())
            print(f"Modalities: {modalities}")
        
        # Generate response
        result = await server.generate_async(
            inputs=test_input,
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 150,
            }
        )
        
        print(f"Response: {result['text']}")
        print(f"Finish reason: {result['finish_reason']}")
        
        # Create OpenAI-compatible response
        openai_response = create_openai_compatible_response(result)
        print(f"OpenAI format: {json.dumps(openai_response, indent=2)}")
    
    # Benchmark performance
    logger.info("\n=== Performance Benchmark ===")
    
    import time
    num_requests = 5
    total_time = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        result = await server.generate_async(
            inputs=test_inputs[0],
            sampling_params={"temperature": 0.7, "max_tokens": 50}
        )
        
        end_time = time.time()
        request_time = end_time - start_time
        total_time += request_time
        
        logger.info(f"Request {i+1}: {request_time:.2f}s")
    
    avg_latency = total_time / num_requests
    logger.info(f"Average latency: {avg_latency:.2f}s")
    logger.info(f"Throughput: {1/avg_latency:.2f} requests/second")
    
    logger.info("vLLM server integration example completed!")


if __name__ == "__main__":
    asyncio.run(main()) 