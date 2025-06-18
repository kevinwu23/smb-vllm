# Usage Guide for Multimodal Qwen3

## Quick Start

### Method 1: Using the Pipeline (Recommended)

```python
import torch
from multimodal_qwen3 import MultimodalQwen3Pipeline

# Initialize pipeline
pipeline = MultimodalQwen3Pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    modality_configs={
        "vision": {"input_dim": 768, "hidden_dim": 4096},
        "audio": {"input_dim": 512, "hidden_dim": 4096},
    }
)

# Prepare input
input_data = {
    "text": "Describe this scene.",
    "multimodal_data": {
        "multimodal_embeddings": {
            "vision": [torch.randn(768), torch.randn(768)],
            "audio": [torch.randn(512)]
        }
    }
}

# Generate response
response = pipeline.generate(input_data, max_tokens=100)
print(response)
```

### Method 2: Using the Server Wrapper

```python
from examples.vllm_server import MultimodalVLLMServer

# Initialize server
server = MultimodalVLLMServer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_vllm_fallback=True,
)

# Generate response
result = server.generate_sync(input_data, {"max_tokens": 100})
print(result["text"])
```

## ❌ INCORRECT Usage with vLLM

**DO NOT DO THIS:**

```python
# This will cause TypeError: EngineArgs.__init__() got an unexpected keyword argument 'custom_model'
from vllm import LLM
from multimodal_qwen3 import MultimodalQwen3Model

llm = LLM(
    model="multimodal-qwen3",
    trust_remote_code=True,
    custom_model=MultimodalQwen3Model  # ❌ This parameter doesn't exist!
)
```

**The `custom_model` parameter does not exist in vLLM's LLM constructor.**

## ✅ CORRECT Usage with vLLM

Our current implementation doesn't follow vLLM's official multimodal pattern. Instead:

```python
# Option 1: Use our pipeline directly
from multimodal_qwen3 import MultimodalQwen3Pipeline
pipeline = MultimodalQwen3Pipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct")
response = pipeline.generate(input_data)

# Option 2: Use our server wrapper (which can use vLLM for text-only)
from examples.vllm_server import MultimodalVLLMServer
server = MultimodalVLLMServer(model_name="Qwen/Qwen2.5-0.5B-Instruct")
result = server.generate_sync(input_data)
```

## Input Format

The expected input format is:

```python
{
    "text": "Your text prompt here",
    "multimodal_data": {
        "multimodal_embeddings": {
            "modality1": [tensor1, tensor2, ...],  # List of tensors
            "modality2": [tensor3, tensor4, ...],  # Each modality can have multiple embeddings
        }
    }
}
```

## Common Issues

### 1. TypeError with vLLM
- **Problem**: `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'custom_model'`
- **Solution**: Use our pipeline or server wrapper instead of trying to pass our model directly to vLLM

### 2. AttributeError in generate
- **Problem**: `AttributeError: 'super' object has no attribute 'generate'`
- **Solution**: This was fixed in the latest version. Update your code.

### 3. Dimension mismatches
- **Problem**: Tensor dimension errors in projector
- **Solution**: Ensure your embedding dimensions match the `input_dim` specified in `modality_configs`

## vLLM Integration Notes

Our current implementation is **NOT** a true vLLM multimodal model. It's a wrapper that:

1. ✅ Works with arbitrary embeddings
2. ✅ Supports multiple modalities 
3. ✅ Can use vLLM for text-only generation as fallback
4. ❌ Does NOT follow vLLM's official multimodal model interface
5. ❌ Does NOT integrate with vLLM's multimodal processing pipeline

### For True vLLM Integration

To create a proper vLLM multimodal model, you would need to:

1. Extend the actual base model class (e.g., `Qwen2ForCausalLM`)
2. Implement `SupportsMultiModal` interface
3. Create vLLM-compatible processor classes (`BaseMultiModalProcessor`, etc.)
4. Register with `@MULTIMODAL_REGISTRY.register_processor`
5. Follow vLLM's multimodal documentation patterns

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Simple usage examples
- `demo.py` - Comprehensive demonstration
- `vllm_server.py` - Server wrapper (corrected)
- `training_example.py` - Training script

## Performance Tips

1. Use smaller models for testing (`Qwen2.5-0.5B-Instruct`)
2. Enable vLLM fallback for text-only requests
3. Batch multiple requests when possible
4. Use appropriate GPU memory settings

## Support

For issues with:
- Our multimodal implementation → Check examples and this guide
- vLLM integration → Use our server wrapper, not direct integration
- Model loading → Ensure sufficient compute resources 