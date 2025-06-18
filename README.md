# Multimodal LLM Extension for vLLM

This project extends a base LLM (Qwen3) in vLLM to handle multimodal inputs (text + arbitrary embeddings) and generate text conditioned on both modalities.

## Features

- **Custom Model Class**: Extends Qwen3 to support multimodal inputs
- **Flexible Input Format**: Supports text input and arbitrary embeddings as dictionaries
- **Multimodal Projection**: MLP connector to align text and multimodal embeddings
- **KV Cache Optimization**: Efficient caching with variable-length embeddings
- **vLLM Integration**: Full integration with vLLM's serving and inference pipeline

## Architecture

The implementation consists of:

1. **MultimodalQwen3Model**: Custom model class extending Qwen3ForCausalLM
2. **MultiModalProjector**: MLP projection layer for embedding alignment
3. **Custom Input Processor**: Handles multimodal data format conversion
4. **vLLM Integration**: Registers the model with vLLM's model registry

## Installation

```bash
# Install vLLM and dependencies
pip install vllm>=0.6.0
pip install torch transformers
pip install numpy pillow

# Install this package
pip install -e .
```

## Usage

### Basic Example

```python
from multimodal_qwen3 import MultimodalQwen3Pipeline
import torch

# Initialize the pipeline
pipeline = MultimodalQwen3Pipeline(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda"
)

# Prepare multimodal input
input_data = {
    "text": "Describe this scene.",
    "multimodal_data": {
        "multimodal_embeddings": {
            "m1": [torch.randn(768), torch.randn(768)],  # Visual embeddings
            "m2": [torch.randn(512)]  # Audio embeddings
        }
    }
}

# Generate response
response = pipeline.generate(input_data, max_tokens=100)
print(f"Response: {response}")
```

### vLLM Server Integration

```python
from vllm import LLM, SamplingParams
from multimodal_qwen3 import MultimodalQwen3Model

# Initialize vLLM with custom model
llm = LLM(
    model="multimodal-qwen3",
    trust_remote_code=True,
    custom_model=MultimodalQwen3Model
)

# Use with vLLM's generate interface
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(inputs, sampling_params)
```

## Input Format

The model accepts inputs in the following format:

```python
{
    "text": "Your text prompt here",
    "multimodal_data": {
        "multimodal_embeddings": {
            "modality_1": [tensor1, tensor2, ...],  # List of tensors
            "modality_2": [tensor3, tensor4, ...],  # Different embedding dims supported
            # ... more modalities
        }
    }
}
```

## Design Choices

### 1. Projection Layer Architecture
- **MLP Connector**: Uses a 2-layer MLP with ReLU activation
- **Dimension Alignment**: Projects multimodal embeddings to text embedding dimension
- **Modality-Specific**: Separate projectors for each modality type
- **Learnable**: All projection weights are trainable parameters

### 2. Embedding Fusion Strategy
- **Early Fusion**: Multimodal embeddings are projected and concatenated with text embeddings
- **Position Encoding**: Maintains positional information for fused embeddings
- **Attention Masking**: Proper masking for variable-length multimodal sequences

### 3. KV Cache Optimization
- **Dynamic Sequence Length**: Handles variable-length embeddings efficiently
- **Memory Management**: Optimized memory usage for multimodal inputs
- **Batch Processing**: Supports batched multimodal inputs

## File Structure

```
multimodal-llm-extension/
├── multimodal_qwen3/
│   ├── __init__.py
│   ├── model.py              # Core multimodal model implementation
│   ├── projector.py          # Multimodal projection layers
│   ├── processor.py          # Input processing and formatting
│   └── pipeline.py           # High-level pipeline interface
├── examples/
│   ├── basic_usage.py        # Basic usage example
│   ├── vllm_server.py        # vLLM server integration
│   └── training_example.py   # Fine-tuning examples
├── tests/
│   ├── test_model.py         # Unit tests
│   └── test_integration.py   # Integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## Performance Considerations

- **Memory Efficiency**: Uses efficient tensor operations and memory management
- **Compute Optimization**: Leverages vLLM's optimized attention mechanisms
- **Scalability**: Supports distributed inference with tensor parallelism

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on top of vLLM and Transformers
- Inspired by LLaVA and other multimodal architectures
- Thanks to the Qwen team for the base model 