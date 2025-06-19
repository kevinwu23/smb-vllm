# Multimodal LLM Extension for vLLM

This project extends a base LLM (Qwen3) to handle multimodal inputs (text + arbitrary embeddings) and generate text conditioned on both modalities.

## Overview

The implementation provides a custom model class that can process:
- **Text input**: Standard token IDs
- **Multimodal embeddings**: Arbitrary embeddings as dictionaries with key-value pairs like `{"m1": [tensor(), tensor(), ...], "m2": [tensor(), tensor(), ...]}`

The model combines these inputs into a single hidden representation before generation, enabling text generation conditioned on multiple modalities.

## Architecture

### 1. Custom Model Class (`MultimodalQwen3Model`)
- Extends Qwen3 to support multimodal inputs
- Handles variable-length embedding sequences
- Integrates multimodal data through projection layers
- Maintains compatibility with vLLM's generation pipeline

### 2. Multimodal Embedding Projection (`MultiModalProjector`)
- **MLP Connector**: 2-layer MLP with configurable activation functions
- **Dimension Alignment**: Projects multimodal embeddings to text embedding dimension
- **Modality-Specific**: Separate projectors for each modality type
- **Flexible Configuration**: Supports different input dimensions per modality

### 3. KV Cache Optimization
- Efficient handling of variable-length multimodal sequences
- Proper attention masking for fused embeddings
- Memory-optimized processing for batched multimodal inputs

## Design Choices

### Early Fusion Strategy
- Multimodal embeddings are projected and concatenated with text embeddings
- Enables the language model to jointly process all modalities
- Maintains positional encoding for proper sequence modeling

### Modality-Agnostic Architecture
- Generic modality names (`m1`, `m2`, etc.) for flexibility
- Configurable input dimensions per modality
- Easy extension to new modality types

### Memory Efficiency
- Lazy loading of projection layers
- Optimized tensor operations
- Batched processing support

## Installation

```bash
# Clone the repository
git clone https://github.com/kevinwu23/smb-vllm.git
cd smb-vllm

# Install in development mode
pip install -e .
```

## Usage

### Basic Example

```python
from multimodal_qwen3 import MultimodalQwen3Pipeline
import torch

# Configure modalities
modality_configs = {
    "m1": {"input_dim": 768, "hidden_dim": 4096},   # e.g., vision embeddings
    "m2": {"input_dim": 512, "hidden_dim": 4096},   # e.g., audio embeddings
}

# Initialize pipeline
pipeline = MultimodalQwen3Pipeline(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    modality_configs=modality_configs,
    device="cuda"
)

# Prepare multimodal input
input_data = {
    "text": "Describe this scene.",
    "multimodal_data": {
        "multimodal_embeddings": {
            "m1": [torch.randn(768), torch.randn(768)],  # Vision embeddings
            "m2": [torch.randn(512)]                     # Audio embeddings
        }
    }
}

# Generate response
response = pipeline.generate(input_data, max_tokens=100)
print(f"Response: {response}")
```

### Real CLIP Example

For a more realistic demonstration using actual CLIP embeddings:

```python
# See examples/clip_example.py for complete implementation
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process image and extract embeddings
inputs = clip_processor(images=image, return_tensors="pt")
image_features = clip_model.get_image_features(**inputs)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Use with multimodal pipeline
pipeline = MultimodalQwen3Pipeline(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    modality_configs={"vision": {"input_dim": 512, "hidden_dim": 4096}},
    device="cuda"
)

response = pipeline.generate({
    "text": "Describe what you see in this image.",
    "multimodal_data": {"multimodal_embeddings": {"vision": [image_features.squeeze(0)]}}
}, max_new_tokens=100)
```

### Input Format

The model accepts inputs in the following format:

```python
{
    "text": "Your text prompt here",
    "multimodal_data": {
        "multimodal_embeddings": {
            "m1": [tensor1, tensor2, ...],  # List of tensors for modality 1
            "m2": [tensor3, tensor4, ...],  # List of tensors for modality 2
            # ... more modalities
        }
    }
}
```

### Example Input/Output

**Input:**
```python
{
    "text": "Describe this scene.",
    "multimodal_data": {
        "multimodal_embeddings": {
            "m1": [torch.randn(768)],  # Simulated vision embedding
            "m2": [torch.randn(512)]   # Simulated audio embedding
        }
    }
}
```

**Output:**
```
"A serene landscape with mountains in the background and gentle sounds of nature. The scene captures the peaceful atmosphere of a quiet outdoor setting."
```

## Project Structure

```
multimodal-llm-extension/
├── multimodal_qwen3/
│   ├── __init__.py              # Package exports
│   ├── model.py                 # Core multimodal model implementation
│   ├── projector.py             # Multimodal projection layers
│   ├── processor.py             # Input processing and formatting
│   └── pipeline.py              # High-level pipeline interface
├── examples/
│   ├── basic_usage.py           # Basic usage example
│   └── clip_example.py          # Real CLIP embeddings example
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Key Features

- **Flexible Modality Support**: Handle arbitrary embedding types and dimensions
- **Efficient Processing**: Optimized for variable-length sequences
- **Easy Integration**: Simple pipeline interface for quick usage
- **Extensible Design**: Easy to add new modalities or modify existing ones
- **Memory Optimized**: Efficient handling of large multimodal inputs

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- vLLM >= 0.6.0
- NumPy >= 1.20.0

## Limitations

- Current implementation uses a custom pipeline rather than following vLLM's official multimodal pattern
- For production use with vLLM serving, the model would need to be rewritten to implement vLLM's `SupportsMultiModal` interface
- Multimodal embeddings must be pre-computed (no built-in encoders)

## Future Improvements

1. **vLLM Integration**: Rewrite to follow vLLM's official multimodal model interface
2. **Built-in Encoders**: Add support for raw image/audio inputs with built-in encoders
3. **Advanced Fusion**: Implement attention-based fusion mechanisms
4. **Quantization Support**: Add support for quantized multimodal models

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request 
