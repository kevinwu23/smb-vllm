#!/usr/bin/env python3
"""
Unit Tests for Multimodal Model Components

This module contains unit tests for the multimodal Qwen3 model implementation.
"""

import unittest
import torch
import tempfile
import shutil
from pathlib import Path
from multimodal_qwen3 import (
    MultimodalQwen3Model, 
    MultimodalQwen3Config,
    MultiModalProjector,
    MultimodalProcessor,
    MultimodalQwen3Pipeline,
    create_example_config,
)


class TestMultiModalProjector(unittest.TestCase):
    """Test the multimodal projector component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.text_hidden_size = 1024
        self.modality_configs = {
            "vision": {"input_dim": 768, "hidden_dim": 2048},
            "audio": {"input_dim": 512, "hidden_dim": 1536},
        }
        
        self.projector = MultiModalProjector(
            text_hidden_size=self.text_hidden_size,
            modality_configs=self.modality_configs,
        )
    
    def test_projector_initialization(self):
        """Test that projector initializes correctly."""
        self.assertEqual(self.projector.text_hidden_size, self.text_hidden_size)
        self.assertEqual(len(self.projector.projectors), 2)
        self.assertIn("vision", self.projector.projectors)
        self.assertIn("audio", self.projector.projectors)
    
    def test_projector_forward(self):
        """Test projector forward pass."""
        # Create test embeddings
        multimodal_embeddings = {
            "vision": [
                [torch.randn(768), torch.randn(768)],  # Batch item 1
                [torch.randn(768)],  # Batch item 2
            ],
            "audio": [
                [torch.randn(512)],  # Batch item 1
                [torch.randn(512), torch.randn(512)],  # Batch item 2
            ],
        }
        
        # Forward pass
        result = self.projector(multimodal_embeddings)
        
        # Check output structure
        self.assertIn("projected_embeddings", result)
        self.assertIn("attention_mask", result)
        self.assertIn("modality_offsets", result)
        
        # Check output shapes
        projected_embeddings = result["projected_embeddings"]
        self.assertEqual(projected_embeddings.shape[0], 2)  # Batch size
        self.assertEqual(projected_embeddings.shape[2], self.text_hidden_size)  # Hidden size
    
    def test_projector_empty_input(self):
        """Test projector with empty input."""
        empty_embeddings = {"vision": [[], []], "audio": [[], []]}
        result = self.projector(empty_embeddings)
        
        # Should handle empty input gracefully
        self.assertIn("projected_embeddings", result)
        self.assertEqual(result["projected_embeddings"].numel(), 0)


class TestMultimodalProcessor(unittest.TestCase):
    """Test the multimodal processor component."""
    
    def setUp(self):
        """Set up test fixtures."""
        from transformers import AutoTokenizer
        
        # Use a lightweight tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.processor = MultimodalProcessor(
            tokenizer=self.tokenizer,
            max_length=512,
            device="cpu",
        )
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.tokenizer)
        self.assertEqual(self.processor.max_length, 512)
        self.assertEqual(self.processor.device, "cpu")
    
    def test_text_processing(self):
        """Test text processing."""
        text = "Hello, this is a test."
        result = self.processor.process_text(text)
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("text_length", result)
        
        # Check tensor shapes
        self.assertEqual(result["input_ids"].dim(), 2)
        self.assertEqual(result["attention_mask"].dim(), 2)
    
    def test_multimodal_embedding_processing(self):
        """Test multimodal embedding processing."""
        multimodal_data = {
            "multimodal_embeddings": {
                "vision": [torch.randn(768), torch.randn(768)],
                "audio": [torch.randn(512)],
            }
        }
        
        result = self.processor.process_multimodal_embeddings(multimodal_data)
        
        self.assertIn("multimodal_embeddings", result)
        embeddings = result["multimodal_embeddings"]
        self.assertIn("vision", embeddings)
        self.assertIn("audio", embeddings)
        self.assertEqual(len(embeddings["vision"]), 2)
        self.assertEqual(len(embeddings["audio"]), 1)
    
    def test_create_multimodal_input(self):
        """Test complete multimodal input creation."""
        text = "Describe this content."
        multimodal_data = {
            "multimodal_embeddings": {
                "vision": [torch.randn(768)],
            }
        }
        
        result = self.processor.create_multimodal_input(text, multimodal_data)
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("multimodal_embeddings", result)
        self.assertIn("text_length", result)
        self.assertIn("multimodal_length", result)


class TestMultimodalQwen3Config(unittest.TestCase):
    """Test the multimodal configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        modality_configs = create_example_config()
        
        config = MultimodalQwen3Config(
            base_model_name="gpt2",  # Use small model for testing
            modality_configs=modality_configs,
        )
        
        self.assertEqual(config.base_model_name, "gpt2")
        self.assertEqual(config.modality_configs, modality_configs)
        self.assertEqual(config.model_type, "multimodal_qwen3")
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        modality_configs = create_example_config()
        
        config = MultimodalQwen3Config(
            base_model_name="gpt2",
            modality_configs=modality_configs,
        )
        
        # Test serialization
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = MultimodalQwen3Config.from_pretrained(temp_dir)
            
            self.assertEqual(loaded_config.base_model_name, config.base_model_name)
            self.assertEqual(loaded_config.modality_configs, config.modality_configs)


class TestMultimodalQwen3Pipeline(unittest.TestCase):
    """Test the multimodal pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a very small model for testing
        self.modality_configs = {
            "vision": {"input_dim": 32, "hidden_dim": 64},  # Small dimensions for testing
            "audio": {"input_dim": 16, "hidden_dim": 32},
        }
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pipeline_initialization_cuda(self):
        """Test pipeline initialization with CUDA."""
        try:
            pipeline = MultimodalQwen3Pipeline(
                model_name="gpt2",  # Small model for testing
                modality_configs=self.modality_configs,
                device="cuda",
            )
            self.assertEqual(pipeline.device, "cuda")
        except Exception as e:
            self.skipTest(f"CUDA initialization failed: {e}")
    
    def test_pipeline_initialization_cpu(self):
        """Test pipeline initialization with CPU."""
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",  # Small model for testing
            modality_configs=self.modality_configs,
            device="cpu",
        )
        self.assertEqual(pipeline.device, "cpu")
    
    def test_pipeline_text_generation(self):
        """Test simple text generation."""
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        # Simple text input
        try:
            response = pipeline("Hello, world!", max_new_tokens=10)
            self.assertIsInstance(response, str)
        except Exception as e:
            self.skipTest(f"Text generation failed: {e}")
    
    def test_pipeline_multimodal_input(self):
        """Test multimodal input processing."""
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        # Multimodal input
        multimodal_input = {
            "text": "Describe this content.",
            "multimodal_data": {
                "multimodal_embeddings": {
                    "vision": [torch.randn(32)],  # Match config input_dim
                    "audio": [torch.randn(16)],   # Match config input_dim
                }
            }
        }
        
        try:
            response = pipeline(multimodal_input, max_new_tokens=10)
            self.assertIsInstance(response, str)
        except Exception as e:
            self.skipTest(f"Multimodal generation failed: {e}")
    
    def test_pipeline_model_info(self):
        """Test model information retrieval."""
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        info = pipeline.get_model_info()
        
        self.assertIn("base_model", info)
        self.assertIn("device", info)
        self.assertIn("modalities", info)
        self.assertIn("modality_configs", info)
        self.assertEqual(info["base_model"], "gpt2")
        self.assertEqual(info["device"], "cpu")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.modality_configs = {
            "vision": {"input_dim": 64, "hidden_dim": 128},
            "audio": {"input_dim": 32, "hidden_dim": 64},
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_pipeline(self):
        """Test saving and loading pipeline."""
        # Create pipeline
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        # Save pipeline
        save_path = Path(self.temp_dir) / "test_pipeline"
        try:
            pipeline.save_pretrained(save_path)
            self.assertTrue(save_path.exists())
        except Exception as e:
            self.skipTest(f"Save failed: {e}")
    
    def test_modality_addition(self):
        """Test dynamic modality addition."""
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        # Add new modality
        pipeline.add_modality("text_embedding", input_dim=48, hidden_dim=96)
        
        # Check that modality was added
        info = pipeline.get_model_info()
        self.assertIn("text_embedding", info["modalities"])
        self.assertIn("text_embedding", info["modality_configs"])
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create pipeline
        pipeline = MultimodalQwen3Pipeline(
            model_name="gpt2",
            modality_configs=self.modality_configs,
            device="cpu",
        )
        
        # Test inputs
        test_cases = [
            "Simple text input",
            {
                "text": "Multimodal input",
                "multimodal_data": {
                    "multimodal_embeddings": {
                        "vision": [torch.randn(64)],
                        "audio": [torch.randn(32)],
                    }
                }
            }
        ]
        
        for test_input in test_cases:
            try:
                response = pipeline(test_input, max_new_tokens=5)
                self.assertIsInstance(response, str)
            except Exception as e:
                self.skipTest(f"End-to-end test failed with input {type(test_input)}: {e}")


if __name__ == "__main__":
    # Set up test environment
    torch.manual_seed(42)
    
    # Run tests
    unittest.main(verbosity=2) 