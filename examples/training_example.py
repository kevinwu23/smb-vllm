#!/usr/bin/env python3
"""
Training Example

This script demonstrates how to fine-tune the multimodal Qwen3 model
on a custom dataset with multimodal inputs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Any, Tuple
import json
import random
from pathlib import Path
from multimodal_qwen3 import MultimodalQwen3Model, MultimodalQwen3Config, create_example_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Dataset class for multimodal training data.
    """
    
    def __init__(
        self,
        data_path: str = None,
        tokenizer=None,
        max_length: int = 512,
        modality_configs: Dict[str, Dict[str, int]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to training data (optional, will create synthetic data if None)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            modality_configs: Configuration for modalities
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modality_configs = modality_configs or create_example_config()
        
        if data_path and Path(data_path).exists():
            self.data = self._load_data(data_path)
        else:
            logger.info("Creating synthetic training data")
            self.data = self._create_synthetic_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        with open(data_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic training data for demonstration."""
        synthetic_data = []
        
        # Define some example scenarios
        scenarios = [
            {
                "input_text": "Describe the visual scene.",
                "output_text": "The image shows a beautiful landscape with mountains in the background and a lake in the foreground. The scene is peaceful and serene.",
                "modalities": ["vision"],
            },
            {
                "input_text": "What do you hear in this audio?",
                "output_text": "I can hear birds chirping, wind rustling through leaves, and distant water flowing. It sounds like a natural outdoor environment.",
                "modalities": ["audio"],
            },
            {
                "input_text": "Analyze this code snippet.",
                "output_text": "This is a Python function that implements a binary search algorithm. It efficiently searches for a target value in a sorted array.",
                "modalities": ["code"],
            },
            {
                "input_text": "Describe the scene and sounds.",
                "output_text": "The visual shows a bustling city street with people walking, while the audio captures the sounds of traffic, conversations, and urban life.",
                "modalities": ["vision", "audio"],
            },
            {
                "input_text": "What can you tell me about this multimedia content?",
                "output_text": "This content combines visual imagery of a nature documentary with corresponding narration and natural sounds, creating an immersive experience.",
                "modalities": ["vision", "audio", "generic"],
            }
        ]
        
        # Generate synthetic data
        for _ in range(100):  # Create 100 synthetic samples
            scenario = random.choice(scenarios)
            
            # Create multimodal embeddings for the scenario
            multimodal_embeddings = {}
            for modality in scenario["modalities"]:
                if modality in self.modality_configs:
                    input_dim = self.modality_configs[modality]["input_dim"]
                    # Create random embeddings (in practice, these would be real embeddings)
                    num_embeddings = random.randint(1, 3)
                    multimodal_embeddings[modality] = [
                        torch.randn(input_dim) for _ in range(num_embeddings)
                    ]
            
            sample = {
                "input_text": scenario["input_text"],
                "output_text": scenario["output_text"],
                "multimodal_embeddings": multimodal_embeddings,
            }
            
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # Tokenize input and output text
        input_text = sample["input_text"]
        output_text = sample["output_text"]
        full_text = f"{input_text} {output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids but with input tokens masked)
        labels = encoding["input_ids"].clone()
        input_length = len(self.tokenizer(input_text, add_special_tokens=False)["input_ids"])
        labels[0, :input_length] = -100  # Mask input tokens for loss calculation
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "multimodal_embeddings": sample.get("multimodal_embeddings", {}),
        }


class MultimodalTrainer:
    """
    Trainer class for multimodal model fine-tuning.
    """
    
    def __init__(
        self,
        model: MultimodalQwen3Model,
        train_dataset: MultimodalDataset,
        val_dataset: MultimodalDataset = None,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_directory: str = "./multimodal_model_checkpoints",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Multimodal model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_ratio: Warmup ratio for learning rate schedule
            max_grad_norm: Maximum gradient norm for clipping
            save_directory: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        logger.info(f"Trainer initialized for {num_epochs} epochs with {total_steps} total steps")
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching multimodal data."""
        # Stack text-related tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # Collect multimodal embeddings
        batched_multimodal = {}
        
        # Find all unique modalities in the batch
        all_modalities = set()
        for item in batch:
            all_modalities.update(item["multimodal_embeddings"].keys())
        
        # Batch embeddings by modality
        for modality in all_modalities:
            batched_multimodal[modality] = []
            for item in batch:
                modality_embeddings = item["multimodal_embeddings"].get(modality, [])
                batched_multimodal[modality].append(modality_embeddings)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "multimodal_embeddings": batched_multimodal,
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                multimodal_embeddings=batch["multimodal_embeddings"],
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        if not self.val_dataset:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    multimodal_embeddings=batch["multimodal_embeddings"],
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Full training loop."""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            if self.val_dataset:
                logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.save_directory / f"epoch_{epoch + 1}"
            self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = self.save_directory / "best_model"
                self.save_checkpoint(best_model_path, epoch, train_loss, val_loss)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
    
    def save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ):
        """Save model checkpoint."""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path))
        
        # Save training state
        state = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        
        torch.save(state, checkpoint_path / "training_state.pt")
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function."""
    logger.info("Starting multimodal model training example")
    
    # Configuration
    modality_configs = create_example_config()
    
    # Initialize model
    config = MultimodalQwen3Config(
        base_model_name="Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for faster training
        modality_configs=modality_configs,
    )
    
    model = MultimodalQwen3Model(config)
    
    # Create datasets
    train_dataset = MultimodalDataset(
        tokenizer=model.tokenizer,
        max_length=512,
        modality_configs=modality_configs,
    )
    
    # Create validation dataset (smaller for demo)
    val_dataset = MultimodalDataset(
        tokenizer=model.tokenizer,
        max_length=512,
        modality_configs=modality_configs,
    )
    val_dataset.data = val_dataset.data[:20]  # Use only 20 samples for validation
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,  # Small batch size for demo
        learning_rate=5e-5,
        num_epochs=2,  # Small number of epochs for demo
        save_directory="./multimodal_checkpoints",
    )
    
    # Start training
    trainer.train()
    
    # Test the trained model
    logger.info("\n=== Testing Trained Model ===")
    
    test_input = {
        "text": "Describe this visual and audio content.",
        "multimodal_data": {
            "multimodal_embeddings": {
                "vision": [torch.randn(768)],
                "audio": [torch.randn(512)],
            }
        }
    }
    
    model.eval()
    with torch.no_grad():
        response = model.generate(
            inputs=test_input,
            max_new_tokens=100,
            temperature=0.7,
        )
    
    generated_text = model.tokenizer.decode(response[0], skip_special_tokens=True)
    print(f"Test input: {test_input['text']}")
    print(f"Generated response: {generated_text}")
    
    logger.info("Training example completed!")


if __name__ == "__main__":
    main() 