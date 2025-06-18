"""
Multimodal Input Processor

This module handles processing and formatting of multimodal inputs for the model.
"""

import torch
from typing import Dict, List, Union, Any, Optional, Tuple
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """
    Processor for handling multimodal inputs and converting them to model format.
    
    This processor handles:
    1. Text tokenization
    2. Multimodal embedding processing
    3. Input formatting for the model
    4. Attention mask generation
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multimodal_token: str = "<|multimodal|>",
        max_length: int = 2048,
        device: str = "cuda",
    ):
        """
        Initialize the multimodal processor.
        
        Args:
            tokenizer: Tokenizer for text processing
            multimodal_token: Special token to represent multimodal embeddings
            max_length: Maximum sequence length
            device: Device for tensor operations
        """
        self.tokenizer = tokenizer
        self.multimodal_token = multimodal_token
        self.max_length = max_length
        self.device = device
        
        # Add multimodal token to tokenizer if not present
        if multimodal_token not in tokenizer.get_vocab():
            self.tokenizer.add_tokens([multimodal_token])
            self.multimodal_token_id = tokenizer.convert_tokens_to_ids(multimodal_token)
            logger.info(f"Added multimodal token: {multimodal_token} (ID: {self.multimodal_token_id})")
        else:
            self.multimodal_token_id = tokenizer.convert_tokens_to_ids(multimodal_token)
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Process text input and tokenize.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with tokenized text information
        """
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "text_length": encoded["input_ids"].shape[1],
        }
    
    def process_multimodal_embeddings(
        self,
        multimodal_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process multimodal embeddings and format them for the model.
        
        Args:
            multimodal_data: Dictionary containing multimodal embeddings
            
        Returns:
            Processed multimodal data
        """
        if "multimodal_embeddings" not in multimodal_data:
            return {"multimodal_embeddings": {}}
        
        embeddings = multimodal_data["multimodal_embeddings"]
        processed_embeddings = {}
        
        for modality_name, embedding_list in embeddings.items():
            # Ensure embeddings are tensors and properly formatted for batch processing
            processed_list = []
            
            if isinstance(embedding_list, list):
                for emb in embedding_list:
                    if isinstance(emb, torch.Tensor):
                        processed_list.append(emb.to(self.device))
                    else:
                        processed_list.append(torch.tensor(emb, device=self.device, dtype=torch.float32))
            else:
                # Single embedding
                if isinstance(embedding_list, torch.Tensor):
                    processed_list = [embedding_list.to(self.device)]
                else:
                    processed_list = [torch.tensor(embedding_list, device=self.device, dtype=torch.float32)]
            
            processed_embeddings[modality_name] = processed_list
        
        # Convert to batch format (wrap in list for single item)
        batch_formatted_embeddings = {}
        for modality_name, embedding_list in processed_embeddings.items():
            # For single input processing, wrap the embedding list in another list to create batch dimension
            batch_formatted_embeddings[modality_name] = [embedding_list]
        
        return {"multimodal_embeddings": batch_formatted_embeddings}
    
    def create_multimodal_input(
        self,
        text: str,
        multimodal_data: Optional[Dict[str, Any]] = None,
        insert_multimodal_tokens: bool = True,
    ) -> Dict[str, Any]:
        """
        Create complete multimodal input for the model.
        
        Args:
            text: Input text
            multimodal_data: Multimodal embeddings data
            insert_multimodal_tokens: Whether to insert multimodal tokens in text
            
        Returns:
            Complete input dictionary for the model
        """
        # Process text
        text_data = self.process_text(text)
        
        # Process multimodal data
        if multimodal_data is None:
            multimodal_data = {"multimodal_embeddings": {}}
        
        processed_multimodal = self.process_multimodal_embeddings(multimodal_data)
        
        # Count total multimodal embeddings
        total_multimodal_tokens = 0
        for modality_embeddings in processed_multimodal["multimodal_embeddings"].values():
            total_multimodal_tokens += len(modality_embeddings)
        
        # Create input dictionary
        model_input = {
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "multimodal_embeddings": processed_multimodal["multimodal_embeddings"],
            "text_length": text_data["text_length"],
            "multimodal_length": total_multimodal_tokens,
        }
        
        # Optionally insert multimodal tokens in text
        if insert_multimodal_tokens and total_multimodal_tokens > 0:
            model_input = self._insert_multimodal_tokens(model_input)
        
        return model_input
    
    def _insert_multimodal_tokens(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert multimodal tokens into the text sequence.
        
        This creates placeholder tokens in the text sequence where multimodal
        embeddings will be inserted during model forward pass.
        """
        input_ids = model_input["input_ids"]
        attention_mask = model_input["attention_mask"]
        multimodal_length = model_input["multimodal_length"]
        
        if multimodal_length == 0:
            return model_input
        
        # Create multimodal tokens
        multimodal_tokens = torch.full(
            (input_ids.shape[0], multimodal_length),
            self.multimodal_token_id,
            device=self.device,
            dtype=input_ids.dtype,
        )
        
        multimodal_attention = torch.ones(
            (attention_mask.shape[0], multimodal_length),
            device=self.device,
            dtype=attention_mask.dtype,
        )
        
        # Insert multimodal tokens at the beginning of the sequence
        # This is a simple strategy - more sophisticated insertion strategies can be implemented
        new_input_ids = torch.cat([multimodal_tokens, input_ids], dim=1)
        new_attention_mask = torch.cat([multimodal_attention, attention_mask], dim=1)
        
        # Truncate if exceeds max length
        if new_input_ids.shape[1] > self.max_length:
            new_input_ids = new_input_ids[:, :self.max_length]
            new_attention_mask = new_attention_mask[:, :self.max_length]
        
        model_input.update({
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "multimodal_token_positions": list(range(multimodal_length)),
        })
        
        return model_input
    
    def batch_process(
        self,
        inputs: List[Dict[str, Any]],
        padding: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a batch of multimodal inputs.
        
        Args:
            inputs: List of input dictionaries
            padding: Whether to pad sequences to same length
            
        Returns:
            Batched input dictionary
        """
        processed_inputs = []
        
        for input_item in inputs:
            text = input_item.get("text", "")
            multimodal_data = input_item.get("multimodal_data", None)
            
            processed = self.create_multimodal_input(text, multimodal_data)
            processed_inputs.append(processed)
        
        if not processed_inputs:
            return {}
        
        # Batch multimodal embeddings
        batched_multimodal = self._batch_multimodal_embeddings(processed_inputs)
        
        if padding:
            return self._pad_and_batch(processed_inputs, batched_multimodal)
        else:
            return self._simple_batch(processed_inputs, batched_multimodal)
    
    def _batch_multimodal_embeddings(
        self,
        processed_inputs: List[Dict[str, Any]]
    ) -> Dict[str, List[List[torch.Tensor]]]:
        """Batch multimodal embeddings across inputs."""
        # Collect all unique modalities
        all_modalities = set()
        for input_item in processed_inputs:
            all_modalities.update(input_item["multimodal_embeddings"].keys())
        
        # Create batched structure
        batched_embeddings = {}
        for modality in all_modalities:
            batched_embeddings[modality] = []
            
            for input_item in processed_inputs:
                modality_embeddings = input_item["multimodal_embeddings"].get(modality, [])
                batched_embeddings[modality].append(modality_embeddings)
        
        return batched_embeddings
    
    def _pad_and_batch(
        self,
        processed_inputs: List[Dict[str, Any]],
        batched_multimodal: Dict[str, List[List[torch.Tensor]]],
    ) -> Dict[str, Any]:
        """Pad and batch all inputs to same length."""
        # Find maximum lengths
        max_length = max(input_item["input_ids"].shape[1] for input_item in processed_inputs)
        max_length = min(max_length, self.max_length)
        
        # Pad and stack text inputs
        batched_input_ids = []
        batched_attention_masks = []
        
        for input_item in processed_inputs:
            input_ids = input_item["input_ids"]
            attention_mask = input_item["attention_mask"]
            
            # Pad sequences
            current_length = input_ids.shape[1]
            if current_length < max_length:
                padding_length = max_length - current_length
                
                # Pad input_ids with pad_token_id
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                input_ids = torch.cat([
                    input_ids,
                    torch.full((1, padding_length), pad_token_id, device=self.device)
                ], dim=1)
                
                # Pad attention_mask with zeros
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros((1, padding_length), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)
            elif current_length > max_length:
                # Truncate
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            
            batched_input_ids.append(input_ids)
            batched_attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.cat(batched_input_ids, dim=0),
            "attention_mask": torch.cat(batched_attention_masks, dim=0),
            "multimodal_embeddings": batched_multimodal,
            "batch_size": len(processed_inputs),
        }
    
    def _simple_batch(
        self,
        processed_inputs: List[Dict[str, Any]],
        batched_multimodal: Dict[str, List[List[torch.Tensor]]],
    ) -> Dict[str, Any]:
        """Simple batching without padding."""
        return {
            "input_ids": [input_item["input_ids"] for input_item in processed_inputs],
            "attention_mask": [input_item["attention_mask"] for input_item in processed_inputs],
            "multimodal_embeddings": batched_multimodal,
            "batch_size": len(processed_inputs),
        }
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer instance."""
        return self.tokenizer
    
    def get_multimodal_token_id(self) -> int:
        """Get the multimodal token ID."""
        return self.multimodal_token_id


class MultimodalInputFormatter:
    """
    Utility class for formatting multimodal inputs in different styles.
    """
    
    @staticmethod
    def format_interleaved(
        text_segments: List[str],
        multimodal_data: Dict[str, List[torch.Tensor]],
        multimodal_positions: List[int],
    ) -> Dict[str, Any]:
        """
        Format inputs with interleaved text and multimodal content.
        
        Args:
            text_segments: List of text segments
            multimodal_data: Multimodal embeddings
            multimodal_positions: Positions where to insert multimodal content
            
        Returns:
            Formatted input dictionary
        """
        # Create interleaved text
        interleaved_text = ""
        multimodal_idx = 0
        
        for i, text_segment in enumerate(text_segments):
            interleaved_text += text_segment
            
            if i in multimodal_positions and multimodal_idx < len(multimodal_data):
                interleaved_text += " <|multimodal|> "
                multimodal_idx += 1
        
        return {
            "text": interleaved_text,
            "multimodal_data": {"multimodal_embeddings": multimodal_data},
        }
    
    @staticmethod
    def format_conversation(
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Format conversational inputs with multimodal content.
        
        Args:
            messages: List of message dictionaries with role, content, and optional multimodal data
            
        Returns:
            Formatted conversation input
        """
        conversation_text = ""
        all_multimodal_data = {}
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            multimodal_data = message.get("multimodal_data", {})
            
            conversation_text += f"{role}: {content}\n"
            
            # Merge multimodal data
            if "multimodal_embeddings" in multimodal_data:
                for modality, embeddings in multimodal_data["multimodal_embeddings"].items():
                    if modality not in all_multimodal_data:
                        all_multimodal_data[modality] = []
                    all_multimodal_data[modality].extend(embeddings)
        
        return {
            "text": conversation_text.strip(),
            "multimodal_data": {"multimodal_embeddings": all_multimodal_data},
        } 