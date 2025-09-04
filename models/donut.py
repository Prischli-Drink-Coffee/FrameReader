"""
Donut model implementation using base classes.
Refactored for better OOP design and modularity.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel, 
    DonutProcessor,
    AutoModel
)

from ..core.base import BaseEncoder, BaseDecoder, BaseOCRModel

logger = logging.getLogger(__name__)


class DonutEncoder(BaseEncoder):
    """Vision encoder for Donut model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        encoder_name = config.get('encoder_name', 'facebook/deit-base-distilled-patch16-224')
        self.vision_model = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        if hasattr(config, 'enable_gradient_checkpointing') and config['enable_gradient_checkpointing']:
            if hasattr(self.vision_model, 'gradient_checkpointing_enable'):
                self.vision_model.gradient_checkpointing_enable()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to feature representations."""
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state
    
    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self.hidden_size


class DonutDecoder(BaseDecoder):
    """Text decoder for Donut model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        decoder_name = config.get('decoder_name', 'facebook/mbart-large-50')
        self.text_model = AutoModel.from_pretrained(decoder_name)
        
        if hasattr(self.text_model, 'decoder'):
            self.decoder_model = self.text_model.decoder
        else:
            self.decoder_model = self.text_model
        
        self.vocab_size = getattr(self.text_model.config, 'vocab_size', 50265)
        self.hidden_size = self.text_model.config.hidden_size
        
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        if hasattr(config, 'enable_gradient_checkpointing') and config['enable_gradient_checkpointing']:
            if hasattr(self.decoder_model, 'gradient_checkpointing_enable'):
                self.decoder_model.gradient_checkpointing_enable()
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode encoder states to text logits."""
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        decoder_outputs = self.decoder_model(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids right for decoder input."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config.get('decoder_start_token_id', 0)
        return shifted_input_ids
    
    @property
    def output_dim(self) -> int:
        """Output vocabulary dimension."""
        return self.vocab_size


class DonutOCRModel(BaseOCRModel):
    """Complete Donut OCR model with enhanced architecture."""
    
    def __init__(
        self,
        encoder: DonutEncoder,
        decoder: DonutDecoder, 
        config: Dict[str, Any]
    ):
        super().__init__(encoder, decoder, config)
        
        self.processor = None
        self.max_length = config.get('max_length', 768)
        self.task_start_token = config.get('task_start_token', '<s>')
        self.prompt_end_token = config.get('prompt_end_token', self.task_start_token)
        
        self._setup_special_tokens()
    
    def _setup_special_tokens(self) -> None:
        """Configure special tokens for the model."""
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
            
            if self.task_start_token not in tokenizer.get_vocab():
                special_tokens_dict = {"additional_special_tokens": [self.task_start_token]}
                tokenizer.add_special_tokens(special_tokens_dict)
                self.decoder.text_model.resize_token_embeddings(len(tokenizer))
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass through the model."""
        
        encoder_outputs = self.encoder(pixel_values)
        
        projected_features = self.projection(encoder_outputs)
        
        logits = self.decoder(
            encoder_hidden_states=projected_features,
            labels=labels
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1)
            )
            
            loss = self._apply_regularization(loss, logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'encoder_last_hidden_state': encoder_outputs
        }
    
    def _apply_regularization(
        self,
        base_loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply custom regularization penalties."""
        if self.processor is None:
            return base_loss
        
        tokenizer = self.processor.tokenizer
        batch_size, seq_len = labels.shape
        
        probs = F.softmax(logits, dim=-1)
        
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        
        eos_positions = (labels == eos_token_id)
        has_eos = eos_positions.any(dim=1)
        
        if has_eos.any():
            positions = torch.arange(seq_len, device=labels.device).expand(batch_size, seq_len)
            first_eos_indices = torch.full((batch_size,), seq_len-1, device=labels.device)
            first_eos_indices[has_eos] = torch.argmax(eos_positions[has_eos].float(), dim=1)
            
            after_eos_mask = (positions >= first_eos_indices.unsqueeze(1)).float()
            non_pad_probs = torch.sum(probs, dim=2) - probs[:, :, pad_token_id]
            post_eos_penalty = torch.sum(non_pad_probs * after_eos_mask) / batch_size
            
            penalty_weight = 0.05
            base_loss = base_loss + penalty_weight * post_eos_penalty
        
        return base_loss
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: int = 3,
        max_length: Optional[int] = None,
        return_json: bool = False,
        **kwargs
    ) -> Union[List[str], List[Dict]]:
        """Generate text from images."""
        if self.processor is None:
            raise ValueError("Processor not initialized. Call set_processor() first.")
        
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        encoder_outputs = self.encoder(pixel_values)
        projected_features = self.projection(encoder_outputs)
        
        batch_size = pixel_values.shape[0]
        decoder_start_token_id = getattr(self.decoder.text_model.config, 'decoder_start_token_id', 0)
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            device=self.device
        )
        
        generated_ids = self._beam_search_generate(
            encoder_hidden_states=projected_features,
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            max_length=max_length,
            **kwargs
        )
        
        decoded_sequences = []
        json_outputs = []
        
        for seq in self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True):
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            decoded_sequences.append(seq)
            
            if return_json:
                try:
                    json_output = self._token_to_json(seq)
                    json_outputs.append(json_output)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    json_outputs.append({"error": "Failed to parse JSON"})
        
        return json_outputs if return_json else decoded_sequences
    
    def _beam_search_generate(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        num_beams: int,
        max_length: int,
        **kwargs
    ) -> torch.Tensor:
        """Simplified beam search generation."""
        current_ids = decoder_input_ids
        
        for _ in range(max_length - 1):
            logits = self.decoder(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=current_ids
            )
            
            next_token_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
            
            if self.processor and (next_token_ids == self.processor.tokenizer.eos_token_id).all():
                break
        
        return current_ids
    
    def _token_to_json(self, tokens: str, is_inner_value: bool = False) -> Union[Dict, List]:
        """Convert token sequence to JSON structure."""
        output = dict()
        
        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
                
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    
                    if r"<s_" in content and r"</s_" in content:
                        value = self._token_to_json(content, is_inner_value=True)
                        if value:
                            output[key] = value[0] if len(value) == 1 else value
                    else:
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (self.processor and 
                                leaf in self.processor.tokenizer.get_added_vocab() and 
                                leaf.startswith("<") and leaf.endswith("/>")):
                                leaf = leaf[1:-2]
                            output[key].append(leaf)
                        
                        if len(output[key]) == 1:
                            output[key] = output[key][0]
                
                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens.startswith(r"<sep/>"):
                    return [output] + self._token_to_json(tokens[6:], is_inner_value=True)
        
        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}
    
    def set_processor(self, processor: DonutProcessor) -> None:
        """Set the processor for tokenization."""
        self.processor = processor
        self._setup_special_tokens()
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        **kwargs
    ) -> "DonutOCRModel":
        """Load model from pretrained directory."""
        model_dir = Path(model_dir)
        
        with open(model_dir / "model_config.json", "r") as f:
            config = json.load(f)
        
        encoder_config = {**config, 'encoder_name': config.get('encoder_name')}
        decoder_config = {**config, 'decoder_name': config.get('decoder_name')}
        
        encoder = DonutEncoder(encoder_config)
        decoder = DonutDecoder(decoder_config)
        
        model = cls(encoder, decoder, config)
        
        encoder.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu"))
        decoder.load_state_dict(torch.load(model_dir / "decoder.pt", map_location="cpu"))
        
        if (model_dir / "projection.pt").exists():
            model.projection.load_state_dict(torch.load(model_dir / "projection.pt", map_location="cpu"))
        
        try:
            processor = DonutProcessor.from_pretrained(model_dir)
            model.set_processor(processor)
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
        
        return model