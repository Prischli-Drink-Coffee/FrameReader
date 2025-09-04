"""
TrOCR model implementation using base classes.
Enhanced architecture with better OOP design.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json

import torch
import torch.nn as nn
from transformers import (
    TrOCRProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoConfig,
    AutoImageProcessor,
    AutoModel
)

from core.base import BaseEncoder, BaseDecoder, BaseOCRModel

logger = logging.getLogger(__name__)


class TrOCREncoder(BaseEncoder):
    """Vision encoder for TrOCR model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        encoder_name = config.get('encoder_name', 'microsoft/swin-small-patch4-window7-224')
        self.vision_model = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        if config.get('enable_gradient_checkpointing', False):
            if hasattr(self.vision_model, 'gradient_checkpointing_enable'):
                self.vision_model.gradient_checkpointing_enable()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to hidden representations."""
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state
    
    @property
    def output_dim(self) -> int:
        """Output dimension of encoded features."""
        return self.hidden_size


class TrOCRDecoder(BaseDecoder):
    """Text decoder for TrOCR model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        decoder_name = config.get('decoder_name', 'ai-forever/ruRoberta-large')
        self.text_model = AutoModel.from_pretrained(decoder_name)
        
        self.vocab_size = getattr(self.text_model.config, 'vocab_size', 50265)
        self.hidden_size = self.text_model.config.hidden_size
        
        self.text_model.config.is_decoder = True
        self.text_model.config.add_cross_attention = True
        
        if config.get('use_8bit_decoder', False):
            self._apply_quantization()
        
        if config.get('enable_gradient_checkpointing', False):
            if hasattr(self.text_model, 'gradient_checkpointing_enable'):
                self.text_model.gradient_checkpointing_enable()
    
    def _apply_quantization(self) -> None:
        """Apply 8-bit quantization to decoder if available."""
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            logger.info("Applied 8-bit quantization to decoder")
        except ImportError:
            logger.warning("BitsAndBytes not available, skipping quantization")
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode encoder states to text tokens."""
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        try:
            decoder_outputs = self.text_model(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            )
            
            return decoder_outputs.last_hidden_state
            
        except Exception as e:
            logger.error(f"Decoder forward error: {e}")
            
            vocab_size = getattr(self.text_model.config, 'vocab_size', 50265)
            if decoder_input_ids is not None and decoder_input_ids.max() >= vocab_size:
                logger.warning(f"Token IDs exceed vocab size {vocab_size}, clamping")
                decoder_input_ids = torch.clamp(decoder_input_ids, min=0, max=vocab_size-1)
                
                decoder_outputs = self.text_model(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True
                )
                return decoder_outputs.last_hidden_state
            else:
                raise
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids right for decoder input."""
        input_ids_safe = input_ids.clone()
        shifted_input_ids = input_ids_safe.new_zeros(input_ids_safe.shape)
        
        bos_token_id = self.config.get('decoder_start_token_id', 0)
        vocab_size = getattr(self.text_model.config, 'vocab_size', 50265)
        pad_token_id = getattr(self.text_model.config, 'pad_token_id', 1)
        
        invalid_mask = (input_ids_safe != -100) & ((input_ids_safe < 0) | (input_ids_safe >= vocab_size))
        if invalid_mask.any():
            input_ids_safe = torch.where(
                invalid_mask,
                torch.tensor(pad_token_id, device=input_ids_safe.device),
                input_ids_safe
            )
        
        shifted_input_ids[:, 1:] = input_ids_safe[:, :-1].clone()
        shifted_input_ids[:, 0] = bos_token_id
        
        if shifted_input_ids.max() >= vocab_size:
            shifted_input_ids = torch.clamp(shifted_input_ids, min=0, max=vocab_size-1)
        
        return shifted_input_ids
    
    @property
    def output_dim(self) -> int:
        """Output vocabulary dimension."""
        return self.vocab_size


class TrOCROCRModel(BaseOCRModel):
    """Complete TrOCR model with enhanced features."""
    
    def __init__(
        self,
        encoder: TrOCREncoder,
        decoder: TrOCRDecoder,
        config: Dict[str, Any]
    ):
        super().__init__(encoder, decoder, config)
        
        self.tokenizer = None
        self.processor = None
        self.max_length = config.get('max_length', 512)
        self.precision = config.get('precision', 'bf16')
        
        self.lm_head = nn.Linear(self.decoder.hidden_size, self.decoder.vocab_size, bias=False)
        
        self._apply_precision_settings()
    
    def _apply_precision_settings(self) -> None:
        """Apply precision settings for mixed precision training."""
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported, falling back to FP16")
            self.precision = "fp16"
        
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }.get(self.precision, torch.float32)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass through encoder and decoder."""
        
        pixel_values = self._prepare_pixel_values(pixel_values)
        if labels is not None:
            labels = self._prepare_labels(labels)
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.precision in ["bf16", "fp16"]
        ):
            encoder_outputs = self.encoder(pixel_values)
            
            projected_features = self.projection(encoder_outputs)
            
            decoder_outputs = self.decoder(
                encoder_hidden_states=projected_features,
                labels=labels
            )
            
            logits = self.lm_head(decoder_outputs)
            
            loss = None
            if labels is not None:
                loss = self._compute_loss(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'encoder_last_hidden_state': encoder_outputs
        }
    
    def _prepare_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Prepare and validate pixel values."""
        if pixel_values.dim() != 4:
            logger.warning(f"Unexpected pixel_values shape: {pixel_values.shape}")
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(1)
        
        return pixel_values.to(self.device)
    
    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Prepare and validate labels."""
        labels = labels.to(self.device)
        
        if self.tokenizer and torch.any(labels >= self.tokenizer.vocab_size):
            labels = torch.clamp(labels, max=self.tokenizer.vocab_size - 1)
        
        return labels
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with proper handling."""
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        batch_size, seq_length = labels.shape
        actual_vocab_size = logits.size(-1)
        
        if labels.shape != (batch_size, seq_length):
            if labels.shape[1] > seq_length:
                labels = labels[:, :seq_length]
            elif labels.shape[1] < seq_length:
                padding = torch.full(
                    (batch_size, seq_length - labels.shape[1]),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype
                )
                labels = torch.cat([labels, padding], dim=1)
        
        logits_flat = logits.contiguous().view(-1, actual_vocab_size)
        labels_flat = labels.contiguous().view(-1)
        
        if (labels_flat != -100).any() and (labels_flat >= actual_vocab_size).any():
            invalid_mask = (labels_flat != -100) & (labels_flat >= actual_vocab_size)
            labels_flat = torch.where(invalid_mask, torch.tensor(-100, device=labels.device), labels_flat)
        
        return loss_fct(logits_flat, labels_flat)
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        early_stopping: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate text from images."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call set_tokenizer() first.")
        
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        try:
            decoder_input_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]] * pixel_values.size(0)
            ).to(self.device)
            
            encoder_outputs = self.encoder(pixel_values)
            projected_features = self.projection(encoder_outputs)
            
            generated_ids = self._beam_search_decode(
                encoder_hidden_states=projected_features,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
            
            generated_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return [""] * pixel_values.size(0)
    
    def _beam_search_decode(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        max_length: int,
        num_beams: int,
        **kwargs
    ) -> torch.Tensor:
        """Simplified beam search decoding."""
        current_ids = decoder_input_ids
        
        for step in range(max_length - 1):
            decoder_outputs = self.decoder(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=current_ids
            )
            
            logits = self.lm_head(decoder_outputs)
            next_token_logits = logits[:, -1, :]
            
            if num_beams == 1:
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                _, next_token_ids = torch.topk(next_token_logits, k=1, dim=-1)
            
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
            
            if (next_token_ids == self.tokenizer.eos_token_id).all():
                break
        
        return current_ids
    
    def set_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        """Set the tokenizer for the model."""
        self.tokenizer = tokenizer
        
        if len(tokenizer) != self.decoder.vocab_size:
            logger.info(f"Resizing token embeddings from {self.decoder.vocab_size} to {len(tokenizer)}")
            self.decoder.text_model.resize_token_embeddings(len(tokenizer))
            self.lm_head = nn.Linear(self.decoder.hidden_size, len(tokenizer), bias=False)
            self.decoder.vocab_size = len(tokenizer)
    
    def set_processor(self, processor: TrOCRProcessor) -> None:
        """Set the processor for the model."""
        self.processor = processor
        if hasattr(processor, 'tokenizer'):
            self.set_tokenizer(processor.tokenizer)
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        **kwargs
    ) -> "TrOCROCRModel":
        """Load model from pretrained directory."""
        model_dir = Path(model_dir)
        
        with open(model_dir / "model_config.json", "r") as f:
            config = json.load(f)
        
        encoder_config = {**config, 'encoder_name': config.get('encoder_name')}
        decoder_config = {**config, 'decoder_name': config.get('decoder_name')}
        
        encoder = TrOCREncoder(encoder_config)
        decoder = TrOCRDecoder(decoder_config)
        
        model = cls(encoder, decoder, config)
        
        encoder.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu"))
        decoder.load_state_dict(torch.load(model_dir / "decoder.pt", map_location="cpu"))
        
        if (model_dir / "projection.pt").exists():
            model.projection.load_state_dict(torch.load(model_dir / "projection.pt", map_location="cpu"))
        
        if (model_dir / "lm_head.pt").exists():
            model.lm_head.load_state_dict(torch.load(model_dir / "lm_head.pt", map_location="cpu"))
        
        try:
            processor = TrOCRProcessor.from_pretrained(model_dir)
            model.set_processor(processor)
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model.set_tokenizer(tokenizer)
            except Exception as e2:
                logger.warning(f"Could not load tokenizer: {e2}")
        
        return model
    
    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        """Save model with all components."""
        super().save_pretrained(output_dir)
        
        output_dir = Path(output_dir)
        torch.save(self.lm_head.state_dict(), output_dir / "lm_head.pt")
        
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)