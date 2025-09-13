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

from core.base import BaseEncoder, BaseDecoder, BaseOCRModel

logger = logging.getLogger(__name__)


class DonutEncoder(BaseEncoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        encoder_name = config.get('encoder_name', 'facebook/deit-base-distilled-patch16-224')
        logger.info(f"DonutEncoder.__init__ called with encoder_name: '{encoder_name}'")
        logger.info(f"Full encoder config: {config}")
        
        if encoder_name is None:
            logger.warning(f"encoder_name is None, using default: 'facebook/deit-base-distilled-patch16-224'")
            encoder_name = 'facebook/deit-base-distilled-patch16-224'
        
        self.vision_model = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        if config.get('enable_gradient_checkpointing'):
            if hasattr(self.vision_model, 'gradient_checkpointing_enable'):
                self.vision_model.gradient_checkpointing_enable()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state
    
    @property
    def output_dim(self) -> int:
        return self.hidden_size


class DonutDecoder(BaseDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        decoder_name = config.get('decoder_name', 'facebook/mbart-large-50')
        logger.info(f"DonutDecoder.__init__ called with decoder_name: '{decoder_name}'")
        logger.info(f"Full decoder config: {config}")
        
        if decoder_name is None:
            logger.warning(f"decoder_name is None, using default: 'facebook/mbart-large-50'")
            decoder_name = 'facebook/mbart-large-50'
        
        self.text_model = AutoModel.from_pretrained(decoder_name)
        
        self.decoder_model = getattr(self.text_model, 'decoder', self.text_model)
        self.vocab_size = getattr(self.text_model.config, 'vocab_size', 50265)
        self.hidden_size = self.text_model.config.hidden_size
        
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        if config.get('enable_gradient_checkpointing'):
            if hasattr(self.decoder_model, 'gradient_checkpointing_enable'):
                self.decoder_model.gradient_checkpointing_enable()
    
    def forward(self, encoder_hidden_states: torch.Tensor, decoder_input_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        decoder_outputs = self.decoder_model(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        return self.lm_head(decoder_outputs.last_hidden_state)
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config.get('decoder_start_token_id', 0)
        return shifted_input_ids
    
    @property
    def output_dim(self) -> int:
        return self.vocab_size


class DonutOCRModel(BaseOCRModel):
    def __init__(self, encoder: DonutEncoder, decoder: DonutDecoder, config: Dict[str, Any]):
        super().__init__(encoder, decoder, config)
        
        self.processor = None
        self.max_length = config.get('max_length', 768)
        self.task_start_token = config.get('task_start_token', '<s>')
        self.prompt_end_token = config.get('prompt_end_token', self.task_start_token)
    
    def _setup_special_tokens(self) -> None:
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
            
            if self.task_start_token not in tokenizer.get_vocab():
                special_tokens_dict = {"additional_special_tokens": [self.task_start_token]}
                tokenizer.add_special_tokens(special_tokens_dict)
                self.decoder.text_model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(pixel_values)
        projected_features = self.projection(encoder_outputs)
        logits = self.decoder(encoder_hidden_states=projected_features, labels=labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
            loss = self._apply_regularization(loss, logits, labels)
        
        return {'loss': loss, 'logits': logits, 'encoder_last_hidden_state': encoder_outputs}
    
    def _apply_regularization(self, base_loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
            
            base_loss = base_loss + 0.05 * post_eos_penalty
        
        return base_loss
    
    def generate(self, pixel_values: torch.Tensor, num_beams: int = 3, max_length: Optional[int] = None, return_json: bool = False, **kwargs) -> Union[List[str], List[Dict]]:
        if self.processor is None:
            raise ValueError("Processor not initialized. Call set_processor() first.")
        
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        encoder_outputs = self.encoder(pixel_values)
        projected_features = self.projection(encoder_outputs)
        
        batch_size = pixel_values.shape[0]
        decoder_start_token_id = getattr(self.decoder.text_model.config, 'decoder_start_token_id', 0)
        
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=self.device)
        
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
                    json_outputs.append(self._token_to_json(seq))
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    json_outputs.append({"error": "Failed to parse JSON"})
        
        return json_outputs if return_json else decoded_sequences
    
    def _beam_search_generate(self, encoder_hidden_states: torch.Tensor, decoder_input_ids: torch.Tensor, num_beams: int, max_length: int, **kwargs) -> torch.Tensor:
        current_ids = decoder_input_ids
        
        for _ in range(max_length - 1):
            logits = self.decoder(encoder_hidden_states=encoder_hidden_states, decoder_input_ids=current_ids)
            next_token_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
            
            if self.processor and (next_token_ids == self.processor.tokenizer.eos_token_id).all():
                break
        
        return current_ids
    
    def _token_to_json(self, tokens: str, is_inner_value: bool = False) -> Union[Dict, List]:
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
        
        return [output] if is_inner_value and len(output) else output or {"text_sequence": tokens}
    
    def set_processor(self, processor: DonutProcessor) -> None:
        self.processor = processor
        self._setup_special_tokens()
    
    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path], token: Optional[str] = None, **kwargs):
        model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
        
        logger.info(f"DonutOCRModel.from_pretrained called with model_dir: '{model_dir}' (type: {type(model_dir)})")
        logger.info(f"model_dir exists: {model_dir.exists() if isinstance(model_dir, Path) else 'N/A'}")
        
        if isinstance(model_dir, Path) and model_dir.exists():
            logger.info(f"Loading local model from: {model_dir}")
            with open(model_dir / "model_config.json", "r") as f:
                config = json.load(f)
            
            logger.info(f"Loaded config from file: {config}")
            
            encoder = DonutEncoder({**config, 'encoder_name': config.get('encoder_name')})
            decoder = DonutDecoder({**config, 'decoder_name': config.get('decoder_name')})
            model = cls(encoder, decoder, config)
            
            encoder.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu"))
            decoder.load_state_dict(torch.load(model_dir / "decoder.pt", map_location="cpu"))
            
            if (model_dir / "projection.pt").exists():
                model.projection.load_state_dict(torch.load(model_dir / "projection.pt", map_location="cpu"))
            
            try:
                processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
                model.set_processor(processor)
            except Exception as e:
                logger.warning(f"Could not load processor: {e}")
            
            return model
        
        else:
            logger.info(f"Loading model from Hugging Face: {model_dir}")
            logger.info(f"Token provided: {token is not None}")
            
            try:
                hf_model = VisionEncoderDecoderModel.from_pretrained(model_dir, token=token, **kwargs)
                logger.info(f"Successfully loaded HF model: {type(hf_model)}")
            except Exception as e:
                logger.error(f"Failed to load HF model from '{model_dir}': {e}")
                raise
            
            config = hf_model.config.to_dict()
            config.update({'model_name_or_path': str(model_dir), 'hf_token': token})
            
            logger.info(f"Created config for HF model: model_name_or_path='{config.get('model_name_or_path')}'")
            
            encoder = DonutEncoder({**config, 'encoder_name': None})
            encoder.vision_model = hf_model.encoder

            decoder = DonutDecoder({**config, 'decoder_name': None})
            decoder.text_model = hf_model.decoder
            
            model = cls(encoder, decoder, config)
            
            try:
                processor = DonutProcessor.from_pretrained(model_dir, token=token, use_fast=True)
                model.set_processor(processor)
                logger.info(f"Successfully loaded processor from HF: {type(processor)}")
            except Exception as e:
                logger.warning(f"Could not load processor from HF: {e}")
            
            return model