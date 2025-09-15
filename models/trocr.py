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
    AutoModel,
    VisionEncoderDecoderModel,
    GPT2LMHeadModel,
    GPT2Config
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from core.base import BaseEncoder, BaseDecoder, BaseOCRModel

logger = logging.getLogger(__name__)

ENCODER_MODELS = {
    "base": "microsoft/swin-small-patch4-window7-224",
    "large": "microsoft/swin-large-patch4-window12-384-in22k",
    "xlarge": "microsoft/swin-large-patch4-window12-384-in22k"
}

DECODER_MODELS = {
    "base": "ai-forever/ruRoberta-large",
    "large": "ai-forever/ruBert-large", 
    "xlarge": "ai-forever/ruBert-large"
}


class TrOCREncoder(BaseEncoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        encoder_name = config.get('encoder_name', ENCODER_MODELS['base'])
        
        try:
            self.vision_encoder = AutoModel.from_pretrained(encoder_name)
        except Exception as e:
            logger.warning(f"Cannot load encoder {encoder_name}: {e}, using default")
            self.vision_encoder = AutoModel.from_pretrained(ENCODER_MODELS['base'])
        
        self._output_dim = getattr(self.vision_encoder.config, 'hidden_size', 768)
        
        if config.get('enable_gradient_checkpointing', False):
            if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                self.vision_encoder.gradient_checkpointing_enable()
        
        if config.get('freeze_encoder', False):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state


class TrOCRDecoder(BaseDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        decoder_name = config.get('decoder_name', DECODER_MODELS['base'])
        
        try:
            decoder_config = AutoConfig.from_pretrained(decoder_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            
            if hasattr(decoder_config, 'use_cache'):
                decoder_config.use_cache = not config.get('enable_gradient_checkpointing', False)
            
            self.text_decoder = AutoModel.from_pretrained(decoder_name, config=decoder_config)
        except Exception as e:
            logger.warning(f"Cannot load decoder {decoder_name}: {e}, using GPT2")
            decoder_config = GPT2Config.from_pretrained('gpt2')
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2', config=decoder_config)
        
        self._output_dim = getattr(self.text_decoder.config, 'hidden_size', 768)
        self.vocab_size = getattr(self.text_decoder.config, 'vocab_size', 50257)
        
        if not hasattr(self.text_decoder, 'lm_head'):
            self.lm_head = nn.Linear(self._output_dim, self.vocab_size, bias=False)
        else:
            self.lm_head = self.text_decoder.lm_head
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, encoder_hidden_states: torch.Tensor, decoder_input_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Any:
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = 1
        
        outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True
        )
        
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, 'past_key_values', None),
            decoder_hidden_states=getattr(outputs, 'hidden_states', None),
            decoder_attentions=getattr(outputs, 'attentions', None),
            cross_attentions=getattr(outputs, 'cross_attentions', None)
        )


class TrOCROCRModel(BaseOCRModel):
    def __init__(self, config: Dict[str, Any], pretrained_model_path: Optional[str] = None):
        self.config = config
        self.max_length = config.get('max_length', 512)
        self.img_size = config.get('img_size', (384, 384))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = config.get('precision', 'fp32')
        
        if pretrained_model_path:
            self._load_from_pretrained(pretrained_model_path)
        else:
            self._build_custom_model(config)
        
        if hasattr(self, 'encoder') and hasattr(self, 'decoder'):
            super().__init__(self.encoder, self.decoder, config)
    
    def _load_from_pretrained(self, model_path: str):
        logger.info(f"Loading TrOCR model from {model_path}")
        
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForVision2Seq.from_pretrained(model_path)
            
            self._configure_pretrained_model()
            
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
            self.projection = nn.Identity()
            
        except Exception as e:
            logger.warning(f"Failed to load as pretrained TrOCR: {e}")
            self._build_custom_model(self.config)
    
    def _build_custom_model(self, config: Dict[str, Any]):
        encoder_name = config.get('encoder_name', None)
        decoder_name = config.get('decoder_name', None)
        encoder_size = config.get('encoder_size', 'base')
        
        if encoder_name is None:
            encoder_name = ENCODER_MODELS.get(encoder_size, ENCODER_MODELS['base'])
        if decoder_name is None:
            decoder_name = DECODER_MODELS.get(encoder_size, DECODER_MODELS['base'])
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=True)
            image_processor = AutoImageProcessor.from_pretrained(encoder_name)
            self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception as e:
            logger.warning(f"Failed to create processor: {e}, using default")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
        
        self._configure_tokenizer()
        
        encoder_config = {**config, 'encoder_name': encoder_name}
        decoder_config = {**config, 'decoder_name': decoder_name}
        
        self.encoder = TrOCREncoder(encoder_config)
        self.decoder = TrOCRDecoder(decoder_config)
        
        encoder_dim = self.encoder.output_dim
        decoder_dim = self.decoder.output_dim
        
        if encoder_dim != decoder_dim:
            self.projection = nn.Linear(encoder_dim, decoder_dim)
            logger.info(f"Created projection layer: {encoder_dim} -> {decoder_dim}")
        else:
            self.projection = nn.Identity()
        
        self._build_full_model()
    
    def _configure_tokenizer(self):
        tokenizer = self.processor.tokenizer
        
        if tokenizer.bos_token_id is None:
            if tokenizer.pad_token_id is not None:
                tokenizer.bos_token = tokenizer.pad_token
                logger.info("Set bos_token to pad_token")
            else:
                # Добавляем базовые токены, которые должны быть в любой модели
                tokenizer.add_special_tokens({
                    'bos_token': '<s>',
                    'eos_token': '</s>',
                    'pad_token': '<pad>'
                })
                logger.info("Added missing special tokens")
        
        # Обратите внимание: дополнительное изменение размеров эмбеддингов должно 
        # происходить после загрузки весов в методах _build_custom_model и _configure_pretrained_model
    
    def _configure_pretrained_model(self):
        config = self.model.config
        tokenizer = self.processor.tokenizer
        
        if not hasattr(config, 'decoder_start_token_id') or config.decoder_start_token_id is None:
            config.decoder_start_token_id = tokenizer.bos_token_id
        
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.is_encoder_decoder = True
        
        if self.config.get('enable_gradient_checkpointing', False):
            if hasattr(config, 'encoder'):
                config.encoder.use_cache = False
            if hasattr(config, 'decoder'):
                config.decoder.use_cache = False
            self.model.gradient_checkpointing_enable()
        
        vocab_size = len(tokenizer)
        current_size = self.model.decoder.get_input_embeddings().num_embeddings
        if current_size != vocab_size:
            self.model.decoder.resize_token_embeddings(vocab_size)
    
    def _build_full_model(self):
        class CustomTrOCRModel(nn.Module):
            def __init__(self, encoder, decoder, projection, config):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.projection = projection
                self.config = config
                
            def forward(self, pixel_values=None, decoder_input_ids=None, labels=None, **kwargs):
                encoder_outputs = self.encoder(pixel_values)
                projected_outputs = self.projection(encoder_outputs)
                
                decoder_outputs = self.decoder(
                    encoder_hidden_states=projected_outputs,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                
                return decoder_outputs
            
            def generate(self, pixel_values, **kwargs):
                encoder_outputs = self.encoder(pixel_values)
                projected_outputs = self.projection(encoder_outputs)
                
                batch_size = pixel_values.size(0)
                max_length = kwargs.get('max_length', 50)
                bos_token_id = kwargs.get('decoder_start_token_id', 1)
                eos_token_id = kwargs.get('eos_token_id', 2)
                
                generated_ids = torch.full(
                    (batch_size, 1), bos_token_id, 
                    device=pixel_values.device, dtype=torch.long
                )
                
                for _ in range(max_length - 1):
                    outputs = self.decoder(
                        encoder_hidden_states=projected_outputs,
                        decoder_input_ids=generated_ids
                    )
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                    
                    if (next_tokens == eos_token_id).all():
                        break
                
                return generated_ids
        
        model_config = type('Config', (), {
            'decoder_start_token_id': self.processor.tokenizer.bos_token_id,
            'pad_token_id': self.processor.tokenizer.pad_token_id,
            'eos_token_id': self.processor.tokenizer.eos_token_id,
            'vocab_size': len(self.processor.tokenizer)
        })()
        
        self.model = CustomTrOCRModel(self.encoder, self.decoder, self.projection, model_config)
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
            if pixel_values.size(0) != labels.size(0):
                min_batch = min(pixel_values.size(0), labels.size(0))
                pixel_values = pixel_values[:min_batch]
                labels = labels[:min_batch]
            
            vocab_size = len(self.processor.tokenizer)
            if torch.any(labels >= vocab_size):
                labels = torch.clamp(labels, max=vocab_size - 1)
        
        if pixel_values.dim() != 4:
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(1)
        
        if self.precision in ["bf16", "fp16"]:
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast(enabled=True, dtype=dtype, device_type=self.device.type):
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
        else:
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                logger.warning("Invalid loss detected, recalculating...")
                if hasattr(outputs, 'logits') and labels is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        outputs.logits.contiguous().view(-1, outputs.logits.size(-1)),
                        labels.contiguous().view(-1)
                    )
                    outputs.loss = loss
        
        return {
            'loss': getattr(outputs, 'loss', None),
            'logits': getattr(outputs, 'logits', None),
            'encoder_hidden_states': None
        }
    
    def generate(self, pixel_values: torch.Tensor, max_length: Optional[int] = None, num_beams: int = 4, **kwargs) -> List[str]:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        self.eval()
        with torch.no_grad():
            try:
                if hasattr(self.model, 'generate') and hasattr(self.model, 'encoder'):
                    decoder_input_ids = torch.tensor(
                        [[self.processor.tokenizer.bos_token_id]] * pixel_values.size(0)
                    ).to(self.device)
                    
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        use_cache=True,
                        **kwargs
                    )
                else:
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=max_length,
                        decoder_start_token_id=self.processor.tokenizer.bos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        **kwargs
                    )
                
                generated_text = self.processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return [""] * pixel_values.size(0)
    
    def freeze_encoder(self):
        if hasattr(self, 'model') and hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        elif hasattr(self, 'encoder'):
            for param in self.encoder.parameters():
                param.requires_grad = False
        logger.info("Encoder parameters frozen")
    
    def to_device(self, precision: str):
        self.precision = precision
        
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        if hasattr(self, 'model'):
            self.model = self.model.to(self.device, dtype=dtype)
        else:
            if hasattr(self, 'encoder'):
                self.encoder = self.encoder.to(self.device, dtype=dtype)
            if hasattr(self, 'decoder'):
                self.decoder = self.decoder.to(self.device, dtype=dtype)
            if hasattr(self, 'projection'):
                self.projection = self.projection.to(self.device, dtype=dtype)
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        if hasattr(self, 'model'):
            return [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = []
            for component in ['encoder', 'decoder', 'projection']:
                if hasattr(self, component):
                    component_obj = getattr(self, component)
                    params.extend([p for p in component_obj.parameters() if p.requires_grad])
            return params
    
    def save_pretrained(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'model') and hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            if hasattr(self, 'encoder'):
                encoder_dir = output_dir / "encoder"
                encoder_dir.mkdir(exist_ok=True)
                torch.save(self.encoder.state_dict(), encoder_dir / "pytorch_model.bin")
            
            if hasattr(self, 'decoder'):
                decoder_dir = output_dir / "decoder"
                decoder_dir.mkdir(exist_ok=True)
                torch.save(self.decoder.state_dict(), decoder_dir / "pytorch_model.bin")
            
            if hasattr(self, 'projection') and not isinstance(self.projection, nn.Identity):
                torch.save(self.projection.state_dict(), output_dir / "projection.pt")
            
            model_config = {
                "custom_model": True,
                "encoder_dim": getattr(self.encoder, 'output_dim', 768),
                "decoder_dim": getattr(self.decoder, 'output_dim', 768),
                "has_projection": not isinstance(self.projection, nn.Identity),
                "max_length": self.max_length,
                "precision": self.precision
            }
            
            with open(output_dir / "model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)
        
        if self.processor:
            self.processor.save_pretrained(output_dir)
        
        logger.info(f"TrOCR model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path], **kwargs) -> "TrOCROCRModel":
        model_dir = Path(model_dir)
        config = kwargs.get('config', {})
        
        return cls(config, pretrained_model_path=str(model_dir))
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrOCROCRModel":
        return cls(config)