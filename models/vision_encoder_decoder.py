from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoImageProcessor,
    GPT2LMHeadModel,
    GPT2Config
)

from core.base import BaseEncoder, BaseDecoder, BaseOCRModel

logger = logging.getLogger(__name__)


class CustomVisionEncoder(BaseEncoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        encoder_name = config.get('encoder_name', 'facebook/deit-base-distilled-patch16-384')
        
        if isinstance(encoder_name, str) and encoder_name.startswith(('microsoft/', 'facebook/', 'google/')):
            self.vision_encoder = AutoModel.from_pretrained(encoder_name)
        else:
            try:
                self.vision_encoder = AutoModel.from_pretrained(encoder_name)
            except:
                logger.warning(f"Cannot load encoder {encoder_name}, using default")
                self.vision_encoder = AutoModel.from_pretrained('facebook/deit-base-distilled-patch16-384')
        
        self._output_dim = getattr(self.vision_encoder.config, 'hidden_size', 768)
        
        if config.get('enable_gradient_checkpointing', False):
            if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                self.vision_encoder.gradient_checkpointing_enable()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state


class CustomTextDecoder(BaseDecoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        decoder_name = config.get('decoder_name', 'microsoft/DialoGPT-medium')
        
        try:
            decoder_config = AutoConfig.from_pretrained(decoder_name)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            self.text_decoder = GPT2LMHeadModel.from_pretrained(decoder_name, config=decoder_config)
        except:
            logger.warning(f"Cannot load decoder {decoder_name}, using default GPT2")
            decoder_config = GPT2Config.from_pretrained('gpt2')
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2', config=decoder_config)
        
        self._output_dim = getattr(self.text_decoder.config, 'hidden_size', 768)
        
        if hasattr(self.text_decoder.config, 'vocab_size'):
            self.vocab_size = self.text_decoder.config.vocab_size
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, encoder_hidden_states: torch.Tensor, decoder_input_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if decoder_input_ids is None:
            batch_size = encoder_hidden_states.size(0)
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=encoder_hidden_states.device)
        
        outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            use_cache=False
        )
        return outputs


class CustomVisionEncoderDecoderModel(BaseOCRModel):
    """
    Кастомная модель Vision Encoder Decoder, позволяющая объединять любые энкодеры и декодеры
    """
    def __init__(self, config: Dict[str, Any], pretrained_model_path: Optional[str] = None):
        self.config = config
        self.max_length = config.get('max_length', 768)
        self.task_start_token = config.get('task_start_token', '<s_ocr>')
        self.prompt_end_token = config.get('prompt_end_token', None)
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if pretrained_model_path:
            self._load_from_pretrained(pretrained_model_path)
        else:
            self._build_custom_model(config)
        
        if hasattr(self, 'encoder') and hasattr(self, 'decoder'):
            super().__init__(self.encoder, self.decoder, config)
    
    def _load_from_pretrained(self, model_path: str):
        logger.info(f"Loading VisionEncoderDecoder model from {model_path}")
        
        try:
            # Попытка загрузить как стандартную модель HuggingFace
            config = VisionEncoderDecoderConfig.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, config=config)
            
            # Настраиваем токенизатор
            try:
                from transformers import AutoFeatureExtractor, AutoProcessor
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                image_processor = AutoFeatureExtractor.from_pretrained(model_path)
                self.processor = AutoProcessor.from_pretrained(model_path)
                if self.processor is None:
                    from transformers import ProcessorMixin
                    class CustomProcessor(ProcessorMixin):
                        def __init__(self, feature_extractor, tokenizer):
                            self.feature_extractor = feature_extractor
                            self.tokenizer = tokenizer
                            
                        def __call__(self, images=None, text=None, **kwargs):
                            if images is not None:
                                return self.feature_extractor(images, **kwargs)
                            elif text is not None:
                                return self.tokenizer(text, **kwargs)
                            
                    self.processor = CustomProcessor(image_processor, tokenizer)
                    
            except Exception as e:
                logger.warning(f"Could not load processor: {e}")
                self.processor = None
                
            self._configure_model_tokens()
            
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
            self.projection = nn.Identity()
            
        except Exception as e:
            logger.warning(f"Failed to load as pretrained VisionEncoderDecoder: {e}")
            self._build_custom_model(self.config)
    
    def _build_custom_model(self, config: Dict[str, Any]):
        logger.info("Building custom VisionEncoderDecoder model")
        
        self.encoder = CustomVisionEncoder(config)
        self.decoder = CustomTextDecoder(config)
        
        encoder_dim = self.encoder.output_dim
        decoder_dim = self.decoder.output_dim
        
        if encoder_dim != decoder_dim:
            self.projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.projection = nn.Identity()
        
        # Создаем процессор если его еще нет
        if self.processor is None:
            try:
                from transformers import AutoImageProcessor, AutoTokenizer, ProcessorMixin
                
                encoder_name = config.get('encoder_name', 'facebook/deit-base-distilled-patch16-384')
                decoder_name = config.get('decoder_name', 'microsoft/DialoGPT-medium')
                
                image_processor = AutoImageProcessor.from_pretrained(encoder_name)
                tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=True)
                
                class CustomProcessor(ProcessorMixin):
                    def __init__(self, image_processor, tokenizer):
                        self.image_processor = image_processor  # Используем image_processor вместо feature_extractor
                        self.tokenizer = tokenizer
                        
                    def __call__(self, images=None, text=None, **kwargs):
                        if images is not None:
                            return self.image_processor(images, **kwargs)
                        elif text is not None:
                            return self.tokenizer(text, **kwargs)
                
                self.processor = CustomProcessor(image_processor, tokenizer)
                self._configure_tokenizer()
                
            except Exception as e:
                logger.warning(f"Could not create processor: {e}")
                self.processor = None
    
    def _configure_tokenizer(self):
        if not self.processor or not hasattr(self.processor, 'tokenizer'):
            return
            
        tokenizer = self.processor.tokenizer
        
        special_tokens_to_add = {}
        
        if tokenizer.pad_token is None:
            special_tokens_to_add['pad_token'] = '<pad>'
        if tokenizer.bos_token is None:
            special_tokens_to_add['bos_token'] = '<s>'
        if tokenizer.eos_token is None:
            special_tokens_to_add['eos_token'] = '</s>'
            
        if special_tokens_to_add:
            tokenizer.add_special_tokens(special_tokens_to_add)
            
        if self.task_start_token:
            task_token_id = tokenizer.convert_tokens_to_ids(self.task_start_token)
            if task_token_id == tokenizer.unk_token_id:
                special_tokens = [self.task_start_token]
                if self.prompt_end_token and self.prompt_end_token != self.task_start_token:
                    special_tokens.append(self.prompt_end_token)
                    
                num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
                if num_added > 0 and hasattr(self.decoder, 'text_decoder'):
                    self.decoder.text_decoder.resize_token_embeddings(len(tokenizer))
                    logger.info(f"Added {num_added} task tokens to tokenizer")
    
    def _configure_model_tokens(self):
        if not hasattr(self.model, 'config') or not hasattr(self.processor, 'tokenizer'):
            return
            
        tokenizer = self.processor.tokenizer
        
        # Установка основных токенов
        if tokenizer.bos_token_id is not None:
            self.model.config.decoder_start_token_id = tokenizer.bos_token_id
        
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id
        
        # Добавление специальных токенов для задачи
        if self.task_start_token:
            task_token_id = tokenizer.convert_tokens_to_ids(self.task_start_token)
            if task_token_id == tokenizer.unk_token_id:
                special_tokens = [self.task_start_token]
                if self.prompt_end_token and self.prompt_end_token != self.task_start_token:
                    special_tokens.append(self.prompt_end_token)
                    
                num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
                if num_added > 0:
                    self.model.decoder.resize_token_embeddings(len(tokenizer))
                    logger.info(f"Added {num_added} task tokens to tokenizer")
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
        
        if hasattr(self, 'model'):
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'encoder_hidden_states': None
            }
        else:
            encoder_outputs = self.encoder(pixel_values)
            projected_outputs = self.projection(encoder_outputs)
            
            decoder_input_ids = None
            if labels is not None:
                decoder_input_ids = labels[:, :-1].contiguous()
                decoder_labels = labels[:, 1:].contiguous()
            else:
                decoder_labels = None
            
            decoder_outputs = self.decoder(
                encoder_hidden_states=projected_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=decoder_labels
            )
            
            return {
                'loss': decoder_outputs.loss if hasattr(decoder_outputs, 'loss') else None,
                'logits': decoder_outputs.logits if hasattr(decoder_outputs, 'logits') else None,
                'encoder_hidden_states': projected_outputs
            }
    
    def generate(self, pixel_values: torch.Tensor, prompt: Optional[str] = None, max_length: Optional[int] = None, num_beams: int = 4, **kwargs) -> List[str]:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        self.eval()
        with torch.no_grad():
            try:
                if hasattr(self, 'model'):
                    if prompt is not None and self.processor:
                        decoder_input_ids = self.processor.tokenizer(
                            prompt,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )["input_ids"].to(self.device)
                    else:
                        decoder_input_ids = None
                    
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        **kwargs
                    )
                else:
                    encoder_outputs = self.encoder(pixel_values)
                    projected_outputs = self.projection(encoder_outputs)
                    
                    # Создаем начальные токены для декодера
                    batch_size = pixel_values.size(0)
                    if prompt is not None and self.processor:
                        decoder_input_ids = self.processor.tokenizer(
                            prompt,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )["input_ids"].to(self.device)
                    else:
                        bos_token_id = getattr(self.processor.tokenizer, 'bos_token_id', 0)
                        decoder_input_ids = torch.full(
                            (batch_size, 1),
                            bos_token_id,
                            device=self.device,
                            dtype=torch.long
                        )
                    
                    generated_ids = self._generate_custom(
                        projected_outputs, 
                        decoder_input_ids, 
                        max_length, 
                        num_beams
                    )
                
                if self.processor:
                    generated_text = self.processor.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                else:
                    generated_text = ["Generated text not available (no tokenizer)"] * pixel_values.size(0)
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return [""] * pixel_values.size(0)
    
    def _generate_custom(self, encoder_outputs: torch.Tensor, decoder_input_ids: torch.Tensor, max_length: int, num_beams: int) -> torch.Tensor:
        batch_size = encoder_outputs.size(0)
        current_ids = decoder_input_ids
        
        eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id', None)
        
        for _ in range(max_length - current_ids.size(1)):
            decoder_outputs = self.decoder(
                encoder_hidden_states=encoder_outputs,
                decoder_input_ids=current_ids
            )
            
            next_token_logits = decoder_outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
            
            if eos_token_id is not None and (next_token_ids == eos_token_id).all():
                break
        
        return current_ids
    
    def freeze_encoder(self):
        if hasattr(self, 'model'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        elif hasattr(self, 'encoder'):
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        logger.info("Encoder parameters frozen")
    
    def to_device(self, precision: str):
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
            encoder_dir = output_dir / "encoder"
            decoder_dir = output_dir / "decoder"
            encoder_dir.mkdir(exist_ok=True)
            decoder_dir.mkdir(exist_ok=True)
            
            torch.save(self.encoder.state_dict(), encoder_dir / "pytorch_model.bin")
            torch.save(self.decoder.state_dict(), decoder_dir / "pytorch_model.bin")
            
            if hasattr(self, 'projection') and not isinstance(self.projection, nn.Identity):
                torch.save(self.projection.state_dict(), output_dir / "projection.pt")
        
        if self.processor:
            if hasattr(self.processor, 'save_pretrained'):
                self.processor.save_pretrained(output_dir)
            else:
                if hasattr(self.processor, 'feature_extractor') and hasattr(self.processor.feature_extractor, 'save_pretrained'):
                    self.processor.feature_extractor.save_pretrained(output_dir)
                if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'save_pretrained'):
                    self.processor.tokenizer.save_pretrained(output_dir)
        
        # Сохранение конфигурации модели
        model_config = {
            "model_type": "vision_encoder_decoder",
            "encoder_name": getattr(self.config, 'encoder_name', None),
            "decoder_name": getattr(self.config, 'decoder_name', None),
            "max_length": self.max_length,
            "task_start_token": self.task_start_token,
            "prompt_end_token": self.prompt_end_token
        }
        
        with open(output_dir / "custom_ved_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Custom VisionEncoderDecoder model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path], **kwargs) -> "CustomVisionEncoderDecoderModel":
        model_dir = Path(model_dir)
        config = kwargs.get('config', {})
        
        return cls(config, pretrained_model_path=str(model_dir))
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CustomVisionEncoderDecoderModel":
        return cls(config)