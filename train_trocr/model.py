import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import sys

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

ENCODER_MODELS = {
    "base": "microsoft/swin-small-patch4-window7-224",
    "large": "microsoft/swin-large-patch4-window12-384-in22k",
    "xlarge": None
}

DECODER_MODELS = {
    "base": "ai-forever/ruRoberta-large",
    "large": "ai-forever/ruBert-large",
    "xlarge": None
}


class TrOCRModel:

    def __init__(
        self,
        pretrained_model_name: str = "raxtemur/trocr-base-ru",
        encoder_model_name: Optional[str] = None,
        decoder_model_name: Optional[str] = None,
        encoder_size: str = "base",
        max_length: int = 512,
        device: Optional[str] = None,
        enable_gradient_checkpointing: bool = True,
        freeze_encoder: bool = True,
        precision: str = "bf16",
        img_size: Tuple[int, int] = (384, 384),
        enable_torch_compile: bool = False,
        flash_attention: bool = True,
        use_8bit_decoder: bool = False
    ):
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.img_size = img_size
        self.flash_attention = flash_attention
        self.custom_encoder = encoder_model_name is not None
        self.custom_decoder = decoder_model_name is not None
        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 не поддерживается на данном GPU, откат к FP16")
            self.precision = "fp16"
        
        logger.info(f"Используемая точность: {self.precision}")

        if self.custom_encoder or self.custom_decoder:
            self._initialize_custom_model(
                encoder_model_name or ENCODER_MODELS.get(encoder_size, ENCODER_MODELS["base"]),
                decoder_model_name or DECODER_MODELS.get(encoder_size, DECODER_MODELS["base"]),
                enable_gradient_checkpointing,
                freeze_encoder,
                use_8bit_decoder
            )
        else:
            self._initialize_base_model(
                pretrained_model_name,
                enable_gradient_checkpointing,
                freeze_encoder
            )

        self._setup_model_for_device(enable_torch_compile)
    
    def _initialize_custom_model(
        self,
        encoder_model_name: str,
        decoder_model_name: str,
        enable_gradient_checkpointing: bool,
        freeze_encoder: bool,
        use_8bit_decoder: bool
    ) -> None:

        logger.info(f"Инициализация кастомной модели с энкодером {encoder_model_name} и декодером {decoder_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, use_fast=True)
        self.image_processor = AutoImageProcessor.from_pretrained(encoder_model_name)

        if self.tokenizer.bos_token_id is None:
            logger.warning("BOS токен не определен в токенизаторе. Использование PAD токена в качестве BOS.")
            if self.tokenizer.pad_token_id is not None:
                self.tokenizer.bos_token = self.tokenizer.pad_token
            else:
                logger.warning("PAD токен тоже не определен. Установка специальных токенов.")
                self.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
                logger.info(f"Добавлены специальные токены. Новый размер словаря: {len(self.tokenizer)}")
        
        self.processor = TrOCRProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer
        )
        
        encoder_config = AutoConfig.from_pretrained(encoder_model_name)
        decoder_config = AutoConfig.from_pretrained(decoder_model_name)
        
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.decoder_start_token_id = self.tokenizer.bos_token_id
        decoder_config.pad_token_id = self.tokenizer.pad_token_id
        decoder_config.eos_token_id = self.tokenizer.eos_token_id

        for param_name, default_value in [
            ("use_cache", True if not enable_gradient_checkpointing else False),
            ("use_bert_attention_implementation", False)
        ]:
            if hasattr(decoder_config, param_name):
                setattr(decoder_config, param_name, default_value)
                logger.debug(f"Установлен параметр конфигурации: {param_name}={default_value}")

        if decoder_config.decoder_start_token_id is None:
            logger.warning("decoder_start_token_id всё еще None. Установка значения по умолчанию.")
            decoder_config.decoder_start_token_id = 0
        
        if enable_gradient_checkpointing:
            encoder_config.use_cache = False
            decoder_config.use_cache = False
        
        dtype = self._get_dtype_from_precision()
        
        quantization_config = None
        if use_8bit_decoder:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        self.encoder = AutoModel.from_pretrained(
            encoder_model_name,
            config=encoder_config,
            torch_dtype=dtype
        )
        
        if use_8bit_decoder and self.device.type == "cuda":
            try:
                self.decoder = AutoModel.from_pretrained(
                    decoder_model_name,
                    config=decoder_config,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except (AttributeError, ImportError, RuntimeError) as e:
                logger.warning(f"Ошибка при загрузке 8-битного декодера: {e}")
                logger.warning("Откат к стандартной загрузке декодера")
                self.decoder = AutoModel.from_pretrained(
                    decoder_model_name,
                    config=decoder_config,
                    torch_dtype=dtype
                )
        else:
            self.decoder = AutoModel.from_pretrained(
                decoder_model_name,
                config=decoder_config,
                torch_dtype=dtype
            )

        # Проверка согласования словаря
        tokenizer_vocab_size = len(self.tokenizer)
        decoder_vocab_size = self.decoder.config.vocab_size
        
        logger.info(f"Размер словаря токенизатора: {tokenizer_vocab_size}, словаря декодера: {decoder_vocab_size}")
        

        max_vocab_size = max(tokenizer_vocab_size, decoder_vocab_size)
        if tokenizer_vocab_size != max_vocab_size:
            logger.info(f"Расширение словаря токенизатора с {tokenizer_vocab_size} до {max_vocab_size}")
            self.tokenizer.add_special_tokens({'additional_special_tokens': 
                                            [f'[unused{i}]' for i in range(max_vocab_size - tokenizer_vocab_size)]})
        
        if decoder_vocab_size != max_vocab_size:
            logger.info(f"Расширение словаря декодера с {decoder_vocab_size} до {max_vocab_size}")
            self.decoder.resize_token_embeddings(max_vocab_size)

        logger.info(f"После согласования: токенизатор={len(self.tokenizer)}, декодер={self.decoder.config.vocab_size}")
        
        hidden_size_encoder = self.encoder.config.hidden_size
        hidden_size_decoder = self.decoder.config.hidden_size
        
        if hidden_size_encoder != hidden_size_decoder:
            self.encoder_projection = nn.Linear(hidden_size_encoder, hidden_size_decoder)
            logger.info(f"Добавлен проекционный слой: {hidden_size_encoder} -> {hidden_size_decoder}")
        else:
            self.encoder_projection = nn.Identity()

        self._build_custom_trocr_model()
        
        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _build_custom_trocr_model(self) -> None:

        class CustomTrOCRModel(nn.Module):
            def __init__(self, encoder, decoder, encoder_projection):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.encoder_projection = encoder_projection
                self.config = decoder.config
            
            def forward(
                self, 
                pixel_values=None, 
                decoder_input_ids=None, 
                decoder_attention_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                labels=None,
                return_dict=True
            ):
                try:
                    if encoder_outputs is None and pixel_values is not None:
                        encoder_outputs = self.encoder(pixel_values).last_hidden_state
                        encoder_outputs = self.encoder_projection(encoder_outputs)

                    if decoder_input_ids is None and labels is not None:
                        decoder_input_ids = self._shift_right(labels)
                    
                    if decoder_input_ids is not None:
                        vocab_size = getattr(self.config, "vocab_size", 50265)

                        if (decoder_input_ids < 0).any() or (decoder_input_ids >= vocab_size).any():
                            # logger.warning(f"Найдены недопустимые индексы в decoder_input_ids (диапазон [0, {vocab_size-1}])")
                            decoder_input_ids = torch.clamp(decoder_input_ids, min=0, max=vocab_size-1)
                            # logger.info("decoder_input_ids ограничены допустимым диапазоном")

                    try:
                        decoder_outputs = self.decoder(
                            input_ids=decoder_input_ids,
                            attention_mask=decoder_attention_mask,
                            encoder_hidden_states=encoder_outputs,
                            past_key_values=past_key_values,
                            return_dict=return_dict
                        )
                    except Exception as e:
                        logger.error(f"Ошибка в декодере: {e}")
                        if decoder_input_ids is not None:
                            logger.info(f"decoder_input_ids: shape={decoder_input_ids.shape}, min={decoder_input_ids.min().item()}, "
                                    f"max={decoder_input_ids.max().item()}")
                            
                            vocab_size = getattr(self.config, "vocab_size", 50265)
                            if decoder_input_ids.max() >= vocab_size:
                                logger.warning(f"decoder_input_ids содержит значения >= {vocab_size}, применяем жесткое ограничение")
                                decoder_input_ids = torch.clamp(decoder_input_ids, min=0, max=vocab_size-1)
                                
                                decoder_outputs = self.decoder(
                                    input_ids=decoder_input_ids,
                                    attention_mask=decoder_attention_mask,
                                    encoder_hidden_states=encoder_outputs,
                                    past_key_values=past_key_values,
                                    return_dict=return_dict
                                )
                            else:
                                raise
                        else:
                            raise
                    
                    loss = None
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        lm_logits = decoder_outputs.logits if hasattr(decoder_outputs, "logits") else decoder_outputs[0]
                        
                        actual_vocab_size = lm_logits.size(-1)
                        if actual_vocab_size != self.config.vocab_size:
                            # logger.warning(f"Размер словаря в конфигурации ({self.config.vocab_size}) не соответствует "
                            #             f"фактическому размеру логитов ({actual_vocab_size}). Используем фактический размер.")
                            pass
                        
                        try:
                            batch_size = lm_logits.size(0)
                            seq_length = lm_logits.size(1)
                            
                            logger.debug(f"Форма lm_logits: {lm_logits.shape}, labels: {labels.shape}")
                            
                            if labels.shape != (batch_size, seq_length):
                                logger.warning(f"Несоответствие формы: labels {labels.shape} vs logits {lm_logits.shape}")
                                if labels.shape[1] > seq_length:
                                    labels = labels[:, :seq_length]
                                elif labels.shape[1] < seq_length:
                                    padding = torch.full((batch_size, seq_length - labels.shape[1]), 
                                                        -100, device=labels.device, dtype=labels.dtype)
                                    labels = torch.cat([labels, padding], dim=1)

                            lm_logits_flat = lm_logits.contiguous().view(-1, actual_vocab_size)
                            labels_flat = labels.contiguous().view(-1)
                            
                            if (labels_flat != -100).any() and (labels_flat >= actual_vocab_size).any():
                                invalid_indices = (labels_flat != -100) & (labels_flat >= actual_vocab_size)
                                num_invalid = invalid_indices.sum().item()
                                # logger.warning(f"Найдено {num_invalid} меток за пределами словаря размера {actual_vocab_size}. "
                                #             f"Заменяем на -100 (игнорирование).")
                                labels_flat = torch.where(invalid_indices, torch.tensor(-100, device=labels.device), labels_flat)

                            loss = loss_fct(lm_logits_flat, labels_flat)
                        except Exception as e:
                            logger.error(f"Ошибка при расчете потерь: {e}")
                            logger.error(f"Размеры: lm_logits={lm_logits.shape}, labels={labels.shape}, "
                                        f"vocab_size_config={self.config.vocab_size}, actual_vocab_size={actual_vocab_size}")
                            loss = torch.tensor(1.0, device=lm_logits.device, requires_grad=True)
                    
                    if not return_dict:
                        output = (decoder_outputs.logits,) + decoder_outputs[1:]
                        return ((loss,) + output) if loss is not None else output
                    
                    from transformers.modeling_outputs import Seq2SeqLMOutput
                    return Seq2SeqLMOutput(
                        loss=loss,
                        logits=decoder_outputs.logits if hasattr(decoder_outputs, "logits") else decoder_outputs[0],
                        past_key_values=decoder_outputs.past_key_values if hasattr(decoder_outputs, "past_key_values") else None,
                        decoder_hidden_states=decoder_outputs.hidden_states if hasattr(decoder_outputs, "hidden_states") else None,
                        decoder_attentions=decoder_outputs.attentions if hasattr(decoder_outputs, "attentions") else None,
                        cross_attentions=decoder_outputs.cross_attentions if hasattr(decoder_outputs, "cross_attentions") else None,
                        encoder_last_hidden_state=encoder_outputs
                    )
                except Exception as e:
                    logger.error(f"Ошибка в forward: {e}")
                    logger.error(f"Типы входных данных: pixel_values={type(pixel_values)}, "
                            f"decoder_input_ids={type(decoder_input_ids)}, labels={type(labels)}")
                    logger.error(f"Размеры входных данных: "
                            f"pixel_values={pixel_values.shape if pixel_values is not None else None}, "
                            f"decoder_input_ids={decoder_input_ids.shape if decoder_input_ids is not None else None}, "
                            f"labels={labels.shape if labels is not None else None}")
                    raise
            
            def _shift_right(self, input_ids):
                input_ids_safe = input_ids.clone()
                shifted_input_ids = input_ids_safe.new_zeros(input_ids_safe.shape)
                bos_token_id = self.config.decoder_start_token_id
                if bos_token_id is None:
                    if hasattr(self.config, "pad_token_id") and self.config.pad_token_id is not None:
                        bos_token_id = self.config.pad_token_id
                        logger.warning(f"bos_token_id не задан, используется pad_token_id: {bos_token_id}")
                    else:
                        bos_token_id = 0
                        logger.warning("bos_token_id не задан и pad_token_id не доступен. Используется значение по умолчанию: 0")
                
                vocab_size = getattr(self.config, "vocab_size", 50265)
                pad_token_id = getattr(self.config, "pad_token_id", 1)
                
                invalid_mask = (input_ids_safe != -100) & ((input_ids_safe < 0) | (input_ids_safe >= vocab_size))
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum().item()
                    logger.warning(f"Обнаружено {invalid_count} токенов вне диапазона [0, {vocab_size-1}] в input_ids.")
                    input_ids_safe = torch.where(invalid_mask, torch.tensor(pad_token_id, device=input_ids_safe.device), input_ids_safe)

                try:
                    shifted_input_ids[..., 1:] = input_ids_safe[..., :-1].clone()
                    shifted_input_ids[..., 0] = bos_token_id
                    if shifted_input_ids.max() >= vocab_size:
                        logger.warning(f"После сдвига все еще есть недопустимые индексы >= {vocab_size}")
                        shifted_input_ids = torch.clamp(shifted_input_ids, min=0, max=vocab_size-1)
                except Exception as e:
                    logger.error(f"Ошибка при сдвиге токенов: {e}")
                    logger.error(f"Форма input_ids: {input_ids_safe.shape}, Форма shifted_input_ids: {shifted_input_ids.shape}")
                    logger.error(f"bos_token_id: {bos_token_id} (тип: {type(bos_token_id)})")
                    batch_size = input_ids_safe.size(0)
                    seq_len = input_ids_safe.size(1)
                    shifted_input_ids = input_ids_safe.new_zeros((batch_size, seq_len))
                    shifted_input_ids[:, 0] = bos_token_id
                    if seq_len > 1:
                        shifted_input_ids[:, 1:] = torch.clamp(input_ids_safe[:, :-1], min=0, max=vocab_size-1)
                
                return shifted_input_ids
            
            def generate(self, **kwargs):
                from transformers.generation import GenerationMixin
                
                class GenerationModel(nn.Module, GenerationMixin):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        self.config = model.config
                        self.main_input_name = "pixel_values"
                    
                    def prepare_inputs_for_generation(self, input_ids, pixel_values=None, **kwargs):
                        inputs = {}
                        inputs["decoder_input_ids"] = input_ids
                        
                        if "encoder_outputs" in kwargs:
                            inputs["encoder_outputs"] = kwargs["encoder_outputs"]
                        else:
                            if pixel_values is not None:
                                encoder_outputs = self.model.encoder(pixel_values).last_hidden_state
                                encoder_outputs = self.model.encoder_projection(encoder_outputs)
                                inputs["encoder_outputs"] = encoder_outputs
                        
                        return inputs
                    
                    def forward(self, **kwargs):
                        return self.model(**kwargs)
                        
                    def can_generate(self):
                        return True
                
                generation_model = GenerationModel(self)
                return generation_model.generate(**kwargs)
        
        self.model = CustomTrOCRModel(
            self.encoder, 
            self.decoder, 
            self.encoder_projection
        )
    
    def _initialize_base_model(
        self,
        pretrained_model_name: str,
        enable_gradient_checkpointing: bool,
        freeze_encoder: bool
    ) -> None:

        logger.info(f"Инициализация базовой модели из {pretrained_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained('raxtemur/trocr-base-ru')
        self.processor = TrOCRProcessor.from_pretrained(
            pretrained_model_name,
            tokenizer=self.tokenizer,
            use_fast=True
        )

        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.decoder_start_token_id = self.tokenizer.bos_token_id
        config.pad_token_id = self.tokenizer.pad_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        config.is_encoder_decoder = True
        
        if enable_gradient_checkpointing:
            if hasattr(config, "encoder"):
                config.encoder.use_cache = False
            if hasattr(config, "decoder"):
                config.decoder.use_cache = False
        else:
            if hasattr(config, "encoder"):
                config.encoder.use_cache = True
            if hasattr(config, "decoder"):
                config.decoder.use_cache = True
        
        dtype = self._get_dtype_from_precision()

        self.model = AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        original_forward = self.model.forward
        
        def forward_with_custom_loss(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            
            if 'labels' in kwargs and kwargs['labels'] is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                labels = kwargs['labels']
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    
                    if hasattr(outputs, 'loss'):
                        outputs.loss = loss
                
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]
                    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    
                    if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                        outputs = (loss,) + outputs[1:]
            
            return outputs

        self.model.forward = forward_with_custom_loss
        
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))
        
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Проверка градиентов включена для экономии памяти")
        
        if freeze_encoder:
            self._freeze_encoder()
        
        for param in self.model.decoder.parameters():
            param.requires_grad = True
    
    def _get_dtype_from_precision(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        else:
            return torch.float32
    
    def _freeze_encoder(self) -> None:
        if hasattr(self, "encoder"):
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        logger.info("Параметры энкодера заморожены для экономии памяти и ускорения обучения")
    
    def _enable_gradient_checkpointing(self) -> None:
        if hasattr(self, "encoder") and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        
        if hasattr(self, "decoder") and hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable()
        
        if hasattr(self, "model") and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        logger.info("Проверка градиентов включена для всех применимых компонентов")
    
    def _setup_model_for_device(self, enable_torch_compile: bool) -> None:

        trainable_params = sum(p.numel() for p in self.get_model_parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.get_model_parameters())
        logger.info(f"Модель имеет {trainable_params:,} обучаемых параметров из {total_params:,} всего")
        
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            if hasattr(self, "encoder_projection"):
                self.encoder_projection.to(self.device)
        else:
            self.model.to(self.device)
        
        if enable_torch_compile and self._check_torch_compile_compatibility():
            self._apply_torch_compile()
        
        if self.flash_attention and self.device.type == "cuda":
            self._apply_flash_attention()
    
    def _check_torch_compile_compatibility(self) -> bool:

        if not hasattr(torch, "compile"):
            logger.warning("torch.compile недоступен в текущей версии PyTorch")
            return False
        
        if not torch.__version__ >= "2.0.0":
            logger.warning(f"torch.compile требует PyTorch 2.0+, текущая версия: {torch.__version__}")
            return False
        
        if self.device.type != "cuda":
            logger.warning("torch.compile оптимизирован для CUDA устройств")
            return False
        
        if hasattr(self.model, "config") and not (self.model.encoder.config.use_cache or self.model.decoder.config.use_cache):
            logger.warning(
                "torch.compile и gradient_checkpointing несовместимы. "
                "Отключение torch.compile для стабильности."
            )
            return False
        
        return True
    
    def _apply_torch_compile(self) -> None:
        try:
            logger.info("Применение torch.compile для оптимизации модели...")
            if hasattr(self, "decoder"):
                self.decoder = torch.compile(
                    self.decoder,
                    mode="max-autotune",
                    fullgraph=False
                )
                logger.info("Декодер скомпилирован с помощью torch.compile")
            else:
                self.model.decoder = torch.compile(
                    self.model.decoder,
                    mode="max-autotune",
                    fullgraph=False
                )
                logger.info("Декодер модели скомпилирован с помощью torch.compile")
        except Exception as e:
            logger.warning(f"Не удалось скомпилировать модель: {e}")
    
    def _apply_flash_attention(self) -> None:
        try:
            from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
            
            def replace_attention(module):
                for name, child in module.named_children():
                    if "self_attn" in name.lower() and hasattr(child, "forward"):
                        try:
                            flash_module = FlashSelfAttention(
                                attention_dropout=getattr(child, "dropout", 0.0)
                            )
                            setattr(module, name, flash_module)
                            logger.debug(f"Заменен модуль self-attention: {name}")
                        except Exception as e:
                            logger.debug(f"Не удалось заменить модуль self-attention {name}: {e}")
                    
                    elif "cross_attn" in name.lower() and hasattr(child, "forward"):
                        try:
                            flash_module = FlashCrossAttention(
                                attention_dropout=getattr(child, "dropout", 0.0)
                            )
                            setattr(module, name, flash_module)
                            logger.debug(f"Заменен модуль cross-attention: {name}")
                        except Exception as e:
                            logger.debug(f"Не удалось заменить модуль cross-attention {name}: {e}")
                    
                    else:
                        replace_attention(child)
            
            if hasattr(self, "decoder"):
                replace_attention(self.decoder)
            else:
                replace_attention(self.model.decoder)
            
            logger.info("FlashAttention применен к применимым модулям внимания")
        except ImportError:
            logger.warning("Библиотека flash-attn не установлена. FlashAttention не применен.")
        except Exception as e:
            logger.warning(f"Ошибка при применении FlashAttention: {e}")
    
    def get_model_parameters(self) -> List[torch.nn.Parameter]:
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            if hasattr(self, "encoder_projection"):
                params += list(self.encoder_projection.parameters())
            return params
        else:
            return list(self.model.parameters())
    
    def train(self, mode: bool = True) -> "TrOCRModel":
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            self.encoder.train(mode)
            self.decoder.train(mode)
            if hasattr(self, "encoder_projection"):
                self.encoder_projection.train(mode)
        else:
            self.model.train(mode)
        return self
    
    def eval(self) -> "TrOCRModel":
        return self.train(False)
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        pixel_values = self._to_device(pixel_values)
        if labels is not None:
            labels = self._to_device(labels)
        
        pixel_values, labels = self._prepare_inputs(pixel_values, labels)

        if self.model.training:
            outputs = self._forward_train(pixel_values, labels)
        else:
            outputs = self._forward_eval(pixel_values, labels)

        if labels is not None and hasattr(outputs, 'loss'):
            # Здесь мы можем убедиться, что потери рассчитаны правильно
            # Если возникают проблемы, можно пересчитать их здесь
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                logger.warning("Обнаружены неправильные потери (NaN/Inf). Пересчет...")
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), 
                                labels.contiguous().view(-1))
                    outputs.loss = loss
        
        return outputs
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)
    
    def _prepare_inputs(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if torch.__version__ >= "2.0.0" and self.device.type == "cuda":
            pixel_values = pixel_values.clone()
            if labels is not None:
                labels = labels.clone()
        return pixel_values, labels
    
    def _forward_train(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if pixel_values.dim() != 4:
            logger.warning(f"Неправильная размерность pixel_values: {pixel_values.shape}. Ожидается [B, C, H, W]")
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(1)
                logger.info(f"Исправлена размерность pixel_values: {pixel_values.shape}")
        
        if labels is not None:
            if pixel_values.size(0) != labels.size(0):
                logger.error(f"Несоответствие размеров батча: pixel_values={pixel_values.shape}, labels={labels.shape}")
                min_batch = min(pixel_values.size(0), labels.size(0))
                pixel_values = pixel_values[:min_batch]
                labels = labels[:min_batch]
                logger.info(f"Скорректированы размеры батча до {min_batch}")
            
            if torch.any(labels >= self.tokenizer.vocab_size):
                invalid_indices = torch.nonzero(labels >= self.tokenizer.vocab_size).view(-1)
                # logger.error(f"Найдены недопустимые индексы токенов: {labels[invalid_indices]} "
                #             f"(vocab_size={self.tokenizer.vocab_size})")
                labels = torch.clamp(labels, max=self.tokenizer.vocab_size-1)
                # logger.info("Индексы токенов ограничены размером словаря")

        if self.precision in ["bf16", "fp16"]:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast(
                    enabled=True,
                    dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
                    device_type=self.device.type
                ):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels,
                        return_dict=True,
                    )
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels,
                        return_dict=True,
                    )
        else:
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True,
            )
        
        if hasattr(outputs, "loss") and not outputs.loss.requires_grad:
            logger.warning("Потери не требуют градиентов, клонирование с requires_grad=True")
            outputs.loss = outputs.loss.clone().requires_grad_(True)
        
        return outputs
    
    def _forward_eval(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True,
            )
        return outputs
    
    def generate(
        self, 
        pixel_values: torch.Tensor, 
        max_length: Optional[int] = None,
        num_beams: int = 4,
        early_stopping: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0
    ) -> List[str]:

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        pixel_values = pixel_values.to(self.device)
        
        try:
            decoder_input_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]] * pixel_values.size(0)
            ).to(self.device)
            
            generate_kwargs = {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
                "max_length": max_length or self.max_length,
                "num_beams": num_beams,
                "early_stopping": early_stopping,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "use_cache": True,  # Меняем на True для генерации
                "return_dict_in_generate": False
            }
            
            generated_ids = self.model.generate(**generate_kwargs)

            generated_text = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text
        except Exception as e:
            logger.error(f"Ошибка при генерации текста: {e}")
            return [""] * pixel_values.size(0)
    
    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            encoder_dir = output_dir / "encoder"
            decoder_dir = output_dir / "decoder"
            encoder_dir.mkdir(exist_ok=True)
            decoder_dir.mkdir(exist_ok=True)
            
            self.encoder.save_pretrained(encoder_dir)
            self.decoder.save_pretrained(decoder_dir)
            
            self.tokenizer.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(decoder_dir)
            
            if hasattr(self, "encoder_projection") and not isinstance(self.encoder_projection, nn.Identity):
                torch.save(self.encoder_projection.state_dict(), output_dir / "encoder_projection.pt")
            
            model_config = {
                "custom_model": True,
                "encoder_dir": "encoder",
                "decoder_dir": "decoder",
                "has_projection": not isinstance(self.encoder_projection, nn.Identity),
                "hidden_size_encoder": self.encoder.config.hidden_size,
                "hidden_size_decoder": self.decoder.config.hidden_size
            }
            import json
            with open(output_dir / "model_config.json", "w") as f:
                json.dump(model_config, f)
        else:
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        logger.info(f"Модель успешно сохранена в {output_dir}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "bf16",
        flash_attention: bool = True,
        enable_torch_compile: bool = False
    ) -> "TrOCRModel":

        model_dir = Path(model_dir)
        
        model_config_path = model_dir / "model_config.json"
        if model_config_path.exists():
            try:
                import json
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)
                
                if model_config.get("custom_model", False):
                    return cls._load_custom_model(
                        model_dir, 
                        model_config, 
                        device, 
                        precision,
                        flash_attention,
                        enable_torch_compile
                    )
            except Exception as e:
                logger.warning(f"Ошибка при чтении конфигурации модели: {e}. Продолжаем со стандартной загрузкой.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer, use_fast=True)

        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.processor = processor
        instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.max_length = getattr(tokenizer, "model_max_length", 512)
        instance.precision = precision
        instance.flash_attention = flash_attention
        
        if isinstance(instance.device, str):
            instance.device = torch.device(instance.device)

        config = AutoConfig.from_pretrained(model_dir)
        if not hasattr(config, 'decoder_start_token_id') or config.decoder_start_token_id is None:
            config.decoder_start_token_id = tokenizer.bos_token_id
        config.is_encoder_decoder = True
        
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 не поддерживается на данном GPU, откат к FP16")
            precision = "fp16"
        
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        instance.model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=dtype
        )
        
        instance.model.to(instance.device)
        
        if flash_attention and instance.device.type == "cuda":
            instance._apply_flash_attention()
        
        if enable_torch_compile and instance._check_torch_compile_compatibility():
            instance._apply_torch_compile()
        
        logger.info(f"Загружена модель TrOCRModel из {model_dir} на {instance.device}")
        return instance
    
    @classmethod
    def _load_custom_model(
        cls,
        model_dir: Path,
        model_config: Dict[str, Any],
        device: Optional[Union[str, torch.device]],
        precision: str,
        flash_attention: bool,
        enable_torch_compile: bool
    ) -> "TrOCRModel":
        """Загрузка кастомной модели с раздельными энкодером и декодером."""
        
        encoder_dir = model_dir / model_config["encoder_dir"]
        decoder_dir = model_dir / model_config["decoder_dir"]
        
        logger.info(f"Загрузка кастомной модели из {model_dir}")
        logger.info(f"Энкодер: {encoder_dir}")
        logger.info(f"Декодер: {decoder_dir}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info(f"Токенизатор успешно загружен из {model_dir}")
            
            try:
                processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer)
                logger.info(f"Процессор успешно загружен из {model_dir}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить процессор из {model_dir}: {e}")
                image_processor = AutoImageProcessor.from_pretrained(encoder_dir)
                processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                logger.info("Создан новый процессор из image_processor и tokenizer")
            
            instance = cls.__new__(cls)
            instance.tokenizer = tokenizer
            instance.processor = processor
            instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            instance.max_length = getattr(tokenizer, "model_max_length", 512)
            instance.precision = precision
            instance.flash_attention = flash_attention
            instance.img_size = (224, 224)

            if isinstance(instance.device, str):
                instance.device = torch.device(instance.device)
                
            instance.dtype = instance._get_dtype_from_precision()
            
            try:
                instance.encoder = AutoModel.from_pretrained(
                    encoder_dir,
                    torch_dtype=instance.dtype
                )
                logger.info(f"Энкодер успешно загружен из {encoder_dir}")
                
                instance.decoder = AutoModel.from_pretrained(
                    decoder_dir,
                    torch_dtype=instance.dtype
                )
                logger.info(f"Декодер успешно загружен из {decoder_dir}")

                decoder_vocab_size = instance.decoder.config.vocab_size
                tokenizer_vocab_size = len(instance.tokenizer)
                logger.info(f"Размер словаря декодера: {decoder_vocab_size}, размер словаря токенизатора: {tokenizer_vocab_size}")

                if decoder_vocab_size != tokenizer_vocab_size:
                    logger.warning(f"Несоответствие размеров словаря. Обновление декодера до {tokenizer_vocab_size}")
                    instance.decoder.resize_token_embeddings(tokenizer_vocab_size)
                
                if model_config.get("has_projection", False):
                    projection_path = model_dir / "encoder_projection.pt"
                    if projection_path.exists():
                        hidden_size_encoder = instance.encoder.config.hidden_size
                        hidden_size_decoder = instance.decoder.config.hidden_size
                        
                        if hidden_size_encoder != hidden_size_decoder:
                            instance.encoder_projection = nn.Linear(hidden_size_encoder, hidden_size_decoder)
                            state_dict = torch.load(projection_path, map_location=instance.device)
                            instance.encoder_projection.load_state_dict(state_dict)
                            logger.info(f"Проекционный слой загружен: {hidden_size_encoder} -> {hidden_size_decoder}")
                        else:
                            instance.encoder_projection = nn.Identity()
                            logger.info("Проекционный слой не требуется (размеры скрытых состояний совпадают)")
                    else:
                        hidden_size_encoder = instance.encoder.config.hidden_size
                        hidden_size_decoder = instance.decoder.config.hidden_size
                        
                        if hidden_size_encoder != hidden_size_decoder:
                            instance.encoder_projection = nn.Linear(hidden_size_encoder, hidden_size_decoder)
                            logger.warning(f"Файл проекционного слоя не найден. Создан новый слой {hidden_size_encoder} -> {hidden_size_decoder}")
                        else:
                            instance.encoder_projection = nn.Identity()
                            logger.info("Проекционный слой не требуется (размеры скрытых состояний совпадают)")
                else:
                    hidden_size_encoder = instance.encoder.config.hidden_size
                    hidden_size_decoder = instance.decoder.config.hidden_size
                    
                    if hidden_size_encoder != hidden_size_decoder:
                        instance.encoder_projection = nn.Linear(hidden_size_encoder, hidden_size_decoder)
                        logger.info(f"Создан проекционный слой: {hidden_size_encoder} -> {hidden_size_decoder}")
                    else:
                        instance.encoder_projection = nn.Identity()
                
                instance._build_custom_trocr_model()
                
                instance._setup_model_for_device(enable_torch_compile)
                
                logger.info(f"Кастомная модель TrOCRModel успешно загружена из {model_dir}")
                return instance
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке энкодера/декодера: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Не удалось загрузить токенизатор из {model_dir}: {e}")
            logger.warning("Пробуем загрузить токенизатор из декодера")
            instance = cls(
                encoder_model_name=str(encoder_dir),
                decoder_model_name=str(decoder_dir),
                device=device,
                precision=precision,
                flash_attention=flash_attention,
                enable_torch_compile=enable_torch_compile
            )

            if model_config.get("has_projection", False):
                projection_path = model_dir / "encoder_projection.pt"
                if projection_path.exists():
                    state_dict = torch.load(projection_path, map_location=instance.device)
                    instance.encoder_projection.load_state_dict(state_dict)
                    logger.info("Проекционный слой загружен из сохраненного файла")
            
            logger.info(f"Кастомная модель TrOCRModel загружена с использованием запасного варианта")
            return instance