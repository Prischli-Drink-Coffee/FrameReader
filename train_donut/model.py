import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn.functional as F
import json
import re

import torch
from transformers import (
    VisionEncoderDecoderConfig, 
    VisionEncoderDecoderModel, 
    DonutProcessor,
    PreTrainedTokenizer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DonutModel:
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        processor: DonutProcessor,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        max_length: int = 768,
        task_start_token: str = "<s>",
        prompt_end_token: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        self.model.to(self.device)

        logger.info("Проверка ключевых токенов:")
        for token_name, token_id in [
            ("task_start_token", self.model.config.decoder_start_token_id),
            ("pad_token", self.processor.tokenizer.pad_token_id),
            ("eos_token", self.processor.tokenizer.eos_token_id),
            ("bos_token", self.processor.tokenizer.bos_token_id),
            ("unk_token", self.processor.tokenizer.unk_token_id)
        ]:
            token_text = self.processor.tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else "None"
            logger.info(f"  - {token_name}: ID={token_id}, текст={token_text}")
        
        self.precision = precision
        self.max_length = max_length
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self._configure_model_for_training()
        
        vocab_size = len(self.processor.tokenizer)
        emb_size = self.model.decoder.get_input_embeddings().num_embeddings
        
        if vocab_size != emb_size:
            logger.warning(
                f"Несоответствие размеров: размер словаря={vocab_size}, "
                f"размер эмбеддингов={emb_size}. Увеличиваем размер эмбеддингов."
            )
            self.model.decoder.resize_token_embeddings(vocab_size)
            logger.info(f"Размер эмбеддингов обновлен до {vocab_size}")
        self._resize_position_embeddings()
        
        logger.info(f"Модель Donut инициализирована на устройстве: {self.device}, точность: {self.precision}")

    def _resize_position_embeddings(self):
        logger.info(f"Проверка необходимости изменения размера позиционных эмбеддингов")    
        decoder = self.model.decoder.model.decoder
        current_max_pos = self.model.config.decoder.max_position_embeddings
        
        if current_max_pos >= self.max_length:
            logger.info(f"Изменение размера не требуется: текущий={current_max_pos}, требуемый={self.max_length}")
            return
        
        logger.info(f"Изменение размера позиционных эмбеддингов с {current_max_pos} на {self.max_length}")
        
        old_embed_positions = decoder.embed_positions
        old_weight = old_embed_positions.weight
        
        embed_dim = old_weight.size(1)
        new_max_positions = self.max_length + 2
        
        new_embeddings = torch.nn.Embedding(new_max_positions, embed_dim, device=old_weight.device)
 
        if new_max_positions > old_weight.size(0):
            new_weight = torch.nn.functional.interpolate(
                old_weight.unsqueeze(0).float(),
                size=(new_max_positions, embed_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).to(old_weight.dtype)
            
            new_embeddings.weight.data = new_weight
            decoder.embed_positions = new_embeddings

            self.model.config.decoder.max_position_embeddings = self.max_length
            logger.info(f"Размер позиционных эмбеддингов успешно изменен на {self.max_length}")
        else:
            logger.info(f"Новый размер меньше или равен текущему, пропускаем изменение")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        max_length: int = 768,
        image_size: Tuple[int, int] = (1280, 960),
        task_start_token: str = "<s>",
        prompt_end_token: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs
    ) -> "DonutModel":

        logger.info(f"Загрузка предварительно обученной модели Donut из {model_name_or_path}")

        try:

            donut_config_path = Path(model_name_or_path) / "donut_config.json"
            if donut_config_path.exists():
                with open(donut_config_path, "r", encoding="utf-8") as f:
                    donut_config = json.load(f)
                    logger.info(f"Загружена пользовательская конфигурация Donut: {donut_config}")
                    
                    if "task_start_token" in donut_config and task_start_token == "<s>":
                        task_start_token = donut_config["task_start_token"]
                        logger.info(f"Использую task_start_token из donut_config: {task_start_token}")
                    
                    if "prompt_end_token" in donut_config and prompt_end_token is None:
                        prompt_end_token = donut_config["prompt_end_token"]
                        logger.info(f"Использую prompt_end_token из donut_config: {prompt_end_token}")
                        
                    if "max_length" in donut_config and max_length == 768:
                        saved_max_length = donut_config["max_length"]
                        logger.info(f"Найдена сохраненная max_length: {saved_max_length}, запрошенная: {max_length}")
                        max_length = max(max_length, saved_max_length)
                        logger.info(f"Использую max_length: {max_length}")

            config = VisionEncoderDecoderConfig.from_pretrained(model_name_or_path, revision=revision)
            config.encoder.image_size = image_size
            config.decoder.max_length = max_length
            processor = DonutProcessor.from_pretrained(model_name_or_path, revision=revision, use_fust=True)
            processor.image_processor.size = image_size[::-1]  # (width, height)
            processor.image_processor.do_align_long_axis = False

            logger.info(f"Анализ словаря токенизатора из {model_name_or_path}:")
            special_tokens = {
                "bos_token": getattr(processor.tokenizer, "bos_token", "<s>"),
                "eos_token": getattr(processor.tokenizer, "eos_token", "</s>"),
                "pad_token": getattr(processor.tokenizer, "pad_token", "<pad>"),
                "unk_token": getattr(processor.tokenizer, "unk_token", "<unk>"),
                "mask_token": getattr(processor.tokenizer, "mask_token", "<mask>")
            }
            logger.info(f"Специальные токены: {special_tokens}")

            vocab = getattr(processor.tokenizer, "get_vocab", lambda: {})()
            if task_start_token in vocab:
                logger.info(f"Токен {task_start_token} найден в словаре с ID {vocab[task_start_token]}")
            else:
                logger.info(f"Токен {task_start_token} не найден в словаре")

            added_tokens = getattr(processor.tokenizer, "added_tokens_decoder", {})
            if added_tokens:
                logger.info(f"Первые 5 добавленных токенов: {list(added_tokens.items())[:5]}")
            else:
                logger.info("Нет добавленных токенов")

            model = VisionEncoderDecoderModel.from_pretrained(
                model_name_or_path,
                config=config,
                revision=revision,
                **kwargs
            )
            
            if processor.tokenizer.pad_token_id is None:
                logger.warning("Токенизатор не имеет pad_token_id, устанавливаем в eos_token_id")
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            
            donut_model = cls(
                model=model,
                processor=processor,
                device=device,
                precision=precision,
                max_length=max_length,
                task_start_token=task_start_token,
                prompt_end_token=prompt_end_token,
            )
            
            logger.info(f"Модель Donut успешно загружена")
            return donut_model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели Donut: {e}")
            raise
    
    def _configure_model_for_training(self) -> None:
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        
        added_tokens_path = Path(self.processor.tokenizer.name_or_path) / "added_tokens.json"
        task_start_token_id = None
        
        if added_tokens_path.exists():
            try:
                with open(added_tokens_path, "r", encoding="utf-8") as f:
                    added_tokens_data = json.load(f)
                    if isinstance(added_tokens_data, dict):
                        if self.task_start_token in added_tokens_data:
                            task_start_token_id = added_tokens_data[self.task_start_token]
                            logger.info(f"Найден токен {self.task_start_token} в added_tokens.json с ID {task_start_token_id}")
                    elif isinstance(added_tokens_data, list):
                        for token_info in added_tokens_data:
                            if isinstance(token_info, dict) and token_info.get('content') == self.task_start_token:
                                task_start_token_id = token_info.get('id')
                                logger.info(f"Найден токен {self.task_start_token} в списке added_tokens.json с ID {task_start_token_id}")
                                break
            except Exception as e:
                logger.warning(f"Ошибка при чтении added_tokens.json: {e}")
        
        if task_start_token_id is None:
            vocab = self.processor.tokenizer.get_vocab()
            if self.task_start_token in vocab:
                task_start_token_id = vocab[self.task_start_token]
                logger.info(f"Найден токен {self.task_start_token} в словаре с ID {task_start_token_id}")
        
        if task_start_token_id is None:
            added_tokens_dict = getattr(self.processor.tokenizer, 'added_tokens_decoder', {})
            print(added_tokens_dict)
            for token_id, token_info in added_tokens_dict.items():
                token_id_int = int(token_id) if isinstance(token_id, str) else token_id
                
                if isinstance(token_info, dict) and token_info.get('content') == self.task_start_token:
                    task_start_token_id = token_id_int
                    logger.info(f"Найден токен {self.task_start_token} в added_tokens_decoder с ID {task_start_token_id}")
                    break
                elif isinstance(token_info, str) and token_info == self.task_start_token:
                    task_start_token_id = token_id_int
                    logger.info(f"Найден токен {self.task_start_token} в added_tokens_decoder с ID {task_start_token_id}")
                    break
        
        if task_start_token_id is None:
            logger.warning(f"Токен {self.task_start_token} не найден, пробую добавить")
            special_tokens_dict = {"additional_special_tokens": [self.task_start_token]}
            num_added = self.processor.tokenizer.add_special_tokens(special_tokens_dict)
            
            if num_added > 0:
                self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
                task_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
                logger.info(f"Добавлен токен {self.task_start_token} с ID {task_start_token_id}")
        
        if task_start_token_id is None or task_start_token_id == self.processor.tokenizer.unk_token_id:
            for token_name, token_id in [
                ("eos_token_id", self.processor.tokenizer.eos_token_id),
                ("bos_token_id", self.processor.tokenizer.bos_token_id),
                ("sep_token_id", getattr(self.processor.tokenizer, "sep_token_id", None)),
                ("cls_token_id", getattr(self.processor.tokenizer, "cls_token_id", None)),
            ]:
                if token_id is not None and token_id != 0:
                    task_start_token_id = token_id
                    logger.warning(f"Использую существующий токен для decoder_start_token_id: {token_name}={token_id}")
                    break
        
        self.model.config.decoder_start_token_id = task_start_token_id
        token_text = self.processor.tokenizer.convert_ids_to_tokens(task_start_token_id)
        logger.info(f"Установлен decoder_start_token_id: {task_start_token_id} ({token_text})")

        if self.prompt_end_token != self.task_start_token:
            prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
            if prompt_end_token_id == self.processor.tokenizer.unk_token_id:
                special_tokens_dict = {"additional_special_tokens": [self.prompt_end_token]}
                num_added = self.processor.tokenizer.add_special_tokens(special_tokens_dict)
                if num_added > 0:
                    self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
                    prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
            
            if prompt_end_token_id != self.processor.tokenizer.unk_token_id:
                logger.info(f"Установлен prompt_end_token_id: {prompt_end_token_id} ({self.prompt_end_token})")
    
    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)

        self.processor.save_pretrained(output_dir)

        config_dict = {
            "task_start_token": self.task_start_token,
            "prompt_end_token": self.prompt_end_token,
            "max_length": self.max_length,
            "precision": self.precision,
        }

        import json
        with open(output_dir / "donut_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Модель и процессор сохранены в {output_dir}")
    
    def get_model_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.model.parameters())
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        import torch.nn.functional as F
        
        pixel_values = pixel_values.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)

            max_len = self.model.config.decoder.max_position_embeddings
            if labels.shape[1] > max_len:
                labels = labels[:, :max_len]
        
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        if labels is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            eos_token_id = self.processor.tokenizer.eos_token_id
            pad_token_id = self.processor.tokenizer.pad_token_id
            task_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            probs = F.softmax(logits, dim=-1)
            batch_size, seq_len = labels.shape
            eos_positions = (labels == eos_token_id)  # [batch_size, seq_len]
            task_start_positions = (labels == task_start_token_id)  # [batch_size, seq_len]
            has_eos = eos_positions.any(dim=1)  # [batch_size]
            positions = torch.arange(seq_len, device=labels.device).expand(batch_size, seq_len)
            first_eos_indices = torch.full((batch_size,), seq_len-1, device=labels.device)
            first_eos_indices[has_eos] = torch.argmax(eos_positions[has_eos].float(), dim=1)
            after_eos_mask = (positions >= first_eos_indices.unsqueeze(1)).float()
            non_pad_probs = torch.sum(probs, dim=2) - probs[:, :, pad_token_id]
            post_eos_penalty = torch.sum(non_pad_probs * after_eos_mask) / batch_size
            has_task_start = task_start_positions.any(dim=1)  # [batch_size]
            missing_task_start_penalty = torch.sum(~has_task_start).float() / batch_size
            has_task_start_expanded = task_start_positions.any(dim=1).unsqueeze(1).expand_as(task_start_positions)
            first_task_start_indices = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
            first_task_start_indices[has_task_start] = torch.argmax(task_start_positions[has_task_start].float(), dim=1)
            early_eos_mask = torch.zeros(batch_size, device=labels.device)
            check_mask = has_task_start & has_eos
            if check_mask.any():
                distance = first_eos_indices[check_mask] - first_task_start_indices[check_mask]
                early_eos_mask[check_mask] = (distance <= 1).float()
            early_eos_penalty = torch.sum(early_eos_mask) / batch_size
            penalty_weight_eos = 0.05
            penalty_weight_missing_task = 0.05
            penalty_weight_early_eos = 0.05
            total_penalty = (
                penalty_weight_eos * post_eos_penalty +
                penalty_weight_missing_task * missing_task_start_penalty +
                penalty_weight_early_eos * early_eos_penalty
            )
            outputs.loss = outputs.loss + total_penalty
        
        return outputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: Optional[str] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        num_beams: int = 3,
        max_length: Optional[int] = None,
        return_json: bool = False,
        **kwargs
    ) -> Union[List[str], torch.Tensor, List[Dict]]:

        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length

        if prompt is not None and decoder_input_ids is None:
            decoded_prompt = self.processor.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            decoder_input_ids = decoded_prompt["input_ids"].to(self.device)
        
        start_token_id = self.model.config.decoder_start_token_id
        token_text = self.processor.tokenizer.convert_ids_to_tokens(start_token_id)

        if decoder_input_ids is None:
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                start_token_id,
                device=self.device
            )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=num_beams,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            **kwargs
        )
        
        decoded_sequences = []
        json_outputs = []
        
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True):
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            decoded_sequences.append(seq)
            
            if return_json:
                try:
                    json_output = self.token2json(seq)
                    json_outputs.append(json_output)
                except Exception as e:
                    logger.warning(f"Ошибка при преобразовании в JSON: {e}")
                    json_outputs.append({"error": "Failed to parse JSON"})
        
        if return_json:
            return json_outputs
        
        return decoded_sequences if not kwargs.get("return_tensors", False) else outputs.sequences

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token sequence into an ordered JSON format
        """
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
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in self.processor.tokenizer.get_added_vocab() and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}
    
    def add_tokens(self, tokens: List[str]) -> int:
        newly_added_num = self.processor.tokenizer.add_tokens(tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            logger.info(f"Добавлено {newly_added_num} новых токенов в токенизатор")
        return newly_added_num
    
    def to(self, device: Union[str, torch.device]) -> "DonutModel":
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.device = device
        logger.info(f"Модель перемещена на устройство: {device}")
        
        return self
    
    def train(self) -> "DonutModel":
        self.model.train()
        return self
    
    def eval(self) -> "DonutModel":
        self.model.eval()
        return self
