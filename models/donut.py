from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
import re
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DonutProcessor,
    PretrainedConfig, 
    PreTrainedModel
)
from transformers.file_utils import ModelOutput
import timm
import numpy as np
from PIL import Image, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate

from core.base import BaseOCRModel

logger = logging.getLogger(__name__)


class SwinEncoder(nn.Module):
    """
    Donut encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations as a Donut Encoder

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self._embed_dim = 128  # Базовая размерность эмбеддингов из конструктора

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=self._embed_dim,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None

        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model("swin_base_patch4_window12_384", pretrained=True).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                    pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                    new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)
    
    @property
    def output_dim(self) -> int:
        """
        Возвращает размерность выходных данных энкодера.
        Для Swin Transformer, размерность увеличивается в 2 раза на каждом этапе.
        """
        # Для Swin Transformer выходная размерность = embed_dim * 2^(число этапов - 1)
        # Число этапов обычно равно длине списка encoder_layer (обычно 4)
        num_stages = len(self.encoder_layer)
        return self._embed_dim * (2 ** (num_stages - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        # Проверяем наличие pos_drop, так как в некоторых версиях timm его может не быть
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    def prepare_input(self, img: Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))


class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
        self, decoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        # Импортируем здесь, чтобы избежать цикличных импортов
        from transformers import MBartConfig, MBartForCausalLM, XLMRobertaTokenizer

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ecjk" if not name_or_path else name_or_path,
            use_fast=True  # Добавляем use_fast=True для избежания предупреждения
        )

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
            )
        )
        self.model.forward = self.forward  #  to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        # weight init with asian-bart - ЗАГРУЖАЕМ ВЕСА ДО ДОБАВЛЕНИЯ СПЕЦИАЛЬНЫХ ТОКЕНОВ
        if not name_or_path:
            bart_state_dict = MBartForCausalLM.from_pretrained("hyunwoongko/asian-bart-ecjk").state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if x.endswith("embed_positions.weight") and self.max_position_embeddings != 1024:
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][: len(self.tokenizer), :]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)
            
        # Только после загрузки весов добавляем специальные токены
        self.add_special_tokens(["<sep/>"])  # <sep/> is used for representing a list in a JSON
    
    @property
    def hidden_size(self) -> int:
        """
        Возвращает размерность скрытого состояния декодера.
        Используется для инициализации новых эмбеддингов из многомерного нормального распределения.
        """
        return self.model.config.d_model

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, past_key_values=None, past=None, use_cache: bool = None, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        # for compatibility with transformers==4.11.x
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,  # Добавляем поддержку decoder_input_ids как alias для input_ids
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            decoder_input_ids: (batch_size, sequence_length) - alias for input_ids
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        # Если передан decoder_input_ids вместо input_ids, используем его
        if input_ids is None and decoder_input_ids is not None:
            input_ids = decoder_input_ids
        
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight


class DonutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "donut"

    def __init__(
        self,
        input_size: List[int] = [2560, 1920],
        align_long_axis: bool = False,
        window_size: int = 10,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 1536,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.name_or_path = name_or_path


class DonutModel(PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = DonutConfig
    base_model_prefix = "donut"

    def __init__(self, config: DonutConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
        )
        self.decoder = BARTDecoder(
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            name_or_path=self.config.name_or_path,
        )

    def forward(self, image_tensors: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(image_tensors)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
        )
        return decoder_outputs

    def inference(
        self,
        image: Image.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
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
                            if (
                                leaf in self.decoder.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(pretrained_model_name_or_path, revision="official", *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model


class DonutOCRModel(BaseOCRModel):
    """
    Адаптер для интеграции DonutModel с архитектурой проекта.
    Этот класс предоставляет совместимость модели Donut с интерфейсом BaseOCRModel.
    """
    def __init__(self, encoder, decoder, config: Dict[str, Any]):
        super().__init__(encoder, decoder, config)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get('max_length', 768)
        self.precision = config.get('precision', 'fp32')
        
        self.task_start_token = config.get("task_start_token", "<s_ocr>")
        self._add_task_tokens()
    
    def _add_task_tokens(self):
        if not hasattr(self, "processor") or self.processor is None:
            return
            
        special_tokens = []
        if self.task_start_token:
            special_tokens.append(self.task_start_token)
        
        prompt_end_token = self.config.get("prompt_end_token", None)
        if prompt_end_token and prompt_end_token != self.task_start_token:
            special_tokens.append(prompt_end_token)
        
        if special_tokens:
            # Добавляем специальные токены в токенизатор
            tokenizer = self.processor.tokenizer
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
            if num_added > 0:
                if hasattr(self.decoder, "model"):
                    # Изменяем размер эмбеддингов модели декодера
                    self.decoder.model.resize_token_embeddings(len(tokenizer))
                    
                    # Проверяем, имеет ли lm_head правильный размер
                    if hasattr(self.decoder.model, 'lm_head'):
                        if self.decoder.model.lm_head.out_features != len(tokenizer):
                            # Создаем новую голову с правильным размером
                            old_lm_head = self.decoder.model.lm_head
                            embedding_dim = old_lm_head.in_features
                            
                            # Создаем новую lm_head с нужным размером
                            new_lm_head = nn.Linear(embedding_dim, len(tokenizer), bias=old_lm_head.bias is not None)
                            
                            # Копируем веса из старой головы
                            with torch.no_grad():
                                new_lm_head.weight[:old_lm_head.out_features, :] = old_lm_head.weight
                                if old_lm_head.bias is not None:
                                    new_lm_head.bias[:old_lm_head.out_features] = old_lm_head.bias
                            
                            # Заменяем старую голову на новую
                            self.decoder.model.lm_head = new_lm_head
                
                logger.info(f"Added {num_added} task tokens and resized token embeddings")
    
    def set_processor(self, processor):
        """Устанавливает процессор для модели"""
        self.processor = processor
        self._add_task_tokens()
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
            
            # Режим обучения - прямая передача через encoder и decoder
            encoder_outputs = self.encoder(pixel_values)
            
            # Shift для автрорегрессивного обучения
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == -100] = 0  # Заменяем ignored_index на PAD
            
            decoder_input_ids = decoder_input_ids[:, :-1].contiguous()
            decoder_labels = labels[:, 1:].contiguous()
            
            outputs = self.decoder(
                encoder_hidden_states=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=decoder_labels
            )
            
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
                "encoder_hidden_states": encoder_outputs
            }
        else:
            # Inference only
            encoder_outputs = self.encoder(pixel_values)
            return {
                "loss": None,
                "logits": None,
                "encoder_hidden_states": encoder_outputs
            }
    
    def generate(self, pixel_values: torch.Tensor, max_length: Optional[int] = None, num_beams: int = 3, **kwargs) -> List[str]:
        pixel_values = pixel_values.to(self.device)
        max_length = max_length or self.max_length
        
        self.eval()
        with torch.no_grad():
            # Получаем энкодер выходы
            encoder_outputs = self.encoder(pixel_values)
            
            # Создаем начальный токен для декодера
            decoder_input_ids = torch.tensor(
                [[self.processor.tokenizer.bos_token_id]] * pixel_values.size(0),
                device=self.device
            )
            
            # Генерируем текст с помощью decoder
            if hasattr(self.decoder, "model") and hasattr(self.decoder.model, "generate"):
                generated_ids = self.decoder.model.generate(
                    decoder_input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    use_cache=True,
                    **kwargs
                )
            else:
                # Базовое авторегрессивное декодирование если нет специального метода generate
                generated_ids = self._greedy_decode(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    max_length=max_length
                )
            
            # Декодируем токены в текст
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text
    
    def _greedy_decode(self, encoder_outputs, decoder_input_ids, max_length):
        batch_size = encoder_outputs.size(0)
        generated_ids = decoder_input_ids
        
        for _ in range(max_length-1):
            outputs = self.decoder(
                encoder_hidden_states=encoder_outputs,
                decoder_input_ids=generated_ids
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # Завершаем генерацию если все последовательности завершились EOS токеном
            if (next_tokens == self.processor.tokenizer.eos_token_id).all():
                break
        
        return generated_ids
    
    def to_device(self, precision: str):
        self.precision = precision
        
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        self.encoder = self.encoder.to(self.device, dtype=dtype)
        self.decoder = self.decoder.to(self.device, dtype=dtype)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        logger.info("Encoder parameters frozen")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = []
        for component in [self.encoder, self.decoder]:
            params.extend([p for p in component.parameters() if p.requires_grad])
        return params
    
    def save_pretrained(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем компоненты
        encoder_dir = output_dir / "encoder"
        encoder_dir.mkdir(exist_ok=True)
        torch.save(self.encoder.state_dict(), encoder_dir / "pytorch_model.bin")
        
        decoder_dir = output_dir / "decoder"
        decoder_dir.mkdir(exist_ok=True)
        torch.save(self.decoder.state_dict(), decoder_dir / "pytorch_model.bin")
        
        # Сохраняем конфигурацию
        with open(output_dir / "donut_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_size": self.config.get("image_size", [384, 384]),
                    "align_long_axis": self.config.get("align_long_axis", False),
                    "window_size": self.config.get("window_size", 10),
                    "encoder_layer": self.config.get("encoder_layer", [2, 2, 14, 2]),
                    "decoder_layer": self.config.get("decoder_layer", 4),
                    "max_position_embeddings": self.config.get("max_position_embeddings", None),
                    "max_length": self.max_length,
                    "task_start_token": self.task_start_token,
                    "prompt_end_token": self.config.get("prompt_end_token", None)
                },
                f,
                indent=2
            )
        
        # Сохраняем процессор если есть
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path], **kwargs) -> "DonutOCRModel":
        model_dir = Path(model_dir)
        config = kwargs.get('config', {})
        
        # Загружаем конфигурацию модели
        config_path = model_dir / "donut_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                config.update(saved_config)
        
        # Создаем экземпляры encoder и decoder
        encoder = SwinEncoder(
            input_size=config.get("image_size", [384, 384]),
            align_long_axis=config.get("align_long_axis", False),
            window_size=config.get("window_size", 10),
            encoder_layer=config.get("encoder_layer", [2, 2, 14, 2])
        )
        
        decoder = BARTDecoder(
            decoder_layer=config.get("decoder_layer", 4),
            max_position_embeddings=config.get("max_position_embeddings", config.get("max_length", 768))
        )
        
        # Загружаем веса если они есть
        encoder_path = model_dir / "encoder" / "pytorch_model.bin"
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        
        decoder_path = model_dir / "decoder" / "pytorch_model.bin"
        if decoder_path.exists():
            decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        
        # Создаем модель
        model = cls(encoder, decoder, config)
        
        # Загружаем процессор
        try:
            from transformers import DonutProcessor
            model.processor = DonutProcessor.from_pretrained(model_dir)
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
        
        return model