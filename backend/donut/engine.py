import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn.functional as F
import json
import re
import time
import numpy as np

import torch
from transformers import (
    VisionEncoderDecoderConfig, 
    VisionEncoderDecoderModel, 
    DonutProcessor,
    PreTrainedTokenizer
)

import torch_tensorrt
import random
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



class TextCleanup:
    @staticmethod
    def cleanup_donut_output(text):
        text = re.sub(r"<s_([^>]*)>", "", text)
        text = re.sub(r"</s_[^>]*>", "", text)
        text = text.replace("<sep/>", ", ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_fields_from_donut_output(text):
        output = {}
        
        while text:
            start_token = re.search(r"<s_(.*?)>", text, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", text, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                text = text.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", text, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = TextCleanup.extract_fields_from_donut_output(content)
                        if value:
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                text = text[text.find(end_token) + len(end_token):].strip()

        if not output:
            return {"text_sequence": text.strip()}
        return output



class DonutInferenceTRT:
    """Класс для инференса с TensorRT моделью."""
    
    def __init__(
        self,
        tensorrt_dir: str,
        device: str = "cuda",
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        self.tensorrt_dir = Path(tensorrt_dir)
        self.model_path = self.tensorrt_dir.parent
        self.device = device
        self.batch_size = batch_size
        self.config_path = self.tensorrt_dir / f"tensorrt_config.json"
        self.temperature = temperature
        self.top_k = top_k
        
        self._resources_initialized = False
        self._cuda_context_valid = True
        self._tensorrt_objects_valid = True
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.base_name = self.config.get('base_name', 'donut')
        self.task_start_token = self.config['task_start_token']
        self.decoder_start_token_id = self.config.get('decoder_start_token_id')
        self.prompt_end_token = self.config.get('prompt_end_token', self.task_start_token)
        self.max_length = self.config['max_length']
        self.image_size = tuple(self.config['image_size'])

        self.encoder_engine_path = self.tensorrt_dir / f"{self.base_name}_encoder.engine"
        self.decoder_engine_path = self.tensorrt_dir / f"{self.base_name}_decoder.engine"
        
        if not self.encoder_engine_path.exists():
            raise FileNotFoundError(f"Encoder engine not found: {self.encoder_engine_path}")
        if not self.decoder_engine_path.exists():
            raise FileNotFoundError(f"Decoder engine not found: {self.decoder_engine_path}")
        
        from transformers import DonutProcessor
        self.processor = DonutProcessor.from_pretrained(self.model_path)
        
        self.encoder_engine = None
        self.decoder_engine = None
        self.encoder_context = None
        self.decoder_context = None
        
        self._init_tensorrt()
        logger.info(f"Инициализирован TensorRT инференс для {tensorrt_dir}")
    
    def _init_tensorrt(self):
        """Инициализация TensorRT для версии 10.9.0.34."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.cuda = cuda
            self.cuda_available = True
            self.trt = trt
        except ImportError as e:
            logger.error(f"Не удалось импортировать TensorRT или PyCUDA: {e}")
            raise ImportError("TensorRT или PyCUDA не установлены")
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(self.encoder_engine_path, 'rb') as f:
            encoder_engine_data = f.read()
        self.encoder_engine = runtime.deserialize_cuda_engine(encoder_engine_data)
        self.encoder_context = self.encoder_engine.create_execution_context()
        
        with open(self.decoder_engine_path, 'rb') as f:
            decoder_engine_data = f.read()
        self.decoder_engine = runtime.deserialize_cuda_engine(decoder_engine_data)
        self.decoder_context = self.decoder_engine.create_execution_context()
        
        logger.info(f"TensorRT engines loaded")
        
        self.cuda_stream = cuda.Stream()
        
        self.encoder_inputs = {}
        self.encoder_outputs = {}
        self.encoder_buffers = {}
        self.encoder_gpu_buffers = {}
        
        for i in range(self.encoder_engine.num_io_tensors):
            name = self.encoder_engine.get_tensor_name(i)
            mode = self.encoder_engine.get_tensor_mode(name)
            shape = tuple(self.encoder_engine.get_tensor_shape(name))
            
            logger.info(f"Encoder tensor {i}: {name}, mode={mode}, shape={shape}")
            
            tensor_info = {
                'mode': mode,
                'shape': shape,
                'dtype': np.float32
            }
            
            if mode == trt.TensorIOMode.INPUT:
                self.encoder_inputs[name] = tensor_info
                if 'pixel_values' in name.lower() or i == 0:
                    self.encoder_input_name = name
                    self.encoder_input_shape = shape
                size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
                gpu_buffer = cuda.mem_alloc(size)
                self.encoder_gpu_buffers[name] = gpu_buffer
                self.encoder_buffers[name] = np.empty(shape, dtype=np.float32)
            else:
                self.encoder_outputs[name] = tensor_info
                size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
                gpu_buffer = cuda.mem_alloc(size)
                self.encoder_gpu_buffers[name] = gpu_buffer
                self.encoder_buffers[name] = np.empty(shape, dtype=np.float32)
                
                if 'encoder_outputs' in name.lower() or 'hidden' in name.lower():
                    self.encoder_output_name = name
                    self.encoder_output_shape = shape
        
        self.decoder_inputs = {}
        self.decoder_outputs = {}
        self.decoder_gpu_buffers = {}
        
        for i in range(self.decoder_engine.num_io_tensors):
            name = self.decoder_engine.get_tensor_name(i)
            mode = self.decoder_engine.get_tensor_mode(name)
            shape = tuple(self.decoder_engine.get_tensor_shape(name))
            
            logger.info(f"Decoder tensor {i}: {name}, mode={mode}, shape={shape}")
            
            if mode == trt.TensorIOMode.INPUT:
                tensor_info = {
                    'mode': mode,
                    'shape': shape,
                    'dtype': np.int32 if 'input_ids' in name.lower() else np.float32
                }
                self.decoder_inputs[name] = tensor_info
                
                if 'input_ids' in name.lower():
                    self.decoder_input_ids_name = name
                    self.decoder_input_ids_shape = shape
                elif 'encoder_hidden' in name.lower() or 'hidden' in name.lower():
                    self.decoder_encoder_hidden_name = name
                    self.decoder_encoder_hidden_shape = shape
            else:
                tensor_info = {
                    'mode': mode,
                    'shape': shape,
                    'dtype': np.float32
                }
                self.decoder_outputs[name] = tensor_info
                
                if 'logits' in name.lower() or i == 0:
                    self.decoder_output_name = name
                    self.decoder_output_shape = shape
        
        self._resources_initialized = True
    
    def cleanup(self):
        """Явное освобождение CUDA ресурсов."""
        if not self._resources_initialized:
            return
            
        try:
            if self._cuda_context_valid:
                try:
                    self.cuda.Context.get_current()
                    if hasattr(self, 'encoder_gpu_buffers'):
                        for buffer in self.encoder_gpu_buffers.values():
                            if buffer:
                                buffer.free()
                        self.encoder_gpu_buffers.clear()
                    if hasattr(self, 'decoder_gpu_buffers'):
                        for buffer in self.decoder_gpu_buffers.values():
                            if buffer:
                                buffer.free()
                        self.decoder_gpu_buffers.clear()
                    logger.info("CUDA буферы успешно освобождены")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении CUDA буферов: {e}")
                    self._cuda_context_valid = False
            if self._tensorrt_objects_valid:
                try:
                    if hasattr(self, 'encoder_context') and self.encoder_context is not None:
                        del self.encoder_context
                        self.encoder_context = None
                    if hasattr(self, 'decoder_context') and self.decoder_context is not None:
                        del self.decoder_context
                        self.decoder_context = None
                    if hasattr(self, 'encoder_engine') and self.encoder_engine is not None:
                        del self.encoder_engine
                        self.encoder_engine = None
                    if hasattr(self, 'decoder_engine') and self.decoder_engine is not None:
                        del self.decoder_engine
                        self.decoder_engine = None
                    logger.info("TensorRT объекты успешно освобождены")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении TensorRT объектов: {e}")
                self._tensorrt_objects_valid = False
            
            if self._cuda_context_valid:
                try:
                    if hasattr(self, 'cuda_stream'):
                        del self.cuda_stream
                    logger.info("CUDA stream успешно освобожден")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении CUDA stream: {e}")
            
            self._cuda_context_valid = False
            self._resources_initialized = False
            logger.info("Все ресурсы успешно освобождены")
            
        except Exception as e:
            logger.warning(f"Общая ошибка при освобождении ресурсов: {e}")
            self._cuda_context_valid = False
            self._tensorrt_objects_valid = False
            self._resources_initialized = False

    def __del__(self):
        """Очистка ресурсов CUDA."""
        try:
            if self._resources_initialized:
                self.cleanup()
        except Exception as e:
            pass
    
    def __enter__(self):
        """Поддержка context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Поддержка context manager."""
        self.cleanup()

    def _set_binding_shapes(self, context, binding_shapes):
        """Устанавливает формы binding для контекста."""
        for binding_name, shape in binding_shapes.items():
            context.set_binding_shape(binding_name, shape)
    
    def predict_batch(self, pixel_values: torch.Tensor) -> List[str]:
        """Генерирует предсказания для пакета изображений с TensorRT."""
        pixel_values = pixel_values.to('cpu').numpy()
        batch_size = pixel_values.shape[0]
        
        predictions = []
        
        for i in range(batch_size):
            try:
                encoder_outputs = self._run_encoder(pixel_values[i:i+1])
                pred = self._generate_single(encoder_outputs)
                logger.info(f"Предсказание для изображения {i}: {pred}")
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Ошибка при генерации для изображения {i}: {e}")
                predictions.append("")
        
        return predictions
    
    def _generate_single(self, encoder_outputs: np.ndarray) -> str:
        """Генерирует предсказание для одного изображения."""
        
        if self.decoder_start_token_id is None:
            try:
                self.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
                if self.decoder_start_token_id == self.processor.tokenizer.unk_token_id:
                    self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
            except:
                self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        
        # logger.info(f"Используем decoder_start_token_id: {decoder_start_token_id}")
        
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int32)
        
        generated_tokens = []
        max_length = self.max_length
        
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        
        for step in range(max_length):
            try:
                logits = self._run_decoder(encoder_outputs, decoder_input_ids)
                next_token_logits = logits[0, -1, :]
                next_token_logits = next_token_logits / self.temperature
                top_k_indices = np.argpartition(next_token_logits, -self.top_k)[-self.top_k:]
                top_k_logits = next_token_logits[top_k_indices]
                exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token_idx = np.argmax(probs)
                next_token_id = int(top_k_indices[next_token_idx])
                
                if next_token_id == eos_token_id:
                    # logger.info(f"Встретили EOS токен на шаге {step}")
                    break
                
                if pad_token_id is not None and next_token_id == pad_token_id:
                    # logger.info(f"Встретили PAD токен на шаге {step}")
                    break
                
                if len(generated_tokens) >= 3 and all(t == next_token_id for t in generated_tokens[-3:]):
                    # logger.warning(f"Обнаружено зацикливание на токене {next_token_id}, останавливаем генерацию")
                    break
                
                generated_tokens.append(next_token_id)
                
                new_input = np.array([[next_token_id]], dtype=np.int32)
                decoder_input_ids = np.concatenate([decoder_input_ids, new_input], axis=1)
                    
            except Exception as e:
                logger.error(f"Ошибка на шаге {step}: {e}")
                break
        
        if generated_tokens:
            try:
                pred_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                pred_text = pred_text.strip()
            except Exception as e:
                logger.error(f"Ошибка при декодировании токенов: {e}")
                pred_text = ""
        else:
            logger.warning("Не сгенерировано ни одного токена!")
            pred_text = ""

        # logger.info(f"Сгенерированные токены: {generated_tokens[:10]}{'...' if len(generated_tokens) > 10 else ''}")
        # logger.info(f"Сгенерированный текст: '{pred_text}'")
        
        return pred_text
    
    def _run_encoder(self, pixel_values: np.ndarray) -> np.ndarray:
        """Запуск encoder через TensorRT."""
        if pixel_values.shape != self.encoder_input_shape:
            pixel_values = pixel_values.reshape(self.encoder_input_shape)
        pixel_values = pixel_values.astype(np.float32)
        
        try:
            self.cuda.memcpy_htod_async(
                self.encoder_gpu_buffers[self.encoder_input_name], 
                pixel_values, 
                self.cuda_stream
            )
            
            for name, gpu_buffer in self.encoder_gpu_buffers.items():
                self.encoder_context.set_tensor_address(name, int(gpu_buffer))
            
            success = self.encoder_context.execute_async_v3(self.cuda_stream.handle)
            if not success:
                raise RuntimeError("Encoder inference failed")
            
            self.cuda_stream.synchronize()
            
            output_buffer = self.encoder_buffers[self.encoder_output_name]
            self.cuda.memcpy_dtoh_async(
                output_buffer, 
                self.encoder_gpu_buffers[self.encoder_output_name], 
                self.cuda_stream
            )
            self.cuda_stream.synchronize()
            
            return output_buffer.copy()
            
        except Exception as e:
            logger.error(f"Ошибка в encoder: {e}")
            raise
    
    def _run_decoder(self, encoder_outputs: np.ndarray, decoder_input_ids: np.ndarray) -> np.ndarray:
        """Запуск decoder через TensorRT."""
        encoder_outputs = encoder_outputs.astype(np.float32)
        decoder_input_ids = decoder_input_ids.astype(np.int32)
        self.decoder_context.set_input_shape(self.decoder_input_ids_name, decoder_input_ids.shape)
        
        decoder_gpu_buffers = {}
        decoder_cpu_buffers = {}
        
        for name, tensor_info in self.decoder_inputs.items():
            if name == self.decoder_input_ids_name:
                shape = decoder_input_ids.shape
                data = decoder_input_ids
            else:
                shape = encoder_outputs.shape
                data = encoder_outputs
            
            size = int(np.prod(shape) * np.dtype(tensor_info['dtype']).itemsize)
            gpu_buffer = self.cuda.mem_alloc(size)
            decoder_gpu_buffers[name] = gpu_buffer
            
            self.cuda.memcpy_htod_async(gpu_buffer, data, self.cuda_stream)
        
        for name, tensor_info in self.decoder_outputs.items():
            actual_shape = self.decoder_context.get_tensor_shape(name)
            size = int(np.prod(actual_shape) * np.dtype(tensor_info['dtype']).itemsize)
            gpu_buffer = self.cuda.mem_alloc(size)
            decoder_gpu_buffers[name] = gpu_buffer
            decoder_cpu_buffers[name] = np.empty(actual_shape, dtype=tensor_info['dtype'])
        
        try:
            for name, gpu_buffer in decoder_gpu_buffers.items():
                self.decoder_context.set_tensor_address(name, int(gpu_buffer))
            
            success = self.decoder_context.execute_async_v3(self.cuda_stream.handle)
            if not success:
                raise RuntimeError("Decoder inference failed")
            
            self.cuda_stream.synchronize()
            
            output_buffer = decoder_cpu_buffers[self.decoder_output_name]
            self.cuda.memcpy_dtoh_async(
                output_buffer, 
                decoder_gpu_buffers[self.decoder_output_name], 
                self.cuda_stream
            )
            self.cuda_stream.synchronize()
            
            return output_buffer.copy()
            
        finally:
            for buffer in decoder_gpu_buffers.values():
                buffer.free()