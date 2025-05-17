import logging
import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# from transformers import VisionEncoderDecoderModel # Not used directly in the TRT class
from model import DonutModel # Assuming 'model' contains your DonutModel
from inference import TextCleanup # Assuming 'inference' contains TextCleanup

# --- TensorRT and PyCUDA Imports ---
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit # This initializes CUDA, must be imported before trt.Runtime
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT and/or PyCUDA are not installed. TensorRT inference is not available.")

# Only try to create TRT objects if TRT is available
if TRT_AVAILABLE:
    # Need to re-get the logger after autoinit might have affected the context
    trt_logger = trt.Logger(trt.Logger.WARNING)
    # The builder is typically for building engines, not strictly needed for inference only
    # builder = trt.Builder(trt_logger)
    # print(f"TensorRT version: {trt.__version__}")
    # # Check CUDA version via PyCUDA or external means, builder might not reflect all CUDA capabilities
    # print(f"CUDA initialized: {cuda.Context.get_current() is not None}")
    # print(f"Available devices (via PyCUDA):")
    # for i in range(cuda.Device.count()):
    #     dev = cuda.Device(i)
    #     print(f"  Device {i}: {dev.name()}")
    #     # Note: num_DLA_cores is a builder property, not a runtime device property
    #     # if hasattr(builder, 'num_DLA_cores'): # This check is only relevant during build
    #     #     print(f"    DLA Cores (requires builder and specific hardware): N/A in runtime check")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TensorRTDonutInference:

    def __init__(
        self,
        path_to_checkpoint: Union[str, Path],
        encoder_engine_path: Union[str, Path],
        decoder_engine_path: Union[str, Path],
        model_info_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 64,
        batch_size: int = 1,
        prompt: str = None,
    ):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT and PyCUDA are required for TensorRT inference.")

        # Ensure CUDA context is initialized before creating TRT Runtime/Engine
        cuda.Context.get_current()

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if self.device.type == 'cpu':
            raise ValueError("TensorRT inference requires a CUDA-enabled device.")

        # Load model config/processor from checkpoint path
        # Note: DonutModel.from_pretrained might load the full torch model temporarily.
        # Ensure it's deleted afterwards to free GPU memory if needed before loading TRT engines.
        model_path = Path(path_to_checkpoint) if isinstance(path_to_checkpoint, str) else path_to_checkpoint
        try:
            model = DonutModel.from_pretrained(
                model_path,
                device="cpu", # Load to CPU first to save GPU memory before TRT
                max_length=max_length
            )
            self.task_start_token = model.task_start_token
            self.prompt_end_token = getattr(model, 'prompt_end_token', self.task_start_token)
            # Ensure processor uses the correct tokenizer properties
            self.processor = model.processor
            self.eos_token_id = self.processor.tokenizer.eos_token_id # Use eos_token_id directly
            del model # Free up torch model memory
            torch.cuda.empty_cache() # Clear cache just in case
        except Exception as e:
            logger.error(f"Ошибка при загрузке процессора и токенов из чекпоинта: {e}", exc_info=True)
            raise RuntimeError("Не удалось загрузить процессор и токены из чекпоинта.")


        self.input_prompt = self.prepare_prompt(prompt)

        self.encoder_engine_path = Path(encoder_engine_path)
        self.decoder_engine_path = Path(decoder_engine_path)
        self.model_info_path = Path(model_info_path)

        if not self.encoder_engine_path.exists():
            raise FileNotFoundError(f"Encoder engine file not found: {self.encoder_engine_path}")
        if not self.decoder_engine_path.exists():
            raise FileNotFoundError(f"Decoder engine file not found: {self.decoder_engine_path}")
        if not self.model_info_path.exists():
            default_model_info = self.encoder_engine_path.parent / "model_info.json"
            if default_model_info.exists():
                self.model_info_path = default_model_info
                logger.warning(f"model_info.json not found at specified path. Using default: {self.model_info_path}")
            else:
                raise FileNotFoundError(f"model_info.json not found at {self.model_info_path} or default location ({default_model_info}).")

        self.max_length = max_length
        self.batch_size = batch_size # This will be potentially overridden by model_info

        logger.info(f"Initializing TensorRT inference engine...")

        try:
            with open(self.model_info_path, 'r') as f:
                self.model_info = json.load(f)

            # Use batch size from model_info if present and different
            if 'optimization_batch_size' in self.model_info:
                 engine_batch_size = self.model_info['optimization_batch_size']
                 if engine_batch_size != self.batch_size:
                     logger.warning(f"Batch size specified ({self.batch_size}) differs from engine batch size ({engine_batch_size}) in model_info.json. Using engine batch size.")
                 self.batch_size = engine_batch_size
            else:
                logger.warning("optimization_batch_size not found in model_info.json. Using batch size specified during initialization.")


            self.encoder_hidden_size = self.model_info.get('encoder_hidden_size')
            self.encoder_seq_length = self.model_info.get('encoder_seq_length')
            self.vocab_size = self.model_info.get('vocab_size')
            original_model_path = self.model_info.get('model_path') # For info purposes
            # Ensure essential info is present
            if self.encoder_hidden_size is None or self.encoder_seq_length is None or self.vocab_size is None:
                 raise ValueError("Essential keys (encoder_hidden_size, encoder_seq_length, vocab_size) missing in model_info.json")

        except Exception as e:
            logger.error(f"Error loading model_info.json: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model_info.json from {self.model_info_path}")

        logger.info(f"Configuration: Device: {self.device}, Max generation length: {max_length}, Engine Batch Size: {self.batch_size}")
        logger.info(f"Model Info: Encoder Hidden Size: {self.encoder_hidden_size}, Encoder Seq Length: {self.encoder_seq_length}, Vocab Size: {self.vocab_size}")
        logger.info(f"Tokens: task_start_token='{self.task_start_token}', prompt_end_token='{self.prompt_end_token}', eos_token_id={self.eos_token_id}")


        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        # runtime creation requires initialized CUDA context
        self.runtime = trt.Runtime(self.trt_logger)

        try:
            with open(self.encoder_engine_path, 'rb') as f:
                self.encoder_engine = self.runtime.deserialize_cuda_engine(f.read())
            logger.info(f"TensorRT encoder engine loaded: {self.encoder_engine_path}")

            with open(self.decoder_engine_path, 'rb') as f:
                self.decoder_engine = self.runtime.deserialize_cuda_engine(f.read())
            logger.info(f"TensorRT decoder engine loaded: {self.decoder_engine_path}")

        except Exception as e:
            logger.error(f"Error loading TensorRT engines: {e}", exc_info=True)
            self.destroy() # Clean up partially loaded resources
            raise RuntimeError("Failed to load TensorRT engines.")

        # Bindings information
        self.encoder_bindings_info = {}
        for i in range(self.encoder_engine.num_io_tensors): # Use num_io_tensors
            name = self.encoder_engine.get_tensor_name(i)
            shape = self.encoder_engine.get_tensor_shape(name)
            dtype = self.encoder_engine.get_tensor_dtype(name)
            # mode = self.encoder_engine.get_tensor_mode(name) # TRT 10+
            is_input = self.encoder_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT # TRT 10+
            # For TRT < 10, use get_binding_is_input
            # try: # Check for older API if get_tensor_mode fails
            #     is_input = self.encoder_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            # except AttributeError: # Fallback for TRT < 10
            #     is_input = self.encoder_engine.get_binding_is_input(i)

            self.encoder_bindings_info[name] = {'shape': shape, 'dtype': dtype, 'is_input': is_input, 'index': i}
            logger.info(f"Encoder binding: {name}, Shape: {shape}, Dtype: {dtype}, IsInput: {is_input}")

        self.decoder_bindings_info = {}
        for i in range(self.decoder_engine.num_io_tensors): # Use num_io_tensors
            name = self.decoder_engine.get_tensor_name(i)
            shape = self.decoder_engine.get_tensor_shape(name)
            dtype = self.decoder_engine.get_tensor_dtype(name)
            is_input = self.decoder_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT # TRT 10+
            # try: # Check for older API if get_tensor_mode fails
            #     is_input = self.decoder_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            # except AttributeError: # Fallback for TRT < 10
            #      is_input = self.decoder_engine.get_binding_is_input(i)

            self.decoder_bindings_info[name] = {'shape': shape, 'dtype': dtype, 'is_input': is_input, 'index': i}
            logger.info(f"Decoder binding: {name}, Shape: {shape}, Dtype: {dtype}, IsInput: {is_input}")

        # Identify input/output names - handle potential variations
        self.encoder_input_name = None
        self.encoder_output_name = None
        for name, info in self.encoder_bindings_info.items():
            if info['is_input']:
                self.encoder_input_name = name
            else:
                self.encoder_output_name = name
        if self.encoder_input_name is None or self.encoder_output_name is None:
             raise ValueError("Could not identify encoder input or output binding.")

        self.decoder_input_ids_name = None
        self.decoder_encoder_hidden_states_name = None
        self.decoder_output_name = None
        for name, info in self.decoder_bindings_info.items():
             if info['is_input']:
                 # Use more robust check as names can vary (e.g. "input_ids" or "decoder_input_ids")
                 if 'input_ids' in name.lower():
                     self.decoder_input_ids_name = name
                 elif 'hidden' in name.lower() or 'encoder_outputs' in name.lower():
                     self.decoder_encoder_hidden_states_name = name
             else:
                 # Use more robust check for output name (e.g. "logits" or "output")
                 if 'logits' in name.lower() or 'output' in name.lower():
                    self.decoder_output_name = name

        if self.decoder_input_ids_name is None or self.decoder_encoder_hidden_states_name is None or self.decoder_output_name is None:
             raise ValueError("Could not identify all required decoder bindings (input_ids, encoder_hidden_states, output).")

        # --- Allocate Device Buffers ---
        self.device_buffers = {}
        try:
            self.stream = cuda.Stream() # Create CUDA stream

            # Allocate for encoder bindings
            for name, info in self.encoder_bindings_info.items():
                shape = list(info['shape']) # Make mutable
                dtype = info['dtype']
                item_size = self.trt_dtype_to_itemsize(dtype)

                # Handle dynamic dimensions (-1) based on expected max sizes
                # Assuming dynamic dims are Batch, Height, Width for input
                # and Batch, Sequence Length for output
                for i, dim in enumerate(shape):
                    if dim == -1:
                         if name == self.encoder_input_name:
                             # Assuming input is (Batch, Channels, Height, Width)
                             if i == 0: shape[i] = self.batch_size
                             elif i == 2: shape[i] = 384 # Max height (adjust if needed)
                             elif i == 3: shape[i] = 384 # Max width (adjust if needed)
                             else: raise ValueError(f"Unexpected dynamic dimension {i} for encoder input {name}")
                         elif name == self.encoder_output_name:
                             # Assuming output is (Batch, Sequence Length, Hidden Size)
                             if i == 0: shape[i] = self.batch_size
                             elif i == 1: shape[i] = self.encoder_seq_length # Use value from model_info
                             else: raise ValueError(f"Unexpected dynamic dimension {i} for encoder output {name}")
                         else:
                             logger.warning(f"Dynamic dimension found for unknown encoder tensor {name} at index {i}. Using default 1.")
                             shape[i] = 1 # Default for unknown dynamic dims
                concrete_shape = tuple(shape)


                size_in_bytes = int(np.prod(concrete_shape) * item_size)
                logger.info(f"Allocating memory for encoder tensor {name}: Shape: {concrete_shape}, Dtype: {dtype}, Size: {size_in_bytes} bytes")
                self.device_buffers[name] = cuda.mem_alloc(size_in_bytes)


            # Allocate for decoder bindings
            for name, info in self.decoder_bindings_info.items():
                shape = list(info['shape']) # Make mutable
                dtype = info['dtype']
                item_size = self.trt_dtype_to_itemsize(dtype)

                # Handle dynamic dimensions (-1) based on expected max sizes
                # Assuming dynamic dims are Batch, Sequence Length for input_ids and output
                # and Batch, Sequence Length, Hidden Size for encoder_outputs/hidden_states
                for i, dim in enumerate(shape):
                    if dim == -1:
                         if name == self.decoder_input_ids_name:
                             # Assuming decoder input_ids is (Batch, Sequence Length)
                             if i == 0: shape[i] = self.batch_size
                             elif i == 1: shape[i] = self.max_length # Max length for generation
                             else: raise ValueError(f"Unexpected dynamic dimension {i} for decoder input_ids {name}")
                         elif name == self.decoder_encoder_hidden_states_name:
                             # Assuming encoder_hidden_states is (Batch, Sequence Length, Hidden Size)
                             if i == 0: shape[i] = self.batch_size
                             elif i == 1: shape[i] = self.encoder_seq_length # Encoder output seq length
                             elif i == 2: shape[i] = self.encoder_hidden_size # Encoder hidden size
                             else: raise ValueError(f"Unexpected dynamic dimension {i} for decoder encoder_hidden_states {name}")
                         elif name == self.decoder_output_name:
                             # Assuming decoder output is (Batch, Sequence Length, Vocab Size)
                             if i == 0: shape[i] = self.batch_size
                             elif i == 1: shape[i] = self.max_length # Max length for generation
                             elif i == 2: shape[i] = self.vocab_size # Vocab size
                             else: raise ValueError(f"Unexpected dynamic dimension {i} for decoder output {name}")
                         else:
                             logger.warning(f"Dynamic dimension found for unknown decoder tensor {name} at index {i}. Using default 1.")
                             shape[i] = 1 # Default for unknown dynamic dims
                concrete_shape = tuple(shape)


                size_in_bytes = int(np.prod(concrete_shape) * item_size)
                logger.info(f"Allocating memory for decoder tensor {name}: Shape: {concrete_shape}, Dtype: {dtype}, Size: {size_in_bytes} bytes")
                self.device_buffers[name] = cuda.mem_alloc(size_in_bytes)

            # Bind the allocated buffers to the contexts
            # This uses the TRT 10+ API with set_tensor_address
            # Need to check if set_tensor_address is available, if not,
            # need to fallback to TRT < 10 context creation where bindings are passed directly.
            # Assuming TRT 10+ is available given the logs and the use of set_tensor_shape/get_tensor_shape later.

            self.encoder_context = self.encoder_engine.create_execution_context()
            self.decoder_context = self.decoder_engine.create_execution_context()

            decoder_methods = [m for m in dir(self.decoder_context) if not m.startswith('_')]
            logger.info(f"Available methods on decoder_context: {decoder_methods}")

            if hasattr(self.encoder_context, 'set_tensor_address'):
                 # TRT 10+ binding
                 logger.info("Using TRT 10+ tensor address binding.")
                 for name, buffer in self.device_buffers.items():
                     if name in self.encoder_bindings_info:
                         self.encoder_context.set_tensor_address(name, int(buffer))
                     if name in self.decoder_bindings_info:
                         self.decoder_context.set_tensor_address(name, int(buffer))
            else:
                 # TRT < 10 binding - requires passing bindings during context creation or execute_async_v2/v3
                 # The code below uses execute_async_v2/v3 with bindings list, so no global set_tensor_address needed.
                 logger.info("Using TRT < 10 binding method (bindings list for execute).")


        except Exception as e:
            logger.error(f"Error allocating device memory or creating CUDA stream/contexts: {e}", exc_info=True)
            self.destroy()
            raise

    def prepare_prompt(self, prompt: Optional[str] = None) -> str:
        if prompt is None or prompt.strip() == "":
            # Use only task_start_token if no specific prompt
            return self.task_start_token
        else:
            # Combine task_start_token, prompt, and prompt_end_token
            # Check if prompt_end_token is different from task_start_token before appending
            if self.prompt_end_token != self.task_start_token:
                return f"{self.task_start_token}{prompt}{self.prompt_end_token}"
            else:
                 # If prompt_end_token is the same as task_start_token, just use task_start_token + prompt
                 # This handles cases where the model doesn't explicitly define a prompt_end_token
                 return f"{self.task_start_token}{prompt}"


    def destroy(self):
        logger.info("Releasing TensorRT and CUDA resources...")

        # Release contexts
        if hasattr(self, 'encoder_context') and self.encoder_context:
            del self.encoder_context
            self.encoder_context = None
        if hasattr(self, 'decoder_context') and self.decoder_context:
            del self.decoder_context
            self.decoder_context = None

        # Release engines
        if hasattr(self, 'encoder_engine') and self.encoder_engine:
            del self.encoder_engine
            self.encoder_engine = None
        if hasattr(self, 'decoder_engine') and self.decoder_engine:
            del self.decoder_engine
            self.decoder_engine = None

        # Release runtime
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime
            self.runtime = None

        # Synchronize and release stream (if still valid)
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.synchronize()
            except Exception as e:
                 logger.warning(f"Error synchronizing CUDA stream during destroy: {e}")
            self.stream = None # PyCUDA streams don't have a explicit destroy/free method, just let Python GC handle it after context/driver is gone.

        # Release device buffers
        # Buffers are released by cuda.Context.pop() or when the context is destroyed
        # explicit buffer.free() is also an option if done before context destruction.
        # Let's explicitly free the buffers for clarity, assuming context is still valid or handled by autoinit cleanup.
        # However, cuda.Context.pop() is the standard way if autoinit was used.
        # Given autoinit, the context is managed outside this class's direct creation/destruction.
        # Explicitly freeing buffers is safer if context lifecycle isn't strictly tied to this object.
        # Try to free buffers if they exist
        for name, buffer in self.device_buffers.items():
             try:
                 if buffer:
                     buffer.free()
                     self.device_buffers[name] = None
             except Exception as e:
                 logger.warning(f"Error freeing device buffer {name}: {e}")
        self.device_buffers = {} # Clear the dictionary

        # The PyCUDA autoinit context is typically popped when the script exits.
        # If not using autoinit, you'd need cuda.Context.pop() here.
        # With autoinit, relying on script exit for final context cleanup is standard.

        logger.info("Resources released.")


    def prepare_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Ensure image processor size matches engine expected size (e.g., 384x384 or 224x224)
        # The actual engine input shape determines this. Check encoder_bindings_info.
        encoder_input_shape = self.encoder_bindings_info[self.encoder_input_name]['shape']
        # Assuming shape is (Batch, Channels, Height, Width)
        expected_height = encoder_input_shape[2] if encoder_input_shape[2] != -1 else 384 # Use a common default if dynamic
        expected_width = encoder_input_shape[3] if encoder_input_shape[3] != -1 else 384 # Use a common default if dynamic
        self.processor.image_processor.size = (expected_height, expected_width)
        logger.info(f"Image processor size set to: {self.processor.image_processor.size}")


        # Process image to get pixel values
        # The processor typically returns torch.Tensor, convert to numpy
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.cpu().numpy()

        # Ensure data type matches expected engine input type
        expected_dtype_trt = self.encoder_bindings_info[self.encoder_input_name]['dtype']
        # Convert TRT dtype to numpy dtype
        if expected_dtype_trt == trt.DataType.FLOAT:
            expected_dtype_np = np.float32
        elif expected_dtype_trt == trt.DataType.HALF:
             expected_dtype_np = np.float16
        else:
             logger.warning(f"Unexpected encoder input dtype: {expected_dtype_trt}. Defaulting to float32.")
             expected_dtype_np = np.float32

        if pixel_values.dtype != expected_dtype_np:
             logger.info(f"Converting image pixel values dtype from {pixel_values.dtype} to {expected_dtype_np}")
             pixel_values = pixel_values.astype(expected_dtype_np)

        # Ensure batch size matches the engine's expected batch size
        if pixel_values.shape[0] != self.batch_size:
             if pixel_values.shape[0] == 1 and self.batch_size > 1:
                 # Repeat the single image for batching if necessary
                 pixel_values = np.repeat(pixel_values, self.batch_size, axis=0)
                 logger.warning(f"Repeating image batch size from 1 to {self.batch_size} to match engine batch size.")
             else:
                 # This scenario (image batch size > 1 but different from engine batch size) might not be supported
                 # or indicates an issue. For simplicity, handle only 1 -> N batching.
                 logger.error(f"Image batch size ({pixel_values.shape[0]}) does not match engine batch size ({self.batch_size}) and cannot be automatically adjusted.")
                 raise ValueError(f"Image batch size mismatch. Got {pixel_values.shape[0]}, expected {self.batch_size}")


        # Ensure array is C-contiguous for CUDA memcpy
        if not pixel_values.flags['C_CONTIGUOUS']:
             logger.info("Converting image array to C-contiguous format.")
             pixel_values = np.ascontiguousarray(pixel_values)

        return pixel_values

    def prepare_initial_decoder_input(self, prompt: Optional[str] = None) -> np.ndarray:
        # Use the stored prepared prompt
        initial_prompt_tokens = self.processor.tokenizer(
            self.input_prompt,
            add_special_tokens=False, # Special tokens like BOS/EOS are handled by the model/tokenizer logic
            return_tensors="pt"
        )["input_ids"]

        # Ensure batch size matches the engine's expected batch size
        if initial_prompt_tokens.shape[0] != self.batch_size:
            if initial_prompt_tokens.shape[0] == 1 and self.batch_size > 1:
                # Repeat the single prompt for batching if necessary
                initial_prompt_tokens = initial_prompt_tokens.repeat(self.batch_size, 1)
                logger.warning(f"Repeating initial decoder input batch size from 1 to {self.batch_size} to match engine batch size.")
            else:
                logger.error(f"Initial decoder input batch size ({initial_prompt_tokens.shape[0]}) does not match engine batch size ({self.batch_size}).")
                raise ValueError(f"Initial decoder input batch size mismatch. Got {initial_prompt_tokens.shape[0]}, expected {self.batch_size}")


        # Ensure data type matches expected engine input type (usually INT32)
        input_name = self.decoder_input_ids_name
        expected_dtype_trt = self.decoder_bindings_info[input_name]['dtype']
        # Convert TRT dtype to numpy dtype
        if expected_dtype_trt == trt.DataType.INT32:
            expected_dtype_np = np.int32
        elif expected_dtype_trt == trt.DataType.INT64: # Some engines might use INT64
             expected_dtype_np = np.int64
        else:
             logger.warning(f"Unexpected decoder input_ids dtype: {expected_dtype_trt}. Defaulting to int32.")
             expected_dtype_np = np.int32

        initial_decoder_input_np = initial_prompt_tokens.cpu().numpy().astype(expected_dtype_np)

        # Ensure array is C-contiguous for CUDA memcpy
        if not initial_decoder_input_np.flags['C_CONTIGUOUS']:
             logger.info("Converting initial decoder input array to C-contiguous format.")
             initial_decoder_input_np = np.ascontiguousarray(initial_decoder_input_np)

        return initial_decoder_input_np

    def postprocess_output(self, token_ids: np.ndarray, return_json: bool = True) -> Union[str, Dict[str, Any]]:
        # Post-processing typically operates on a single sequence.
        # If batch size > 1, process the first one or handle batching in postprocess.
        # The current TextCleanup seems designed for a single sequence.
        if token_ids.shape[0] > 1:
             logger.warning("Post-processing applied only to the first sample in the batch.")
             token_ids_single = token_ids[0:1, :] # Take the first sample and keep it as a batch of 1
        else:
             token_ids_single = token_ids


        # Decode tokens back to text, keeping special tokens initially
        # Ensure tokenizer can handle numpy arrays or convert to list/tensor if needed
        # Hugging Face tokenizers typically accept list of lists or torch.Tensor
        decoded_text = self.processor.tokenizer.batch_decode(token_ids_single.tolist(), skip_special_tokens=False)[0] # Decode the batch (size 1), get the first string

        # Clean up and potentially parse into JSON
        if return_json:
            try:
                # TextCleanup.extract_fields_from_donut_output expects a single string
                result = TextCleanup.extract_fields_from_donut_output(decoded_text)
                # Validate if the result is a dictionary, otherwise fallback
                if not isinstance(result, dict):
                     logger.warning("TextCleanup.extract_fields_from_donut_output did not return a dictionary. Returning as cleaned text.")
                     result = TextCleanup.cleanup_donut_output(decoded_text)
                     if isinstance(result, str):
                          result = {"text_sequence": result} # Wrap in dict if it's just a string
            except Exception as e:
                logger.warning(f"Error during JSON conversion using TextCleanup: {e}. Returning as cleaned text.", exc_info=True)
                # Fallback to cleanup_donut_output
                cleaned_text = TextCleanup.cleanup_donut_output(decoded_text)
                result = {"text_sequence": cleaned_text} if isinstance(cleaned_text, str) else cleaned_text # Ensure it's a dict

        else:
            # Return just the cleaned text string
            result = TextCleanup.cleanup_donut_output(decoded_text)
            # If cleanup_donut_output returns something else, handle it
            if not isinstance(result, str):
                 logger.warning("TextCleanup.cleanup_donut_output did not return a string. Returning its output directly.")

        return result

    @staticmethod
    def trt_dtype_to_itemsize(dtype):
        if dtype == trt.DataType.FLOAT: return 4
        elif dtype == trt.DataType.HALF: return 2
        elif dtype == trt.DataType.INT32: return 4
        elif dtype == trt.DataType.INT64: return 8 # TRT 10+ might have INT64
        elif dtype == trt.DataType.BOOL: return 1
        else: return 4 # Default to 4 bytes (e.g., for INT8 or custom types, might need adjustment)

    def get_buffer_size(self, device_allocation) -> int:
        """
        Get the size in bytes of a CUDA device allocation.
        
        Args:
            device_allocation: A PyCUDA DeviceAllocation object
            
        Returns:
            int: The size of the buffer in bytes
        """
        if not device_allocation:
            return 0
            
        # Method 1: Use the size property if available (in some CUDA versions)
        if hasattr(device_allocation, 'size'):
            return device_allocation.size
            
        # Method 2: Using PyCUDA's memory info function
        mem_info = cuda.mem_get_info()
        
        # Attempt to get size from binding info based on shape and data type
        # Find the key in self.device_buffers that matches this device_allocation
        for name, buffer in self.device_buffers.items():
            if buffer is device_allocation:
                if name in self.encoder_bindings_info:
                    info = self.encoder_bindings_info[name]
                    shape = info['shape']
                    dtype = info['dtype']
                    return int(np.prod(shape) * self.trt_dtype_to_itemsize(dtype))
                elif name in self.decoder_bindings_info:
                    info = self.decoder_bindings_info[name]
                    shape = info['shape']
                    dtype = info['dtype']
                    return int(np.prod(shape) * self.trt_dtype_to_itemsize(dtype))
        
        # Method 3: Fall back to a default size or calculate from the pointer difference
        # This is a last resort and might not be accurate
        return (1 << 20)  # Assume 1MB as a fallback (you should adjust based on your model)

    def infer(self, image: Union[str, Path, Image.Image], prompt: Optional[str] = None, 
          max_new_tokens: int = 64, return_json: bool = True) -> Union[str, Dict[str, Any]]:
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Cannot perform inference.")

        # Ensure CUDA context is current for this thread
        cuda.Context.get_current()

        logger.info(f"Starting inference for image...")

        # --- Prepare Input Data ---
        image_np = self.prepare_image(image)
        initial_decoder_input_ids_np = self.prepare_initial_decoder_input(prompt)

        # --- Encoder Inference ---
        logger.info("Running encoder...")
        start_time_encoder = time.time()

        self.encoder_context.set_input_shape(self.encoder_input_name, image_np.shape)
        cuda.memcpy_htod_async(self.device_buffers[self.encoder_input_name], image_np, self.stream)
        self.encoder_context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()
        end_time_encoder = time.time()
        logger.info(f"Encoder finished in {end_time_encoder - start_time_encoder:.4f} s")

        encoder_output_buffer_ptr = int(self.device_buffers[self.encoder_output_name])
        decoder_input_buffer_ptr = int(self.device_buffers[self.decoder_encoder_hidden_states_name])

        if encoder_output_buffer_ptr != decoder_input_buffer_ptr:
            logger.warning("Encoder output buffer and decoder input buffer are different. Performing device-to-device copy.")
            encoder_output_shape_after_inf = self.encoder_context.get_tensor_shape(self.encoder_output_name)
            encoder_output_dtype = self.encoder_bindings_info[self.encoder_output_name]['dtype']
            item_size = self.trt_dtype_to_itemsize(encoder_output_dtype)
            encoder_output_size_bytes = int(np.prod(encoder_output_shape_after_inf) * item_size)

            decoder_input_shape = self.decoder_bindings_info[self.decoder_encoder_hidden_states_name]['shape']
            decoder_input_dtype = self.decoder_bindings_info[self.decoder_encoder_hidden_states_name]['dtype']
            decoder_input_buffer_size_bytes = int(np.prod(decoder_input_shape) * self.trt_dtype_to_itemsize(decoder_input_dtype))

            if encoder_output_size_bytes > decoder_input_buffer_size_bytes:
                logger.error(f"Decoder input buffer for {self.decoder_encoder_hidden_states_name} ({decoder_input_buffer_size_bytes} bytes) is too small for encoder output ({encoder_output_size_bytes} bytes).")
                raise RuntimeError("Decoder input buffer is too small for encoder output.")

            cuda.memcpy_dtod_async(self.device_buffers[self.decoder_encoder_hidden_states_name],
                                    self.device_buffers[self.encoder_output_name],
                                    encoder_output_size_bytes,
                                    self.stream)
            self.stream.synchronize()
            logger.debug("Finished device-to-device copy of encoder output to decoder input.")
        else:
            logger.debug("Encoder output buffer and decoder input buffer are the same. No device-to-device copy needed.")

        logger.info(f"Starting text generation (max {max_new_tokens} tokens)...")
        start_time_decoder = time.time()

        generated_ids = initial_decoder_input_ids_np.copy()
        dec_input_ids_info = self.decoder_bindings_info[self.decoder_input_ids_name]
        dec_enc_h_states_info = self.decoder_bindings_info[self.decoder_encoder_hidden_states_name]
        dec_output_info = self.decoder_bindings_info[self.decoder_output_name]

        # Get the size information for decoder input buffer
        input_shape = dec_input_ids_info['shape']
        input_dtype = dec_input_ids_info['dtype']
        decoder_input_buffer_size_bytes = self.get_buffer_size(self.device_buffers[self.decoder_input_ids_name])

        for _ in range(max_new_tokens):
            current_seq_length = generated_ids.shape[1]
            if current_seq_length >= self.max_length:
                logger.warning(f"Maximum sequence length ({self.max_length}) reached. Stopping generation.")
                break

            current_decoder_input_ids_np = generated_ids[:, :]

            dec_input_buffer = self.device_buffers[self.decoder_input_ids_name]
            required_size_bytes = current_decoder_input_ids_np.nbytes
            
            # Ensure the buffer is large enough
            if required_size_bytes > decoder_input_buffer_size_bytes:
                logger.error(f"Decoder input_ids buffer ({decoder_input_buffer_size_bytes} bytes) is too small for current sequence ({required_size_bytes} bytes). Increase allocated buffer size.")
                raise RuntimeError("Decoder input_ids buffer overflow during generation.")

            cuda.memcpy_htod_async(dec_input_buffer, current_decoder_input_ids_np, self.stream)

            current_decoder_input_shape = current_decoder_input_ids_np.shape  # (Batch, CurrentSeqLength)
            encoder_hidden_states_shape = (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size)  # This is static per image

            self.decoder_context.set_input_shape(self.decoder_input_ids_name, current_decoder_input_shape)
            self.decoder_context.set_input_shape(self.decoder_encoder_hidden_states_name, encoder_hidden_states_shape)

            self.decoder_context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

            output_shape_after_step = self.decoder_context.get_tensor_shape(self.decoder_output_name)

            h_decoder_output = np.zeros(output_shape_after_step, dtype=np.float32)
            output_dtype_trt = dec_output_info['dtype']
            if output_dtype_trt == trt.DataType.HALF:
                h_decoder_output = np.zeros(output_shape_after_step, dtype=np.float16)
            elif output_dtype_trt != trt.DataType.FLOAT:
                logger.warning(f"Unexpected decoder output dtype: {output_dtype_trt}. Expecting FLOAT or HALF for logits.")

            # Get the size of the decoder output buffer
            output_buffer = self.device_buffers[self.decoder_output_name]
            output_buffer_size = self.get_buffer_size(output_buffer)
            required_output_size_bytes = h_decoder_output.nbytes
            
            if required_output_size_bytes > output_buffer_size:
                logger.error(f"Decoder output buffer ({output_buffer_size} bytes) is too small for current step output ({required_output_size_bytes} bytes). Increase allocated buffer size.")
                raise RuntimeError("Decoder output buffer overflow during generation step.")

            cuda.memcpy_dtoh_async(h_decoder_output, output_buffer, self.stream)
            self.stream.synchronize()

            last_token_logits = h_decoder_output[:, -1, :]
            next_token_ids = np.argmax(last_token_logits, axis=-1)
            generated_ids = np.concatenate([generated_ids, next_token_ids[:, np.newaxis]], axis=1)
            if generated_ids[0, -1] == self.eos_token_id:
                logger.info("EOS token detected in the first sample. Stopping generation.")
                break

        end_time_decoder = time.time()
        logger.info(f"Generation finished in {end_time_decoder - start_time_decoder:.4f} s")
        total_time = end_time_decoder - start_time_encoder
        logger.info(f"Total inference time: {total_time:.4f} s")

        result = self.postprocess_output(generated_ids, return_json=return_json)

        return result


# --- Main execution block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT Donut Inference")
    parser.add_argument("--path_to_checkpoint", type=str, required=True,
                        help="Path to the Hugging Face checkpoint files (contains processor/tokenizer config)")
    parser.add_argument("--encoder_engine", type=str, required=True,
                        help="Path to the TensorRT encoder engine file (.engine)")
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--decoder_engine", type=str,
                        help="Path to the TensorRT decoder engine file (.engine) for independent generation.")
    # parser_group.add_argument("--combined_engine", type=str, # Example for a combined engine case
    #                     help="Path to a combined TensorRT engine (.engine) if encoder and decoder are fused.")
    parser.add_argument("--model_info", type=str, default=None,
                        help="Path to the model_info.json file created during conversion. Defaults to 'model_info.json' in encoder engine directory.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image file for processing")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional prompt string for the model (e.g., '<s_cord-v2>')")
    parser.add_argument("--max_new_tokens", type=int, default=256, # Increased default slightly
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--no_json", action="store_true",
                        help="Do not attempt to parse the output into JSON, return cleaned text instead")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Expected batch size of the TensorRT engines. Will be overridden by model_info.json if present.")


    args = parser.parse_args()

    # Check for CUDA availability early
    if not torch.cuda.is_available():
         logger.error("CUDA is not available. TensorRT inference requires a GPU. Exiting.")
         sys.exit(1)
    # Ensure PyCUDA and TensorRT were imported successfully
    if not TRT_AVAILABLE:
         logger.error("TensorRT and/or PyCUDA are not available. Exiting.")
         sys.exit(1)

    device = torch.device("cuda") # Already checked availability


    inference_engine = None
    try:
        # Determine model_info_path if not provided
        model_info_path = args.model_info
        if model_info_path is None:
             encoder_engine_dir = Path(args.encoder_engine).parent
             model_info_path = encoder_engine_dir / "model_info.json"
             logger.info(f"model_info path not specified, defaulting to: {model_info_path}")


        inference_engine = TensorRTDonutInference(
            path_to_checkpoint=args.path_to_checkpoint,
            encoder_engine_path=args.encoder_engine,
            decoder_engine_path=args.decoder_engine, # Assumes separate engines
            model_info_path=model_info_path,
            device=device,
            max_length=args.max_new_tokens, # Pass max_new_tokens as max_length for init logic
            batch_size=args.batch_size
        )

        logger.info(f"Processing image: {args.image_path}")
        start_time = time.time()
        prediction_result = inference_engine.infer(
            image=args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            return_json=not args.no_json
        )
        end_time = time.time()
        logger.info(f"Total inference time (including data prep and postproc): {end_time - start_time:.4f} s")

        print("\nPrediction Result:")
        if isinstance(prediction_result, dict):
            # Use json.dumps for pretty printing dictionaries/JSON objects
            print(json.dumps(prediction_result, ensure_ascii=False, indent=2))
        else:
            # Print raw string output
            print(prediction_result)

    except FileNotFoundError as e:
        logger.error(f"Error: Required file not found: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error during initialization or inference: {e}")
    except ValueError as e:
        logger.error(f"Configuration or data mismatch error: {e}")
    except NotImplementedError as e:
         logger.error(f"TensorRT API not supported error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during inference: {e}", exc_info=True)

    finally:
        # Ensure resources are cleaned up even if errors occur
        if inference_engine:
            logger.info("Cleaning up resources.")
            inference_engine.destroy()
        else:
            logger.info("Inference engine was not initialized, no resources to clean up.")