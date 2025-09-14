#!/usr/bin/env python

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import torch
import tensorrt as trt
import torch_tensorrt
from PIL import Image
import cv2
from transformers import AutoTokenizer, DonutProcessor, TrOCRProcessor

from models.donut import DonutDecoder
from models.trocr import TrOCRDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorRTEngine:
    def __init__(self, engine_path: Union[str, Path], cuda_device: int = 0):
        self.engine_path = Path(engine_path)
        self.cuda_device = cuda_device
        self.engine = None
        self.context = None
        self.stream = None
        self.host_inputs = None
        self.host_outputs = None
        self.device_inputs = None
        self.device_outputs = None
        self.bindings = None

        self._load_engine()
        self._create_context()
        self._allocate_buffers()
    
    def _load_engine(self):
        logger.info(f"Loading TensorRT engine from {self.engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
    
    def _create_context(self):
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        np.copyto(self.host_inputs[0], input_data.ravel())
        
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        
        self.stream.synchronize()
        
        output_shape = self.context.get_binding_shape(1)
        output = self.host_outputs[0].reshape(output_shape)
        
        return output


class TorchTensorRTEngine:
    def __init__(self, engine_path: Union[str, Path], device: str = "cuda"):
        self.engine_path = Path(engine_path)
        self.device = torch.device(device)
        self.model = None
        
        self._load_engine()
    
    def _load_engine(self):
        logger.info(f"Loading Torch-TensorRT engine from {self.engine_path}")
        self.model = torch.jit.load(self.engine_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            return output


class TensorRTInference:
    def __init__(
        self,
        engine_dir: Union[str, Path],
        model_type: str,
        engine_type: str = "torch_tensorrt",
        device: str = "cuda",
        max_length: int = 64,
    ):
        self.engine_dir = Path(engine_dir)
        self.model_type = model_type
        self.engine_type = engine_type
        self.device = device
        self.max_length = max_length
        
        self.metadata = None
        self.engine = None
        self.decoder = None
        self.processor = None
        self.tokenizer = None
        
        self._load_metadata()
        self._load_engine()
        self._load_processor()
        self._load_decoder()
    
    def _load_metadata(self):
        metadata_path = self.engine_dir / "engine_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {self.metadata}")
    
    def _load_engine(self):
        engine_path = self.engine_dir / f"{self.model_type}_model.engine"
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found at {engine_path}")
        
        if self.engine_type == "torch_tensorrt":
            self.engine = TorchTensorRTEngine(engine_path, self.device)
        else:
            self.engine = TensorRTEngine(engine_path)
    
    def _load_processor(self):
        if self.model_type == "donut":
            try:
                self.processor = DonutProcessor.from_pretrained(self.engine_dir)
                self.tokenizer = self.processor.tokenizer
            except Exception as e:
                logger.warning(f"Failed to load DonutProcessor: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.engine_dir)
        elif self.model_type == "trocr":
            try:
                self.processor = TrOCRProcessor.from_pretrained(self.engine_dir)
                self.tokenizer = self.processor.tokenizer
            except Exception as e:
                logger.warning(f"Failed to load TrOCRProcessor: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.engine_dir)
    
    def _load_decoder(self):
        config = self.metadata["orig_config"]
        
        if self.model_type == "donut":
            self.decoder = DonutDecoder({**config, 'decoder_name': config.get('decoder_name')})
            
            decoder_path = self.engine_dir / "decoder.pt"
            if decoder_path.exists():
                self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
                self.decoder = self.decoder.to(torch.device(self.device))
                self.decoder.eval()
            else:
                logger.warning(f"Decoder weights not found at {decoder_path}")
        
        elif self.model_type == "trocr":
            self.decoder = TrOCRDecoder({**config, 'decoder_name': config.get('decoder_name')})
            
            decoder_path = self.engine_dir / "decoder.pt"
            if decoder_path.exists():
                self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
                self.decoder = self.decoder.to(torch.device(self.device))
                self.decoder.eval()
            else:
                logger.warning(f"Decoder weights not found at {decoder_path}")
        
        lm_head_path = self.engine_dir / "lm_head.pt"
        if hasattr(self, 'lm_head') and lm_head_path.exists():
            self.lm_head = torch.load(lm_head_path, map_location=self.device)
            self.lm_head = self.lm_head.to(torch.device(self.device))
        elif self.model_type == "trocr":
            self.lm_head = torch.nn.Linear(self.decoder.hidden_size, self.decoder.vocab_size, bias=False)
            if lm_head_path.exists():
                self.lm_head.load_state_dict(torch.load(lm_head_path, map_location=self.device))
                self.lm_head = self.lm_head.to(torch.device(self.device))
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            image = Image.fromarray(image)
        
        if self.processor:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
        else:
            image = image.convert('RGB').resize((224, 224))
            image_array = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
            pixel_values = torch.tensor(image_array).unsqueeze(0)
        
        return pixel_values
    
    def infer(self, image: Union[np.ndarray, Image.Image, torch.Tensor, str], prompt: Optional[str] = None) -> str:
        start_time = time.time()
        
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert("RGB")
            pixel_values = self.preprocess_image(image)
        elif isinstance(image, torch.Tensor):
            pixel_values = image
        else:
            pixel_values = self.preprocess_image(image)
        
        encoder_hidden_states = self.engine.infer(pixel_values)
        
        if isinstance(encoder_hidden_states, np.ndarray):
            encoder_hidden_states = torch.tensor(encoder_hidden_states).to(torch.device(self.device))
        
        if prompt is not None and self.tokenizer is not None:
            decoder_input_ids = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(torch.device(self.device))
        else:
            decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(torch.device(self.device))
        
        with torch.no_grad():
            generated_ids = self._beam_search_decode(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                num_beams=4,
            )
            
            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        logger.debug(f"Inference completed in {inference_time:.4f} seconds")
        
        return decoded_text
    
    def _beam_search_decode(
        self, 
        encoder_hidden_states: torch.Tensor, 
        decoder_input_ids: torch.Tensor, 
        max_length: int, 
        num_beams: int = 4
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        
        sequences = decoder_input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                decoder_outputs = self.decoder(encoder_hidden_states=encoder_hidden_states, decoder_input_ids=sequences)
                
                if hasattr(self, 'lm_head'):
                    logits = self.lm_head(decoder_outputs)
                else:
                    logits = decoder_outputs
                
                next_token_logits = logits[:, -1, :]
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                sequences = torch.cat([sequences, next_token_ids], dim=1)
                
                if (next_token_ids == self.tokenizer.eos_token_id).all():
                    break
        
        return sequences


def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Inference for OCR models")
    
    parser.add_argument("--engine_dir", type=str, required=True,
                      help="Directory containing TensorRT engine and metadata")
    
    parser.add_argument("--model_type", type=str, required=True, choices=["donut", "trocr"],
                      help="Type of OCR model")
    
    parser.add_argument("--engine_type", type=str, default="torch_tensorrt",
                      choices=["tensorrt", "torch_tensorrt"],
                      help="Type of TensorRT engine")
    
    parser.add_argument("--image_path", type=str, default=None,
                      help="Path to input image for inference")
    
    parser.add_argument("--numpy_file", type=str, default=None,
                      help="Path to numpy array file (.npy) for inference")
    
    parser.add_argument("--prompt", type=str, default=None,
                      help="Optional text prompt for the model")
    
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to save inference results (JSON format)")
    
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run inference on (cuda or cpu)")
    
    parser.add_argument("--max_length", type=int, default=64,
                      help="Maximum sequence length for generation")
    
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark to measure inference speed")
    
    parser.add_argument("--benchmark_iterations", type=int, default=10,
                      help="Number of iterations for benchmark")
    
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize inference results")
    
    return parser.parse_args()


def load_image(image_path: Union[str, Path]) -> Image.Image:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    return image


def benchmark(inference_engine: TensorRTInference, image: Union[Image.Image, np.ndarray, torch.Tensor], iterations: int = 10) -> Dict[str, float]:
    logger.info(f"Running benchmark with {iterations} iterations")
    
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        preprocessed_image = inference_engine.preprocess_image(image)
    else:
        preprocessed_image = image
    
    inference_engine.infer(preprocessed_image)
    
    total_time = 0
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        inference_engine.infer(preprocessed_image)
        end_time = time.time()
        
        iteration_time = end_time - start_time
        times.append(iteration_time)
        total_time += iteration_time
        
        logger.debug(f"Iteration {i+1}/{iterations}: {iteration_time:.4f} seconds")
    
    avg_time = total_time / iterations
    fps = 1 / avg_time
    
    times = np.array(times)
    p50 = np.percentile(times, 50) * 1000
    p95 = np.percentile(times, 95) * 1000
    p99 = np.percentile(times, 99) * 1000
    
    logger.info(f"Benchmark results:")
    logger.info(f"  Average inference time: {avg_time:.4f} seconds")
    logger.info(f"  Throughput: {fps:.2f} FPS")
    logger.info(f"  p50 latency: {p50:.2f} ms")
    logger.info(f"  p95 latency: {p95:.2f} ms")
    logger.info(f"  p99 latency: {p99:.2f} ms")
    
    return {
        "avg_time": avg_time,
        "fps": fps,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99
    }


def visualize_results(image: Union[Image.Image, np.ndarray], text: str, output_path: Optional[str] = None) -> None:
    try:
        from visualization.inference import InferenceVisualizer
        
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                raise ValueError("Image array should be RGB with shape (H, W, 3)")
        
        visualizer = InferenceVisualizer()
        result_image = visualizer.visualize_ocr_result(
            image=image,
            text_prediction=text,
            save_path=output_path if output_path else None
        )
        
        if output_path is None:
            result_image.show()
        
    except ImportError:
        logger.warning("Visualization module not available. Results will not be visualized.")
        
        if output_path:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
                
            cv2.putText(
                img_array,
                text[:50],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            
            cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    
    inference_engine = TensorRTInference(
        engine_dir=args.engine_dir,
        model_type=args.model_type,
        engine_type=args.engine_type,
        device=args.device,
        max_length=args.max_length
    )
    
    if args.image_path:
        input_data = load_image(args.image_path)
    elif args.numpy_file:
        input_data = np.load(args.numpy_file)
    else:
        raise ValueError("Either --image_path or --numpy_file must be provided")
    
    if args.benchmark:
        benchmark_results = benchmark(
            inference_engine=inference_engine,
            image=input_data,
            iterations=args.benchmark_iterations
        )
    
    result_text = inference_engine.infer(input_data, prompt=args.prompt)
    logger.info(f"Inference result: {result_text}")
    
    if args.visualize:
        output_path = args.output_path if args.output_path else None
        visualize_results(input_data, result_text, output_path)
    
    if args.output_path and not args.visualize:
        output_dir = Path(args.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "text": result_text,
            "model_type": args.model_type,
            "engine_type": args.engine_type,
            "timestamp": time.time()
        }
        
        if args.benchmark:
            result["benchmark"] = benchmark_results
        
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    try:
        from cuda import cuda
        main()
    except ImportError:
        logger.error("CUDA Python bindings not found. Please install cuda-python package.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise