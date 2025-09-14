#!/usr/bin/env python

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import torch
import tensorrt as trt
import torch_tensorrt
import numpy as np
from PIL import Image

from models.donut import DonutOCRModel
from models.trocr import TrOCROCRModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEFAULT_BATCH_SIZE = 1
DEFAULT_HEIGHT = 224
DEFAULT_WIDTH = 224


def parse_args():
    parser = argparse.ArgumentParser(description='Convert OCR models to TensorRT')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model directory')
    
    parser.add_argument('--model_type', type=str, required=True, choices=['donut', 'trocr'],
                      help='Type of OCR model to convert')
    
    parser.add_argument('--method', type=str, default='torch_tensorrt', choices=['onnx', 'torch_tensorrt'],
                      help='Conversion method: through ONNX or directly with torch-tensorrt')
    
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for TensorRT engine files')
    
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'],
                      help='Precision to use for TensorRT engine')
    
    parser.add_argument('--max_batch_size', type=int, default=1,
                      help='Maximum batch size for the TensorRT engine')
    
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                      help=f'Input image height (default: {DEFAULT_HEIGHT})')
    
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                      help=f'Input image width (default: {DEFAULT_WIDTH})')
    
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()


def load_model(model_path: Union[str, Path], model_type: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not torch.cuda.is_available() and device == 'cuda':
        logger.warning("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    model_path = Path(model_path)
    config_path = model_path / "model_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if model_type.lower() == 'donut':
        logger.info(f"Loading DonutOCRModel from {model_path}")
        model = DonutOCRModel.from_pretrained(model_path)
    elif model_type.lower() == 'trocr':
        logger.info(f"Loading TrOCROCRModel from {model_path}")
        model = TrOCROCRModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.eval()
    model = model.to(device)
    
    return model, config


def convert_model_to_onnx(model: torch.nn.Module, 
                         input_shape: Tuple[int, int, int, int],
                         onnx_path: str, 
                         verbose: bool = False) -> str:
    logger.info(f"Converting model to ONNX format, input shape: {input_shape}")
    
    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
    
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    else:
        logger.warning("Model does not have a separate encoder attribute, using full model")
        encoder = model
    
    torch.onnx.export(
        encoder,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=verbose
    )
    
    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path


def build_engine_from_onnx(onnx_path: str, 
                          engine_path: str,
                          precision: str = 'fp16',
                          max_batch_size: int = 1,
                          verbose: bool = False) -> None:
    logger_verbosity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    trt_logger = trt.Logger(logger_verbosity)
    
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(f"TensorRT ONNX parser error: {parser.get_error(error)}")
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = input_tensor.shape
    
    profile.set_shape(
        input_tensor.name,
        (1, shape[1], shape[2], shape[3]),
        (max_batch_size, shape[1], shape[2], shape[3]),
        (max_batch_size, shape[1], shape[2], shape[3])
    )
    
    config.add_optimization_profile(profile)
    
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Building TensorRT engine with FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Building TensorRT engine with INT8 precision")
    else:
        logger.info("Building TensorRT engine with FP32 precision")
    
    logger.info("Building TensorRT engine - this might take a while...")
    engine = builder.build_engine(network, config)
    
    if not engine:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"TensorRT engine saved to {engine_path}")


def convert_model_to_torch_tensorrt(model: torch.nn.Module,
                                   input_shape: Tuple[int, int, int, int],
                                   engine_path: str,
                                   precision: str = 'fp16',
                                   max_batch_size: int = 1) -> None:
    logger.info(f"Converting model to TensorRT with torch-tensorrt, input shape: {input_shape}")
    
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    else:
        logger.warning("Model does not have a separate encoder attribute, using full model")
        encoder = model
    
    enabled_precisions = {torch.float32}
    if precision == 'fp16':
        enabled_precisions.add(torch.float16)
    elif precision == 'int8':
        enabled_precisions.add(torch.int8)
    
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, input_shape[1], input_shape[2], input_shape[3]],
            opt_shape=[1, input_shape[1], input_shape[2], input_shape[3]],
            max_shape=[max_batch_size, input_shape[1], input_shape[2], input_shape[3]],
            dtype=torch.float32
        )
    ]
    
    trt_model = torch_tensorrt.compile(
        encoder,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=1 << 30,
    )
    
    torch.jit.save(trt_model, engine_path)
    logger.info(f"TensorRT engine (torch-tensorrt format) saved to {engine_path}")


def save_model_metadata(output_dir: Union[str, Path], 
                       model_config: Dict[str, Any], 
                       input_shape: Tuple[int, int, int, int],
                       precision: str,
                       model_type: str) -> None:
    metadata = {
        'model_type': model_type,
        'precision': precision,
        'input_shape': list(input_shape),
        'orig_config': model_config
    }
    
    output_path = Path(output_dir) / 'engine_metadata.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {output_path}")


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.model_path, args.model_type, device)
    
    input_shape = (args.max_batch_size, 3, args.height, args.width)
    
    if args.method == 'onnx':
        onnx_path = str(output_dir / f"{args.model_type}_model.onnx")
        engine_path = str(output_dir / f"{args.model_type}_model.engine")
        
        convert_model_to_onnx(model, input_shape, onnx_path, args.verbose)
        build_engine_from_onnx(onnx_path, engine_path, args.precision, args.max_batch_size, args.verbose)
    else:
        engine_path = str(output_dir / f"{args.model_type}_model.engine")
        convert_model_to_torch_tensorrt(model, input_shape, engine_path, args.precision, args.max_batch_size)
    
    save_model_metadata(output_dir, config, input_shape, args.precision, args.model_type)
    
    if hasattr(model, 'processor') and model.processor is not None:
        logger.info(f"Saving processor files to {output_dir}")
        model.processor.save_pretrained(output_dir)
    elif hasattr(model, 'tokenizer') and model.tokenizer is not None:
        logger.info(f"Saving tokenizer files to {output_dir}")
        model.tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model successfully converted to TensorRT format and saved to {output_dir}")


if __name__ == "__main__":
    main()