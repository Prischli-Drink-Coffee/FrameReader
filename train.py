#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
from torch.utils.data import DataLoader

from core.config import ConfigManager, ModelConfig, TrainingConfig, DataConfig, VisualizationConfig
from models.donut import DonutOCRModel
from models.trocr import TrOCROCRModel
from data.dataset import DonutDataset, TrOCRDataset
from training.trainer import TwoStageTrainer
from visualization.inference import InferenceVisualizer
from visualization.realtime_inference import RealtimeInferenceEngine, TrainingInferenceDisplayer
from visualization.attention import AttentionVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FrameReaderTrainingPipeline:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_manager = self._load_or_create_config(config_path)
        self.model = None
        self.trainer = None
        self.visualizer = InferenceVisualizer()
        self.realtime_engine = None
        self.inference_displayer = None
    
    def _load_or_create_config(self, config_path: Optional[Path]) -> ConfigManager:
        if config_path and config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            config_manager = ConfigManager.load_all(config_path)
            logger.info(f"Loaded model_name_or_path: {config_manager.model.model_name_or_path}")
            logger.info(f"Loaded model_type: {config_manager.model.model_type}")
            logger.info(f"Loaded all model config: {config_manager.model.to_dict()}")
            return config_manager
        logger.info("Creating default configuration")
        config_manager = ConfigManager()
        logger.info(f"Default model_name_or_path: {config_manager.model.model_name_or_path}")
        return config_manager
    
    def setup_model(self, model_type: str = "donut") -> None:
        model_config = self.config_manager.model
        
        if model_type.lower() == "donut":
            self.model = self._setup_donut_model(model_config)
        elif model_type.lower() == "trocr":
            self.model = self._setup_trocr_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model {model_type} setup complete")
        self._setup_realtime_inference(model_type)
    
    def _setup_donut_model(self, config: ModelConfig) -> DonutOCRModel:
        from transformers import DonutProcessor
        
        logger.info(f"Setting up Donut model with config: {config.to_dict()}")
        logger.info(f"model_name_or_path before check: '{config.model_name_or_path}' (type: {type(config.model_name_or_path)})")
        
        # Проверяем, что model_name_or_path не None и не пустая строка
        model_name_or_path = config.model_name_or_path
        if not model_name_or_path:
            logger.error(f"model_name_or_path is None or empty: '{model_name_or_path}'")
            logger.error(f"Full config dict: {config.to_dict()}")
            raise ValueError(f"model_name_or_path is not set in configuration: {model_name_or_path}")
        
        logger.info(f"Loading DonutProcessor from: {model_name_or_path}")
        # Добавляем use_fast=True для исправления предупреждения
        processor = DonutProcessor.from_pretrained(model_name_or_path, use_fast=True)
        
        model_config_dict = {
            'encoder_name': config.encoder_name,
            'decoder_name': config.decoder_name,
            'max_length': config.max_length,
            'task_start_token': config.task_start_token,
            'enable_gradient_checkpointing': config.enable_gradient_checkpointing
        }
        
        if config.model_name_or_path and Path(config.model_name_or_path).exists():
            model = DonutOCRModel.from_pretrained(config.model_name_or_path)
        else:
            from models.donut import DonutEncoder, DonutDecoder
            encoder = DonutEncoder(model_config_dict)
            decoder = DonutDecoder(model_config_dict)
            model = DonutOCRModel(encoder, decoder, model_config_dict)
        
        model.set_processor(processor)
        model.to_device(config.precision)
        
        if config.freeze_encoder:
            model.freeze_encoder()
        
        return model
    
    def _setup_trocr_model(self, config: ModelConfig) -> TrOCROCRModel:
        from transformers import TrOCRProcessor
        
        logger.info(f"Setting up TrOCR model with config: {config.to_dict()}")
        logger.info(f"model_name_or_path before check: '{config.model_name_or_path}' (type: {type(config.model_name_or_path)})")
        
        # Проверяем, что model_name_or_path не None и не пустая строка
        model_name_or_path = config.model_name_or_path
        if not model_name_or_path:
            logger.error(f"model_name_or_path is None or empty: '{model_name_or_path}'")
            logger.error(f"Full config dict: {config.to_dict()}")
            raise ValueError(f"model_name_or_path is not set in configuration: {model_name_or_path}")
        
        logger.info(f"Loading TrOCRProcessor from: {model_name_or_path}")
        # Добавляем use_fast=True для исправления предупреждения
        processor = TrOCRProcessor.from_pretrained(model_name_or_path, use_fast=True)
        
        model_config_dict = {
            'encoder_name': config.encoder_name,
            'decoder_name': config.decoder_name,
            'max_length': config.max_length,
            'precision': config.precision,
            'enable_gradient_checkpointing': config.enable_gradient_checkpointing,
            'freeze_encoder': config.freeze_encoder,
            'flash_attention': config.flash_attention
        }
        
        if config.model_name_or_path and Path(config.model_name_or_path).exists():
            model = TrOCROCRModel.from_pretrained(config.model_name_or_path)
        else:
            from models.trocr import TrOCREncoder, TrOCRDecoder
            encoder = TrOCREncoder(model_config_dict)
            decoder = TrOCRDecoder(model_config_dict)
            model = TrOCROCRModel(encoder, decoder, model_config_dict)
        
        model.set_processor(processor)
        model.to_device(config.precision)
        
        return model
    
    def _setup_realtime_inference(self, model_type: str = "donut") -> None:
        try:
            self.realtime_engine = RealtimeInferenceEngine(
                model=self.model,
                device=self.model.device,
                precision=self.config_manager.model.precision,
                max_length=min(64, self.config_manager.model.max_length),
                num_beams=1
            )
            
            display_interval = getattr(self.config_manager.training, 'inference_display_interval', 100)
            self.inference_displayer = TrainingInferenceDisplayer(
                self.realtime_engine, 
                display_interval=display_interval
            )
            
            logger.info(f"Real-time inference engine setup complete for {model_type}")
            
        except Exception as e:
            logger.warning(f"Failed to setup real-time inference engine: {e}")
            self.realtime_engine = None
            self.inference_displayer = None
    
    def save_checkpoint(self, epoch: int, step: int, loss: float, metrics: Optional[Dict[str, float]] = None, is_best: bool = False) -> Path:
        checkpoint_name = f"checkpoint-epoch-{epoch:03d}-step-{step:05d}"
        if is_best:
            checkpoint_name += "-best"
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        model_path = checkpoint_path / "model"
        self.model.save_pretrained(model_path)
        
        if hasattr(self.model, 'processor') and self.model.processor is not None:
            processor_path = checkpoint_path / "processor"
            processor_path.mkdir(exist_ok=True)
            self.model.processor.save_pretrained(processor_path)
        
        training_state = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics or {},
            'model_config': self.config_manager.model.to_dict(),
            'training_config': self.config_manager.training.to_dict(),
            'data_config': self.config_manager.data.to_dict(),
            'timestamp': time.time(),
            'model_type': getattr(self.config_manager.model, 'model_type', 'unknown')
        }
        
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path / "training_state.json", "r") as f:
            training_state = json.load(f)
        
        logger.info(f"Loading checkpoint from epoch {training_state['epoch']}, step {training_state['step']}")
        
        if 'model_config' in training_state:
            self.config_manager.model = self.config_manager.model.from_dict(training_state['model_config'])
        if 'training_config' in training_state:
            self.config_manager.training = self.config_manager.training.from_dict(training_state['training_config'])
        if 'data_config' in training_state:
            self.config_manager.data = self.config_manager.data.from_dict(training_state['data_config'])
        
        model_type = training_state.get('model_type', 'donut')
        self.setup_model(model_type)
        
        model_path = checkpoint_path / "model"
        if model_path.exists():
            if hasattr(self.model.__class__, 'from_pretrained'):
                self.model = self.model.__class__.from_pretrained(model_path)
            else:
                model_weights = torch.load(model_path / "pytorch_model.bin", map_location=self.model.device)
                self.model.load_state_dict(model_weights)
        
        processor_path = checkpoint_path / "processor"
        if processor_path.exists() and hasattr(self.model, 'processor'):
            if hasattr(self.model.processor.__class__, 'from_pretrained'):
                self.model.processor = self.model.processor.__class__.from_pretrained(processor_path)
        
        return training_state
    
    def prepare_datasets(self, model_type: str = "donut") -> tuple:
        data_config = self.config_manager.data
        
        dataset_class = DonutDataset if model_type.lower() == "donut" else TrOCRDataset
        
        dataset_config = {
            'apply_augmentation': data_config.apply_augmentation,
            'augmentation_prob': data_config.augmentation_prob,
            'max_rotation': data_config.max_rotation,
            'noise_level': data_config.noise_level,
            'color_jitter': data_config.color_jitter,
            'elastic_transform': data_config.elastic_transform,
            'random_perspective': data_config.random_perspective,
            'enable_caching': data_config.enable_caching,
            'cache_dir': data_config.cache_dir,
            'max_length': self.config_manager.model.max_length,
            'current_epoch': 0,
            'max_epochs': self.config_manager.training.num_epochs
        }
        
        processor = self.model.processor if hasattr(self.model, 'processor') else None
        if processor is None:
            from transformers import DonutProcessor, TrOCRProcessor
            processor_class = DonutProcessor if model_type == "donut" else TrOCRProcessor
            
            # Проверяем, что model_name_or_path не None
            model_name_or_path = self.config_manager.model.model_name_or_path
            if not model_name_or_path:
                raise ValueError(f"model_name_or_path is not set in configuration: {model_name_or_path}")
            
            # Добавляем use_fast=True
            processor = processor_class.from_pretrained(model_name_or_path, use_fast=True)
        
        train_dataset = dataset_class(
            processor=processor,
            data_dir=data_config.data_dir,
            split="train",
            config=dataset_config
        )
        
        val_dataset = dataset_class(
            processor=processor,
            data_dir=data_config.data_dir,
            split="validation",
            config=dataset_config
        )
        
        logger.info(f"Training dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset) -> tuple:
        train_config = self.config_manager.training
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_dataloader, val_dataloader
    
    def train(self, model_type: str = "donut", resume_from_state: Optional[Dict[str, Any]] = None) -> dict:
        self.setup_model(model_type)
        
        train_dataset, val_dataset = self.prepare_datasets(model_type)
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        self.trainer = TwoStageTrainer(
            model=self.model,
            train_config=self.config_manager.training,
            data_config=self.config_manager.data,
            output_dir=self.config_manager.training.output_dir
        )
        
        config_output_dir = Path(self.config_manager.training.output_dir)
        self.config_manager.save_all(config_output_dir / "configs")
        
        self.checkpoint_dir = config_output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info("Starting training")
        training_history = self.trainer.train(train_dataloader, val_dataloader)
        
        final_model_path = config_output_dir / "final_model"
        self.model.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return training_history
    
    def run_inference_demo(self, image_path: str, model_path: Optional[str] = None) -> None:
        if self.model is None:
            if model_path:
                model_config = self.config_manager.model
                if model_config.model_type == "donut":
                    self.model = DonutOCRModel.from_pretrained(model_path)
                else:
                    self.model = TrOCROCRModel.from_pretrained(model_path)
            else:
                raise ValueError("No model available for inference")
        
        from PIL import Image
        image = Image.open(image_path)
        
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'processor'):
                pixel_values = self.model.processor(image, return_tensors="pt")["pixel_values"]
            else:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                pixel_values = transform(image).unsqueeze(0)
            
            predictions = self.model.generate(pixel_values.to(self.model.device))
            prediction_text = predictions[0] if predictions else ""
        
        output_dir = Path(self.config_manager.training.output_dir)
        viz_path = output_dir / "inference_demo.png"
        
        result_image = self.visualizer.visualize_ocr_result(
            image=image,
            text_prediction=prediction_text,
            save_path=viz_path
        )
        
        logger.info(f"Inference result: '{prediction_text}'")
        logger.info(f"Visualization saved to {viz_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="FrameReader OCR Training")
    
    parser.add_argument("--config", type=Path, help="Configuration directory path")
    parser.add_argument("--model-type", choices=["donut", "trocr"], default="donut", help="Model type to train")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--two-stage", action="store_true", help="Enable two-stage training")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="Training precision")
    parser.add_argument("--demo", type=str, help="Run inference demo on image")
    parser.add_argument("--model-path", type=str, help="Path to trained model for demo")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint path")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    pipeline = FrameReaderTrainingPipeline(args.config)
    
    pipeline.config_manager.training.output_dir = args.output_dir
    pipeline.config_manager.data.data_dir = args.data_dir
    pipeline.config_manager.training.num_epochs = args.epochs
    pipeline.config_manager.training.batch_size = args.batch_size
    pipeline.config_manager.training.learning_rate = args.learning_rate
    pipeline.config_manager.training.enable_two_stage = args.two_stage
    pipeline.config_manager.model.model_type = args.model_type
    pipeline.config_manager.model.precision = args.precision
    
    try:
        if args.demo:
            pipeline.run_inference_demo(args.demo, args.model_path)
        elif args.resume_from:
            logger.info(f"Resuming training from {args.resume_from}")
            training_state = pipeline.load_checkpoint(args.resume_from)
            training_history = pipeline.train(args.model_type, resume_from_state=training_state)
            logger.info("Training resumed and completed successfully!")
            logger.info(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
            if training_history.get('eval_loss'):
                logger.info(f"Final validation loss: {training_history['eval_loss'][-1]:.4f}")
        else:
            training_history = pipeline.train(args.model_type)
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
            if training_history.get('eval_loss'):
                logger.info(f"Final validation loss: {training_history['eval_loss'][-1]:.4f}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()