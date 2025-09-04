#!/usr/bin/env python3
"""
FrameReader OCR System - Enhanced Training Script
Demonstrates the new OOP architecture with two-stage training.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Import the new modular components
from core.config import ConfigManager, ModelConfig, TrainingConfig, DataConfig, VisualizationConfig
from models.donut import DonutOCRModel
from models.trocr import TrOCROCRModel
from data.dataset import DonutDataset, TrOCRDataset
from training.trainer import TwoStageTrainer
from visualization.inference import InferenceVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FrameReaderTrainingPipeline:
    """Main training pipeline with enhanced architecture."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_manager = self._load_or_create_config(config_path)
        self.model = None
        self.trainer = None
        self.visualizer = InferenceVisualizer()
    
    def _load_or_create_config(self, config_path: Optional[Path]) -> ConfigManager:
        """Load configuration or create default."""
        if config_path and config_path.exists():
            return ConfigManager.load_all(config_path)
        else:
            logger.info("Creating default configuration")
            return ConfigManager()
    
    def setup_model(self, model_type: str = "donut") -> None:
        """Setup OCR model based on configuration."""
        model_config = self.config_manager.model
        
        if model_type.lower() == "donut":
            self.model = self._setup_donut_model(model_config)
        elif model_type.lower() == "trocr":
            self.model = self._setup_trocr_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model {model_type} setup complete")
    
    def _setup_donut_model(self, config: ModelConfig) -> DonutOCRModel:
        """Setup Donut model with configuration."""
        from transformers import DonutProcessor
        
        # Load processor
        processor = DonutProcessor.from_pretrained(config.model_name_or_path)
        
        # Create model configuration
        model_config_dict = {
            'encoder_name': config.encoder_name,
            'decoder_name': config.decoder_name,
            'max_length': config.max_length,
            'task_start_token': config.task_start_token,
            'enable_gradient_checkpointing': config.enable_gradient_checkpointing
        }
        
        # Load or create model
        if config.model_name_or_path and Path(config.model_name_or_path).exists():
            model = DonutOCRModel.from_pretrained(config.model_name_or_path)
        else:
            # Create new model from config
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
        """Setup TrOCR model with configuration."""
        from transformers import TrOCRProcessor
        
        # Load processor
        processor = TrOCRProcessor.from_pretrained(config.model_name_or_path)
        
        # Create model configuration
        model_config_dict = {
            'encoder_name': config.encoder_name,
            'decoder_name': config.decoder_name,
            'max_length': config.max_length,
            'precision': config.precision,
            'enable_gradient_checkpointing': config.enable_gradient_checkpointing,
            'freeze_encoder': config.freeze_encoder,
            'flash_attention': config.flash_attention
        }
        
        # Load or create model
        if config.model_name_or_path and Path(config.model_name_or_path).exists():
            model = TrOCROCRModel.from_pretrained(config.model_name_or_path)
        else:
            # Create new model from config
            from models.trocr import TrOCREncoder, TrOCRDecoder
            encoder = TrOCREncoder(model_config_dict)
            decoder = TrOCRDecoder(model_config_dict)
            model = TrOCROCRModel(encoder, decoder, model_config_dict)
        
        model.set_processor(processor)
        model.to_device(config.precision)
        
        return model
    
    def prepare_datasets(self, model_type: str = "donut") -> tuple:
        """Prepare training and validation datasets."""
        data_config = self.config_manager.data
        
        if model_type.lower() == "donut":
            dataset_class = DonutDataset
        else:
            dataset_class = TrOCRDataset
        
        # Prepare dataset configuration
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
        
        # Create datasets
        if hasattr(self.model, 'processor'):
            processor = self.model.processor
        else:
            # Fallback processor loading
            from transformers import DonutProcessor, TrOCRProcessor
            processor_class = DonutProcessor if model_type == "donut" else TrOCRProcessor
            processor = processor_class.from_pretrained(self.config_manager.model.model_name_or_path)
        
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
        
        # Log dataset statistics
        train_stats = train_dataset.get_data_statistics()
        val_stats = val_dataset.get_data_statistics()
        logger.info(f"Training data types: {train_stats['data_types']}")
        logger.info(f"Validation data types: {val_stats['data_types']}")
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset) -> tuple:
        """Create training and validation dataloaders."""
        train_config = self.config_manager.training
        
        # Training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None  # Datasets handle their own collation
        )
        
        # Validation dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None
        )
        
        return train_dataloader, val_dataloader
    
    def setup_two_stage_training(self, train_dataset) -> Optional[tuple]:
        """Setup separate dataloaders for two-stage training."""
        if not self.config_manager.training.enable_two_stage:
            return None
        
        # Split dataset by data type
        synthetic_samples = []
        real_samples = []
        
        for sample in train_dataset.samples:
            if sample.get('data_type') == 'synthetic':
                synthetic_samples.append(sample)
            else:
                real_samples.append(sample)
        
        if not synthetic_samples or not real_samples:
            logger.warning("Cannot setup two-stage training: missing synthetic or real data")
            return None
        
        # Create separate datasets
        synthetic_dataset = train_dataset.__class__(
            processor=train_dataset.processor,
            data_dir=train_dataset.data_dir,
            split="train",
            config=train_dataset.config
        )
        synthetic_dataset.samples = synthetic_samples
        
        real_dataset = train_dataset.__class__(
            processor=train_dataset.processor,
            data_dir=train_dataset.data_dir,
            split="train", 
            config=train_dataset.config
        )
        real_dataset.samples = real_samples
        
        # Create dataloaders
        synthetic_dataloader = DataLoader(
            synthetic_dataset,
            batch_size=self.config_manager.training.batch_size,
            shuffle=True,
            num_workers=self.config_manager.training.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        real_dataloader = DataLoader(
            real_dataset,
            batch_size=self.config_manager.training.batch_size,
            shuffle=True,
            num_workers=self.config_manager.training.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Two-stage training setup: {len(synthetic_samples)} synthetic, {len(real_samples)} real")
        return synthetic_dataloader, real_dataloader
    
    def train(self, model_type: str = "donut") -> dict:
        """Execute the complete training pipeline."""
        
        # Setup model
        self.setup_model(model_type)
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(model_type)
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Setup two-stage training if enabled
        two_stage_dataloaders = self.setup_two_stage_training(train_dataset)
        
        # Create trainer
        self.trainer = TwoStageTrainer(
            model=self.model,
            train_config=self.config_manager.training,
            data_config=self.config_manager.data,
            output_dir=self.config_manager.training.output_dir,
            synthetic_dataloader=two_stage_dataloaders[0] if two_stage_dataloaders else None,
            real_dataloader=two_stage_dataloaders[1] if two_stage_dataloaders else None
        )
        
        # Save configuration
        config_output_dir = Path(self.config_manager.training.output_dir)
        self.config_manager.save_all(config_output_dir / "configs")
        
        # Start training
        logger.info("Starting training with enhanced architecture")
        training_history = self.trainer.train(train_dataloader, val_dataloader)
        
        # Final model save
        final_model_path = config_output_dir / "final_model"
        self.model.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return training_history
    
    def run_inference_demo(self, image_path: str, model_path: Optional[str] = None) -> None:
        """Run inference demo with visualization."""
        if self.model is None:
            if model_path:
                # Load model for inference
                model_config = self.config_manager.model
                if model_config.model_type == "donut":
                    self.model = DonutOCRModel.from_pretrained(model_path)
                else:
                    self.model = TrOCROCRModel.from_pretrained(model_path)
            else:
                raise ValueError("No model available for inference")
        
        # Load and process image
        from PIL import Image
        image = Image.open(image_path)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'processor'):
                pixel_values = self.model.processor(image, return_tensors="pt")["pixel_values"]
            else:
                # Fallback processing
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                pixel_values = transform(image).unsqueeze(0)
            
            # Generate prediction
            predictions = self.model.generate(pixel_values.to(self.model.device))
            prediction_text = predictions[0] if predictions else ""
        
        # Visualize result
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FrameReader Enhanced OCR Training")
    
    parser.add_argument("--config", type=Path, help="Configuration directory path")
    parser.add_argument("--model-type", choices=["donut", "trocr"], default="donut", help="Model type to train")
    parser.add_argument("--output-dir", type=str, default="./output_enhanced", help="Output directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--two-stage", action="store_true", help="Enable two-stage training")
    
    # Inference demo
    parser.add_argument("--demo", type=str, help="Run inference demo on image")
    parser.add_argument("--model-path", type=str, help="Path to trained model for demo")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create pipeline
    pipeline = FrameReaderTrainingPipeline(args.config)
    
    # Update configuration with command line arguments
    pipeline.config_manager.training.output_dir = args.output_dir
    pipeline.config_manager.data.data_dir = args.data_dir
    pipeline.config_manager.training.num_epochs = args.epochs
    pipeline.config_manager.training.batch_size = args.batch_size
    pipeline.config_manager.training.learning_rate = args.learning_rate
    pipeline.config_manager.training.enable_two_stage = args.two_stage
    pipeline.config_manager.model.model_type = args.model_type
    
    try:
        if args.demo:
            # Run inference demo
            pipeline.run_inference_demo(args.demo, args.model_path)
        else:
            # Run training
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