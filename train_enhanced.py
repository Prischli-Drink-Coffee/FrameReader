#!/usr/bin/env python3
"""
FrameReader OCR System - Enhanced Training Script
Demonstrates the new OOP architecture with two-stage training.
"""

import argparse
import json
import logging
import sys
import time
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
from visualization.realtime_inference import RealtimeInferenceEngine, TrainingInferenceDisplayer
from visualization.attention import AttentionVisualizer

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
        self.realtime_engine = None
        self.inference_displayer = None
    
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
        
        # Setup real-time inference engine for training monitoring
        self._setup_realtime_inference(model_type)
    
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
    
    def _setup_realtime_inference(self, model_type: str = "donut") -> None:
        """Setup real-time inference engine for training monitoring."""
        try:
            # Create realtime inference engine
            self.realtime_engine = RealtimeInferenceEngine(
                model=self.model,
                device=self.model.device,
                precision=self.config_manager.model.precision,
                max_length=min(64, self.config_manager.model.max_length),  # Shorter for speed
                num_beams=1  # Fast greedy decoding for real-time display
            )
            
            # Create inference displayer
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
    
    def _setup_enhanced_checkpointing(self, output_dir: Path) -> None:
        """Setup enhanced checkpointing system."""
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Configure trainer for enhanced checkpointing
        if hasattr(self.trainer, 'enable_enhanced_checkpointing'):
            self.trainer.enable_enhanced_checkpointing(
                self.checkpoint_dir,
                save_tokenizer=True,
                save_processor=True,
                save_training_state=True
            )
        
        logger.info(f"Enhanced checkpointing setup at {self.checkpoint_dir}")
    
    def save_enhanced_checkpoint(
        self, 
        epoch: int, 
        step: int, 
        loss: float, 
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """Save enhanced checkpoint with all necessary components."""
        
        checkpoint_name = f"checkpoint-epoch-{epoch:03d}-step-{step:05d}"
        if is_best:
            checkpoint_name += "-best"
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        try:
            # Save model weights
            model_path = checkpoint_path / "model"
            self.model.save_pretrained(model_path)
            
            # Save processor/tokenizer
            if hasattr(self.model, 'processor') and self.model.processor is not None:
                processor_path = checkpoint_path / "processor"
                processor_path.mkdir(exist_ok=True)
                self.model.processor.save_pretrained(processor_path)
                
                # Save tokenizer separately for clarity
                tokenizer_path = checkpoint_path / "tokenizer" 
                tokenizer_path.mkdir(exist_ok=True)
                self.model.processor.tokenizer.save_pretrained(tokenizer_path)
            
            # Save training state
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
            
            # Save recent inference metrics if available
            if self.inference_displayer:
                recent_metrics = self.inference_displayer.get_recent_metrics(last_n=50)
                training_state['inference_metrics'] = recent_metrics
            
            with open(checkpoint_path / "training_state.json", "w") as f:
                json.dump(training_state, f, indent=2, ensure_ascii=False)
            
            # Save optimizer state if trainer has one
            if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                torch.save(self.trainer.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
            
            # Save scheduler state if trainer has one
            if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None:
                torch.save(self.trainer.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
            
            logger.info(f"Enhanced checkpoint saved to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save enhanced checkpoint: {e}")
            raise
    
    def load_enhanced_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load enhanced checkpoint and resume training state."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load training state
        with open(checkpoint_path / "training_state.json", "r") as f:
            training_state = json.load(f)
        
        logger.info(f"Loading checkpoint from epoch {training_state['epoch']}, step {training_state['step']}")
        
        # Update configs from checkpoint
        if 'model_config' in training_state:
            self.config_manager.model = self.config_manager.model.from_dict(training_state['model_config'])
        if 'training_config' in training_state:
            self.config_manager.training = self.config_manager.training.from_dict(training_state['training_config'])
        if 'data_config' in training_state:
            self.config_manager.data = self.config_manager.data.from_dict(training_state['data_config'])
        
        # Load model
        model_type = training_state.get('model_type', 'donut')
        self.setup_model(model_type)
        
        # Load model weights
        model_path = checkpoint_path / "model"
        if model_path.exists():
            if hasattr(self.model.__class__, 'from_pretrained'):
                self.model = self.model.__class__.from_pretrained(model_path)
            else:
                logger.warning("Model does not support from_pretrained, loading state dict")
                # Load state dict as fallback
                model_weights = torch.load(model_path / "pytorch_model.bin", map_location=self.model.device)
                self.model.load_state_dict(model_weights)
        
        # Load processor/tokenizer
        processor_path = checkpoint_path / "processor"
        if processor_path.exists() and hasattr(self.model, 'processor'):
            if hasattr(self.model.processor.__class__, 'from_pretrained'):
                self.model.processor = self.model.processor.__class__.from_pretrained(processor_path)
        
        return training_state
    
    def _setup_attention_visualization(self, output_dir: Path) -> None:
        """Setup attention visualization system."""
        attention_dir = output_dir / "attention_visualizations"
        attention_dir.mkdir(exist_ok=True)
        
        self.attention_visualizer = AttentionVisualizer(output_dir=attention_dir)
        
        # Update training config with attention save directory
        self.config_manager.training.attention_save_dir = str(attention_dir)
        
        logger.info(f"Attention visualization setup at {attention_dir}")
    
    def visualize_model_attention(
        self,
        batch_data: Dict[str, torch.Tensor],
        step: int,
        sample_idx: int = 0
    ) -> None:
        """Generate and save attention visualizations."""
        if not hasattr(self, 'attention_visualizer'):
            return
            
        interval = self.config_manager.training.attention_visualization_interval
        if step % interval != 0:
            return
            
        try:
            # Extract sample from batch
            pixel_values = batch_data.get('pixel_values', None)
            if pixel_values is None or pixel_values.dim() != 4:
                return
                
            sample_image_tensor = pixel_values[sample_idx] if sample_idx < pixel_values.shape[0] else pixel_values[0]
            
            # Convert tensor to PIL Image for visualization
            # Denormalize if needed
            import torchvision.transforms.functional as TF
            sample_image = TF.to_pil_image(sample_image_tensor.cpu())
            
            # Get attention weights from model if available
            self.model.eval()
            with torch.no_grad():
                # This would need to be customized based on your model's attention output
                # For now, we'll create a placeholder visualization
                if hasattr(self.model, 'get_attention_weights'):
                    attention_weights = self.model.get_attention_weights(sample_image_tensor.unsqueeze(0))
                    
                    # Visualize encoder attention
                    if 'encoder_attention' in attention_weights:
                        save_path = Path(self.config_manager.training.attention_save_dir) / f"encoder_attention_step_{step:05d}.png"
                        self.attention_visualizer.visualize_encoder_attention(
                            sample_image,
                            attention_weights['encoder_attention'],
                            save_path=save_path
                        )
                    
                    # Visualize decoder attention
                    if 'decoder_attention' in attention_weights:
                        # Get tokens for visualization
                        tokens = []
                        if 'input_ids' in batch_data:
                            sample_ids = batch_data['input_ids'][sample_idx] if sample_idx < batch_data['input_ids'].shape[0] else batch_data['input_ids'][0]
                            if hasattr(self.model, 'processor') and self.model.processor is not None:
                                tokens = self.model.processor.tokenizer.convert_ids_to_tokens(sample_ids.cpu().tolist())
                        
                        save_path = Path(self.config_manager.training.attention_save_dir) / f"decoder_attention_step_{step:05d}.png"
                        self.attention_visualizer.visualize_decoder_attention(
                            sample_image,
                            tokens,
                            attention_weights['decoder_attention'],
                            save_path=save_path
                        )
                        
                logger.info(f"Attention visualization saved for step {step}")
                
        except Exception as e:
            logger.warning(f"Failed to generate attention visualization: {e}")
    
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
    
    def train(self, model_type: str = "donut", resume_from_state: Optional[Dict[str, Any]] = None) -> dict:
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
        
        # Setup enhanced checkpointing
        self._setup_enhanced_checkpointing(config_output_dir)
        
        # Setup attention visualization if enabled
        if self.config_manager.training.enable_attention_visualization:
            self._setup_attention_visualization(config_output_dir)
        
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
    
    # Enhanced features
    parser.add_argument("--enable-realtime-inference", action="store_true", 
                        help="Enable real-time inference display during training")
    parser.add_argument("--inference-interval", type=int, default=100,
                        help="Interval for real-time inference display")
    parser.add_argument("--enable-attention-viz", action="store_true",
                        help="Enable attention visualization during training")
    parser.add_argument("--attention-interval", type=int, default=500,
                        help="Interval for attention visualization")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint path")
    
    # Precision control
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16",
                        help="Training precision for performance optimization")
    
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
    pipeline.config_manager.model.precision = args.precision
    
    # Enhanced features
    if args.enable_realtime_inference:
        pipeline.config_manager.training.inference_display_interval = args.inference_interval
    if args.enable_attention_viz:
        pipeline.config_manager.training.enable_attention_visualization = True
        pipeline.config_manager.training.attention_visualization_interval = args.attention_interval
    
    try:
        if args.demo:
            # Run inference demo
            pipeline.run_inference_demo(args.demo, args.model_path)
        elif args.resume_from:
            # Resume training from checkpoint
            logger.info(f"Resuming training from {args.resume_from}")
            training_state = pipeline.load_enhanced_checkpoint(args.resume_from)
            
            # Continue training
            training_history = pipeline.train(args.model_type, resume_from_state=training_state)
            
            logger.info("Training resumed and completed successfully!")
            logger.info(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
            
            if training_history.get('eval_loss'):
                logger.info(f"Final validation loss: {training_history['eval_loss'][-1]:.4f}")
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