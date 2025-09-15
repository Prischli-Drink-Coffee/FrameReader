#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import signal
from pathlib import Path
from typing import Optional

# Отключаем предупреждения о параллелизме tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from core.config import ConfigManager
from models.donut import DonutOCRModel
from models.trocr import TrOCROCRModel  
from data.dataset import create_dataset, collate_fn
from training.trainer import StandardTrainer, TwoStageTrainer
from visualization.inference import InferenceVisualizer
from visualization.realtime_inference import RealtimeInferenceEngine, TrainingInferenceDisplayer

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
        if (config_path and config_path.exists()):
            logger.info(f"Loading configuration from: {config_path}")
            return ConfigManager.load_all(config_path)
        
        logger.info("Creating default configuration")
        return ConfigManager()
    
    def setup_model(self) -> None:
        model_config = self.config_manager.model
        training_config = self.config_manager.training
        data_config = self.config_manager.data
        
        model_type = model_config.model_type.lower()
        
        if model_type == "donut":
            self.model = self._setup_donut_model(
                model_config.to_dict(),
                training_config.to_dict(),
                data_config.to_dict()
            )
        elif model_type == "trocr":
            self.model = self._setup_trocr_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model {model_type} setup complete")
        self._setup_realtime_inference()
    
    def _setup_donut_model(self, model_config_dict, training_config_dict, data_config_dict) -> DonutOCRModel:
        from models.donut import DonutOCRModel, SwinEncoder, BARTDecoder
        from transformers import DonutProcessor

        img_size = model_config_dict.get("image_size", [384, 384])
        window_size = model_config_dict.get("window_size", 7)
        encoder_layer = model_config_dict.get("encoder_layer", [2, 2, 14, 2])
        decoder_layer = model_config_dict.get("decoder_layer", 4)
        max_position_embeddings = model_config_dict.get("max_position_embeddings", 512)
        align_long_axis = model_config_dict.get("align_long_axis", False)
        use_pretrained = model_config_dict.get("use_pretrained", False)
        
        encoder = SwinEncoder(
            input_size=img_size, 
            align_long_axis=align_long_axis,
            window_size=window_size, 
            encoder_layer=encoder_layer
        )
        
        decoder = BARTDecoder(
            decoder_layer=decoder_layer, 
            max_position_embeddings=max_position_embeddings
        )
        
        model = DonutOCRModel(encoder, decoder, model_config_dict)
        
        if use_pretrained and "pretrained_model" in model_config_dict:
            pretrained_model = model_config_dict["pretrained_model"]
            try:
                processor = DonutProcessor.from_pretrained(pretrained_model)
                model.set_processor(processor)
                logger.info(f"Loaded processor from {pretrained_model}")
            except Exception as e:
                logger.warning(f"Failed to load processor from {pretrained_model}: {e}")
                processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
                model.set_processor(processor)
        else:
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            model.set_processor(processor)
        
        precision = training_config_dict.get('precision', 'fp32')
        model.to_device(precision)
        
        if model_config_dict.get("freeze_encoder", False):
            model.freeze_encoder()
            logger.info("Encoder frozen")
        
        return model
    
    def _setup_trocr_model(self, config) -> TrOCROCRModel:
        from transformers import TrOCRProcessor
        from models.trocr import TrOCREncoder, TrOCRDecoder
        
        model_name = config.model_name_or_path or "microsoft/trocr-base-printed"
        processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
        
        model_config_dict = config.to_dict()
        encoder = TrOCREncoder(model_config_dict)
        decoder = TrOCRDecoder(model_config_dict)
        model = TrOCROCRModel(encoder, decoder, model_config_dict)
        
        model.set_processor(processor)
        model.to_device(config.precision)
        
        if config.freeze_encoder:
            model.freeze_encoder()
        
        return model
    
    def _setup_realtime_inference(self) -> None:
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
            
            logger.info("Real-time inference engine setup complete")
            
        except Exception as e:
            logger.warning(f"Failed to setup real-time inference engine: {e}")
            self.realtime_engine = None
            self.inference_displayer = None
    
    def prepare_datasets(self) -> tuple:
        data_config = self.config_manager.data
        model_config = self.config_manager.model
        
        dataset_config = data_config.to_dict()
        dataset_config.update({
            'max_length': model_config.max_length,
            'current_epoch': 0,
            'max_epochs': self.config_manager.training.num_epochs
        })
        
        processor = self.model.processor if hasattr(self.model, 'processor') else None
        if processor is None:
            raise ValueError("Model processor not available")
        
        train_dataset = create_dataset(
            model_type=model_config.model_type,
            processor=processor,
            data_dir=data_config.data_dir,
            split="train",
            config=dataset_config
        )
        
        val_dataset = create_dataset(
            model_type=model_config.model_type,
            processor=processor,
            data_dir=data_config.data_dir,
            split="valid",
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
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        
        return train_dataloader, val_dataloader
    
    def train(self) -> dict:
        self.setup_model()
        
        train_dataset, val_dataset = self.prepare_datasets()
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        output_dir = Path(self.config_manager.training.output_dir)
        
        if self.config_manager.training.enable_two_stage:
            synthetic_dataloader = None
            real_dataloader = None
            
            data_config = self.config_manager.data
            model_config = self.config_manager.model
            dataset_config = data_config.to_dict()
            dataset_config.update({
                'max_length': model_config.max_length,
                'current_epoch': 0,
                'max_epochs': self.config_manager.training.num_epochs
            })
            
            synth_path = Path(data_config.synthetic_data_path) if data_config.synthetic_data_path else Path(data_config.data_dir) / "synth"
            if synth_path.exists():
                synthetic_dataset = create_dataset(model_config.model_type, self.model.processor, synth_path, "train", dataset_config)
                synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=self.config_manager.training.batch_size, 
                                                shuffle=True, collate_fn=collate_fn)
            
            real_path = Path(data_config.real_data_path) if data_config.real_data_path else Path(data_config.data_dir) / "real"
            if real_path.exists():
                real_dataset = create_dataset(model_config.model_type, self.model.processor, real_path, "train", dataset_config)
                real_dataloader = DataLoader(real_dataset, batch_size=self.config_manager.training.batch_size, 
                                           shuffle=True, collate_fn=collate_fn)
            
            self.trainer = TwoStageTrainer(
                model=self.model,
                train_config=self.config_manager.training,
                data_config=self.config_manager.data,
                output_dir=output_dir,
                synthetic_dataloader=synthetic_dataloader,
                real_dataloader=real_dataloader,
                inference_displayer=self.inference_displayer
            )
        else:
            self.trainer = StandardTrainer(
                model=self.model,
                train_config=self.config_manager.training,
                data_config=self.config_manager.data,
                output_dir=output_dir,
                inference_displayer=self.inference_displayer
            )
        
        self.config_manager.save_all(output_dir / "configs")
        
        logger.info("🎯 Starting training with real-time inference monitoring")
        if self.inference_displayer:
            logger.info("✅ Real-time inference engine is active")
        else:
            logger.warning("⚠️  Real-time inference engine is not available")
            
        training_history = self.trainer.train(train_dataloader, val_dataloader)
        
        final_model_path = output_dir / "final_model"
        self.model.save_pretrained(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return training_history
    
    def save_emergency_checkpoint(self, output_dir: Optional[Path] = None) -> None:
        """
        Сохраняет экстренную контрольную точку модели при прерывании обучения
        """
        if not output_dir:
            output_dir = Path(self.config_manager.training.output_dir)
        
        emergency_dir = output_dir / "emergency_checkpoint"
        
        try:
            if self.model:
                logger.info(f"Сохранение модели в экстренном режиме: {emergency_dir}")
                emergency_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(emergency_dir)
                
                # Сохранение текущего состояния обучения, если доступен тренер
                if self.trainer:
                    self.trainer._save_checkpoint('emergency_checkpoint')
                    
                    # Сохранение визуализации прогресса обучения
                    if hasattr(self.trainer, 'visualizer'):
                        try:
                            if hasattr(self.trainer, 'history'):
                                history = self.trainer.history
                            else:
                                # Попытка воссоздать историю из собранных метрик
                                history = {'train_loss': [], 'eval_loss': [], 'learning_rates': []}
                                if hasattr(self.trainer, 'metrics_collector'):
                                    metrics = self.trainer.metrics_collector.get_all_metrics()
                                    if 'losses' in metrics:
                                        history['train_loss'] = metrics['losses']
                                
                            if hasattr(self.trainer.visualizer, 'finalize_training'):
                                self.trainer.visualizer.finalize_training(history)
                        except Exception as e:
                            logger.warning(f"Не удалось сохранить визуализацию: {e}")
                
                logger.info(f"Экстренная контрольная точка сохранена в {emergency_dir}")
        except Exception as e:
            logger.error(f"Ошибка при экстренном сохранении модели: {e}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Загружает модель и состояние обучения из указанной контрольной точки
        
        Args:
            checkpoint_path: Путь к директории с контрольной точкой
        """
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path не существует: {checkpoint_path}")
        
        # Сначала настраиваем модель
        self.setup_model()
        
        # Загружаем состояние модели
        if hasattr(self.model, 'from_pretrained'):
            logger.info(f"Загрузка модели из {checkpoint_path}")
            # Для моделей с методом from_pretrained
            model_type = self.config_manager.model.model_type
            if model_type == "donut":
                self.model = DonutOCRModel.from_pretrained(checkpoint_path)
            else:
                self.model = TrOCROCRModel.from_pretrained(checkpoint_path)
        else:
            # Для других типов моделей
            model_path = checkpoint_path / "model.pt"
            if not model_path.exists():
                raise ValueError(f"Файл модели не найден: {model_path}")
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
            
        logger.info(f"Модель загружена из {checkpoint_path}")
        
        # Настраиваем тренера и инициализируем датасеты и датализеры
        output_dir = Path(self.config_manager.training.output_dir)
        train_dataset, val_dataset = self.prepare_datasets()
        train_dataloader, val_dataloader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Создаем соответствующий тренер
        if self.config_manager.training.enable_two_stage:
            # Для двухэтапного обучения настраиваем дополнительные датализеры
            synthetic_dataloader = None
            real_dataloader = None
            
            data_config = self.config_manager.data
            model_config = self.config_manager.model
            dataset_config = data_config.to_dict()
            dataset_config.update({
                'max_length': model_config.max_length,
                'current_epoch': 0,
                'max_epochs': self.config_manager.training.num_epochs
            })
            
            synth_path = Path(data_config.synthetic_data_path) if data_config.synthetic_data_path else Path(data_config.data_dir) / "synth"
            if synth_path.exists():
                synthetic_dataset = create_dataset(model_config.model_type, self.model.processor, synth_path, "train", dataset_config)
                synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=self.config_manager.training.batch_size, 
                                                shuffle=True, collate_fn=collate_fn)
            
            real_path = Path(data_config.real_data_path) if data_config.real_data_path else Path(data_config.data_dir) / "real"
            if real_path.exists():
                real_dataset = create_dataset(model_config.model_type, self.model.processor, real_path, "train", dataset_config)
                real_dataloader = DataLoader(real_dataset, batch_size=self.config_manager.training.batch_size, 
                                           shuffle=True, collate_fn=collate_fn)
            
            self.trainer = TwoStageTrainer(
                model=self.model,
                train_config=self.config_manager.training,
                data_config=self.config_manager.data,
                output_dir=output_dir,
                synthetic_dataloader=synthetic_dataloader,
                real_dataloader=real_dataloader,
                inference_displayer=self.inference_displayer
            )
        else:
            self.trainer = StandardTrainer(
                model=self.model,
                train_config=self.config_manager.training,
                data_config=self.config_manager.data,
                output_dir=output_dir,
                inference_displayer=self.inference_displayer
            )
        
        # Загружаем состояние обучения
        self.trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Состояние обучения загружено из {checkpoint_path}")
        
        # Сбрасываем метрику лучшей модели, чтобы сохранение продолжалось корректно
        self.trainer.best_metric = float('inf')
        
        # Обновляем real-time inference engine с новой моделью
        self._setup_realtime_inference()
    
    def run_inference_demo(self, image_path: str, model_path: Optional[str] = None) -> None:
        if self.model is None:
            if model_path:
                model_type = self.config_manager.model.model_type
                if model_type == "donut":
                    self.model = DonutOCRModel.from_pretrained(model_path)
                else:
                    self.model = TrOCROCRModel.from_pretrained(model_path)
            else:
                raise ValueError("No model available for inference")
        
        from PIL import Image
        image = Image.open(image_path)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.generate(
                self.model.processor(image, return_tensors="pt")["pixel_values"].to(self.model.device)
            )
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
    parser.add_argument("--config", type=Path, required=True, help="Configuration directory path")
    parser.add_argument("--resume", type=Path, help="Resume training from checkpoint path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    pipeline = FrameReaderTrainingPipeline(args.config)
    
    # Глобальная переменная для отслеживания обучения
    global training_interrupted
    training_interrupted = False
    
    # Обработчик сигнала SIGINT (Ctrl+C)
    def signal_handler(sig, frame):
        global training_interrupted
        if training_interrupted:  # Если уже обрабатываем прерывание
            logger.warning("Принудительное завершение без сохранения...")
            sys.exit(1)
            
        training_interrupted = True
        logger.warning("\n\nОбучение прервано пользователем! Сохранение текущего состояния...")
        
        # Сохраняем экстренную контрольную точку
        if hasattr(pipeline, 'save_emergency_checkpoint'):
            pipeline.save_emergency_checkpoint()
        
        logger.info("Экстренное сохранение завершено. Завершение работы...")
        sys.exit(0)
    
    # Регистрируем обработчик сигнала
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Загружаем чекпоинт, если указан параметр --resume
        if args.resume:
            logger.info(f"Продолжаем обучение с чекпоинта: {args.resume}")
            pipeline.load_checkpoint(args.resume)
        
        training_history = pipeline.train()
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
        if training_history.get('eval_loss'):
            logger.info(f"Final validation loss: {training_history['eval_loss'][-1]:.4f}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()