"""
Configuration management for OCR training pipeline.
Centralizes all parameters and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str = "naver-clova-ix/donut-base"
    model_type: str = "donut"
    encoder_name: Optional[str] = None
    decoder_name: Optional[str] = None
    max_length: int = 512
    hidden_size: int = 768
    vocab_size: int = 50265
    image_size: tuple = (224, 224)
    patch_size: int = 16
    precision: str = "bf16"
    enable_gradient_checkpointing: bool = True
    freeze_encoder: bool = False
    flash_attention: bool = False
    task_start_token: str = "<s_ocr>"
    prompt_end_token: Optional[str] = None
    decoder_start_token_id: Optional[int] = None
    enable_torch_compile: bool = False
    hf_token: Optional[str] = None
    # Параметры для Donut
    align_long_axis: bool = False
    window_size: int = 10
    encoder_layer: List[int] = field(default_factory=lambda: [2, 2, 14, 2])
    decoder_layer: int = 4
    max_position_embeddings: Optional[int] = None
    # Параметры для TrOCR
    encoder_size: str = "base"
    # Параметры для кастомного VisionEncoderDecoder
    projection_dim: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        filtered_data = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**filtered_data)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class DataConfig:
    data_dir: str = "./dataset"
    real_data_path: Optional[str] = None
    synthetic_data_path: Optional[str] = None
    cache_dir: str = "./cache"
    enable_caching: bool = True
    apply_augmentation: bool = True
    augmentation_prob: float = 0.3
    max_rotation: float = 5.0
    noise_level: float = 0.02
    color_jitter: float = 0.1
    elastic_transform: bool = True
    random_perspective: bool = True
    max_samples_per_split: Optional[int] = None
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        filtered_data = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**filtered_data)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    run_name: str = "exp1"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    save_interval: int = 1
    eval_interval: int = 1
    log_interval: int = 10
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: float = 1e-4
    seed: int = 42
    enable_distributed: bool = False
    report_to: str = "none"
    enable_two_stage: bool = False
    stage_transition_epochs: int = 5
    synthetic_data_ratio: float = 0.7
    synthetic_lr_factor: float = 1.0
    real_data_lr_factor: float = 0.5
    inference_display_interval: int = 100
    enable_attention_visualization: bool = False
    attention_visualization_interval: int = 500
    attention_save_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        filtered_data = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**filtered_data)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class VisualizationConfig:
    enable_training_plots: bool = True
    enable_realtime_inference: bool = True
    enable_attention_viz: bool = False
    plot_interval: int = 10
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        filtered_data = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**filtered_data)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class ConfigManager:
    def __init__(self, model: Optional[ModelConfig] = None, data: Optional[DataConfig] = None, 
                 training: Optional[TrainingConfig] = None, visualization: Optional[VisualizationConfig] = None):
        self.model = model or ModelConfig()
        self.data = data or DataConfig()
        self.training = training or TrainingConfig()
        self.visualization = visualization or VisualizationConfig()
    
    def save_all(self, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = {
            'model_config.json': self.model.to_dict(),
            'data_config.json': self.data.to_dict(),
            'training_config.json': self.training.to_dict(),
            'visualization_config.json': self.visualization.to_dict()
        }
        
        for filename, config_data in configs.items():
            with open(output_dir / filename, 'w') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configurations saved to {output_dir}")
    
    @classmethod
    def load(cls, config_dir: Union[str, Path]) -> "ConfigManager":
        return cls.load_all(config_dir)
    
    @classmethod
    def load_all(cls, config_dir: Union[str, Path]) -> "ConfigManager":
        config_dir = Path(config_dir)
        
        config_files = {
            'model': config_dir / 'model_config.json',
            'data': config_dir / 'data_config.json', 
            'training': config_dir / 'training_config.json',
            'visualization': config_dir / 'visualization_config.json'
        }
        
        configs = {}
        for config_type, config_file in config_files.items():
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    
                if config_type == 'model':
                    configs['model'] = ModelConfig.from_dict(data)
                elif config_type == 'data':
                    configs['data'] = DataConfig.from_dict(data)
                elif config_type == 'training':
                    configs['training'] = TrainingConfig.from_dict(data)
                elif config_type == 'visualization':
                    configs['visualization'] = VisualizationConfig.from_dict(data)
        
        return cls(**configs)