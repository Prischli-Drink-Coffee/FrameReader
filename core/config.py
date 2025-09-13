"""
Configuration management for OCR training pipeline.
Centralizes all parameters and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json


class ConfigMixin:
    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**{key: value for key, value in config_dict.items() if key in cls.__dataclass_fields__})
    
    def save(self, config_path: Union[str, Path]) -> None:
        Path(config_path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    
    @classmethod
    def load(cls, config_path: Union[str, Path]):
        return cls.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))


@dataclass
class ModelConfig(ConfigMixin):
    model_type: str = "donut"
    model_name_or_path: str = "Akajackson/donut_rus"
    encoder_name: Optional[str] = None
    decoder_name: Optional[str] = None
    image_size: Tuple[int, int] = (384, 384)
    max_length: int = 768
    hidden_size: int = 768
    vocab_size: int = 50265
    task_start_token: str = "<s>"
    prompt_end_token: Optional[str] = None
    decoder_start_token_id: Optional[int] = None
    precision: str = "bf16"
    enable_gradient_checkpointing: bool = True
    freeze_encoder: bool = True
    flash_attention: bool = True
    enable_torch_compile: bool = False
    hf_token: Optional[str] = None


@dataclass
class TrainingConfig(ConfigMixin):
    output_dir: str = "./output"
    run_name: Optional[str] = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 1
    eval_interval: int = 1
    seed: int = 42
    dataloader_num_workers: int = 4
    enable_distributed: bool = False
    mixed_precision: bool = True
    report_to: str = "none"
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.001
    enable_two_stage: bool = False
    synthetic_data_ratio: float = 0.7
    stage_transition_epochs: int = 5
    synthetic_lr_factor: float = 1.0
    real_data_lr_factor: float = 0.5
    inference_display_interval: int = 100
    enable_attention_visualization: bool = False
    attention_visualization_interval: int = 500
    attention_save_dir: Optional[str] = None


@dataclass
class DataConfig(ConfigMixin):
    data_dir: str = "./dataset"
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    apply_augmentation: bool = True
    augmentation_prob: float = 0.3
    max_rotation: float = 8.0
    noise_level: float = 0.05
    color_jitter: bool = True
    elastic_transform: bool = True
    random_perspective: bool = True
    balance_datasets: bool = True
    synthetic_data_path: Optional[str] = None
    real_data_path: Optional[str] = None
    max_samples_per_class: Optional[int] = None
    enable_caching: bool = True
    cache_dir: str = "./data_cache"


@dataclass
class VisualizationConfig(ConfigMixin):
    enable_tensorboard: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "cer", "wer"])
    plot_interval: int = 100
    attention_viz_samples: int = 5
    show_bounding_boxes: bool = True
    overlay_text: bool = True
    confidence_threshold: float = 0.5
    enable_model_comparison: bool = False
    comparison_models: List[str] = field(default_factory=list)


class ConfigManager:
    def __init__(self, model_config: Optional[ModelConfig] = None, training_config: Optional[TrainingConfig] = None,
                 data_config: Optional[DataConfig] = None, viz_config: Optional[VisualizationConfig] = None):
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.visualization = viz_config or VisualizationConfig()
    
    def save_all(self, base_dir: Union[str, Path]) -> None:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(base_dir / "model_config.json")
        self.training.save(base_dir / "training_config.json") 
        self.data.save(base_dir / "data_config.json")
        self.visualization.save(base_dir / "visualization_config.json")
    
    @classmethod
    def load_all(cls, base_dir: Union[str, Path]):
        base_dir = Path(base_dir)
        
        model_config = ModelConfig.load(base_dir / "model_config.json") if (base_dir / "model_config.json").exists() else ModelConfig()
        training_config = TrainingConfig.load(base_dir / "training_config.json") if (base_dir / "training_config.json").exists() else TrainingConfig()
        data_config = DataConfig.load(base_dir / "data_config.json") if (base_dir / "data_config.json").exists() else DataConfig()
        viz_config = VisualizationConfig.load(base_dir / "visualization_config.json") if (base_dir / "visualization_config.json").exists() else VisualizationConfig()
        
        return cls(model_config, training_config, data_config, viz_config)