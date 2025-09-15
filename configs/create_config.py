from core.config import ConfigManager, ModelConfig, TrainingConfig, DataConfig
import logging

logging.basicConfig(level=logging.INFO)


# Пример конфигурации для оригинальной модели Donut
donut_config = ConfigManager(
    model=ModelConfig(
        model_type='donut',
        model_name_or_path='Nyaaneet/donut-base-ru',
        image_size=(384, 384),
        max_length=768,
        hidden_size=768,
        vocab_size=50265,
        task_start_token='<s_ocr>',
        prompt_end_token=None,
        decoder_start_token_id='<s_ocr>',
        precision='bf16',
        enable_gradient_checkpointing=True,
        freeze_encoder=False,
        flash_attention=False,
        enable_torch_compile=False,
        hf_token=None,
        align_long_axis=False,
        window_size=10,
        encoder_layer=[2, 2, 14, 2],
        decoder_layer=4,
        max_position_embeddings=768,
    ),
    training=TrainingConfig(
        output_dir="./output",
        run_name="donut_exp",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=10,
        warmup_ratio=0.05,
        batch_size=16,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_interval=10,
        save_interval=1,
        eval_interval=1,
        seed=42,
        dataloader_num_workers=4,
        enable_distributed=False,
        mixed_precision=True,
        report_to="none",
        early_stopping_patience=None,
        early_stopping_threshold=0.001,
        enable_two_stage=False,
    ),
    data=DataConfig(
        data_dir="./dataset",
        apply_augmentation=True,
        augmentation_prob=0.3,
        max_rotation=8.0,
        noise_level=0.05,
        color_jitter=True,
        elastic_transform=True,
        random_perspective=True,
        max_samples_per_split=1000,
        enable_caching=True,
        cache_dir="./data_cache"
    )
)

# Пример конфигурации для кастомной модели VisionEncoderDecoder
ved_config = ConfigManager(
    model=ModelConfig(
        model_type='vision_encoder_decoder',
        model_name_or_path=None,
        encoder_name='facebook/deit-base-distilled-patch16-384',
        decoder_name='microsoft/DialoGPT-medium',
        image_size=(384, 384),
        max_length=768,
        hidden_size=768,
        vocab_size=50265,
        task_start_token='<s_ocr>',
        prompt_end_token=None,
        decoder_start_token_id='<s_ocr>',
        precision='bf16',
        enable_gradient_checkpointing=True,
        freeze_encoder=False,
        flash_attention=False,
        enable_torch_compile=False,
        hf_token=None,
    ),
    training=TrainingConfig(
        output_dir="./output",
        run_name="ved_exp",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=10,
        warmup_ratio=0.05,
        batch_size=16,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_interval=10,
        save_interval=1,
        eval_interval=1,
        seed=42,
        dataloader_num_workers=4,
        enable_distributed=False,
        mixed_precision=True,
        report_to="none",
    ),
    data=DataConfig(
        data_dir="./dataset",
        apply_augmentation=True,
        augmentation_prob=0.3,
        max_rotation=8.0,
        noise_level=0.05,
        color_jitter=True,
        elastic_transform=True,
        random_perspective=True,
        max_samples_per_split=1000,
        enable_caching=True,
        cache_dir="./data_cache"
    )
)

# Пример конфигурации для модели TrOCR
trocr_config = ConfigManager(
    model=ModelConfig(
        model_type='trocr',
        model_name_or_path=None,
        encoder_name='microsoft/swin-base-patch4-window12-384-in22k',
        decoder_name='ai-forever/ruRoberta-large',
        encoder_size='base',
        image_size=(384, 384),
        max_length=512,
        hidden_size=768,
        vocab_size=50265,
        task_start_token='<s>',
        prompt_end_token=None,
        decoder_start_token_id=None,
        precision='bf16',
        enable_gradient_checkpointing=True,
        freeze_encoder=False,
        flash_attention=False,
        enable_torch_compile=False,
        hf_token=None,
    ),
    training=TrainingConfig(
        output_dir="./output",
        run_name="trocr_exp",
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=10,
        warmup_ratio=0.1,
        batch_size=16,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_interval=10,
        save_interval=1,
        eval_interval=1,
        seed=42,
        dataloader_num_workers=4,
        enable_distributed=False,
        mixed_precision=True,
        report_to="none",
    ),
    data=DataConfig(
        data_dir="./dataset",
        apply_augmentation=True,
        augmentation_prob=0.3,
        max_rotation=8.0,
        noise_level=0.05,
        color_jitter=True,
        elastic_transform=True,
        random_perspective=True,
        max_samples_per_split=1000,
        enable_caching=True,
        cache_dir="./data_cache"
    )
)

if __name__ == "__main__":
    
    donut_config.save_all('./configs/donut_exp1')
    ved_config.save_all('./configs/ved_exp1')
    trocr_config.save_all('./configs/trocr_exp1')