from core.config import ConfigManager, ModelConfig, TrainingConfig, DataConfig



config = ConfigManager(
    model_config=ModelConfig(
        model_type='donut',
        model_name_or_path='https://huggingface.co/Nyaaneet/donut-base-ru',
        image_size=(384, 384),
        max_length=768,
        hidden_size=768,
        vocab_size=50265,
        task_start_token='<s>',
        prompt_end_token=None,
        decoder_start_token_id=None,
        precision='bf16',
        enable_gradient_checkpointing=True,
        freeze_encoder=False,
        flash_attention=True,
        enable_torch_compile=True,
        hf_token=None
    ),
    training_config=TrainingConfig(
        output_dir="./output",
        run_name="exp1",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=10,
        warmup_ratio=0.05,
        batch_size=32,
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
        synthetic_data_ratio=0.7,
        stage_transition_epochs=5,
        synthetic_lr_factor=1.0,
        real_data_lr_factor=0.5,
        inference_display_interval=100,
        enable_attention_visualization=False,
        attention_visualization_interval=500,
        attention_save_dir=None
    ),
    data_config=DataConfig(
        data_dir="./dataset",
        split_ratios=(0.8, 0.1, 0.1),
        apply_augmentation=True,
        augmentation_prob=0.3,
        max_rotation=8.0,
        noise_level=0.05,
        color_jitter=True,
        elastic_transform=True,
        random_perspective=True,
        balance_datasets=True,
        synthetic_data_path=None,
        real_data_path=None,
        max_samples_per_class=None,
        enable_caching=True,
        cache_dir="./data_cache"
    )
)


if __name__ == "__main__":
    config.save_all('./configs/exp1')