import os
import sys
import logging
import argparse
import json
from pathlib import Path
import torch
from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from model import DonutModel
from dataset import DonutDataModule
from trainer import DonutTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Обучение модели Donut")
    
    # Параметры модели
    model_group = parser.add_argument_group("Параметры модели")
    model_group.add_argument("--model_name_or_path", type=str, default="Akajackson/donut_rus",
                        help="Имя или путь к предобученной модели")
    model_group.add_argument("--pretrained_checkout", type=str, default=None, 
                        help="Путь к контрольной точке для продолжения обучения")
    model_group.add_argument("--image_size", type=int, nargs=2, default=[384, 384],
                        help="Размер изображения для модели [высота, ширина]")
    model_group.add_argument("--max_length", type=int, default=64,
                        help="Максимальная длина последовательности токенов")
    model_group.add_argument("--task_start_token", type=str, default="<s_500k>",
                        help="Токен начала задачи")
    model_group.add_argument("--prompt_end_token", type=str, default="<s_prompt>",
                        help="Токен конца промпта (если None, используется task_start_token)")
    model_group.add_argument("--sort_json_key", action="store_true",
                        help="Сортировать ключи JSON при преобразовании")
    
    # Параметры обучения
    train_group = parser.add_argument_group("Параметры обучения")
    train_group.add_argument("--output_dir", type=str, default="./output_donut",
                        help="Директория для сохранения обученной модели")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Скорость обучения")
    train_group.add_argument("--weight_decay", type=float, default=0.005,
                        help="Коэффициент регуляризации весов")
    train_group.add_argument("--num_epochs", type=int, default=5,
                        help="Количество эпох обучения")
    train_group.add_argument("--warmup_ratio", type=float, default=0.005,
                        help="Доля шагов разогрева для планировщика")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Количество шагов для накопления градиента")
    train_group.add_argument("--max_grad_norm", type=float, default=100.0,
                        help="Максимальная норма градиента")
    train_group.add_argument("--batch_size", type=int, default=80,
                        help="Размер пакета для обучения")
    train_group.add_argument("--save_interval", type=int, default=1,
                        help="Интервал сохранения контрольных точек (в эпохах)")
    train_group.add_argument("--log_interval", type=int, default=1,
                        help="Интервал логирования (в шагах)")
    train_group.add_argument("--early_stopping_patience", type=int, default=60,
                        help="Терпение для раннего останова")
    train_group.add_argument("--early_stopping_threshold", type=float, default=0.1,
                        help="Порог улучшения для раннего останова")
    
    # Параметры данных
    data_group = parser.add_argument_group("Параметры данных")
    data_group.add_argument("--data_dir", type=str, required=True,
                        help="Директория с данными")
    data_group.add_argument("--num_workers", type=int, default=8,
                        help="Количество рабочих процессов для загрузки данных")
    data_group.add_argument("--cache_images", action="store_true",
                        help="Кэшировать изображения в памяти")
    data_group.add_argument("--apply_augmentation", action="store_true",
                        help="Применять аугментации для тренировочных данных")
    data_group.add_argument("--train_limit_samples", type=int, default=None,
                        help="Ограничение количества образцов для тренировки (для отладки)")
    data_group.add_argument("--val_limit_samples", type=int, default=800,
                        help="Ограничение количества образцов для валидации (для отладки)")

    # Параметры аугментаций
    augmentation_group = parser.add_argument_group("Параметры аугментаций")
    augmentation_group.add_argument("--augmentation_prob", type=float, default=0.3,
                        help="Вероятность применения каждой аугментации")
    augmentation_group.add_argument("--max_rotation", type=float, default=8.0,
                        help="Максимальный угол поворота в градусах")
    augmentation_group.add_argument("--brightness_min", type=float, default=0.8,
                        help="Минимальный фактор яркости")
    augmentation_group.add_argument("--brightness_max", type=float, default=1.2,
                        help="Максимальный фактор яркости")
    augmentation_group.add_argument("--contrast_min", type=float, default=0.8,
                        help="Минимальный фактор контрастности")
    augmentation_group.add_argument("--contrast_max", type=float, default=1.2,
                        help="Максимальный фактор контрастности")
    augmentation_group.add_argument("--blur_min", type=int, default=0,
                        help="Минимальный радиус размытия")
    augmentation_group.add_argument("--blur_max", type=int, default=2,
                        help="Максимальный радиус размытия")
    augmentation_group.add_argument("--noise_level", type=float, default=0.05,
                        help="Уровень шума (соль и перец)")
    augmentation_group.add_argument("--sharpness_min", type=float, default=0.8,
                        help="Минимальный фактор резкости")
    augmentation_group.add_argument("--sharpness_max", type=float, default=1.5,
                        help="Максимальный фактор резкости")
    
    # Параметры вычислений
    compute_group = parser.add_argument_group("Параметры вычислений")
    compute_group.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    compute_group.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Точность вычислений")
    compute_group.add_argument("--distributed", action="store_true",
                        help="Использовать распределенное обучение")
    compute_group.add_argument("--report_to", type=str, default="none",
                        choices=["tensorboard", "wandb", "none"],
                        help="Система для логирования результатов")
    
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def setup_model(args):
    if args.pretrained_checkout:
        logger.info(f"Загрузка модели из контрольной точки: {args.pretrained_checkout}")

        if not Path(args.pretrained_checkout).exists():
            raise FileNotFoundError(f"Контрольная точка не найдена: {args.pretrained_checkout}")

        return DonutModel.from_pretrained(
            args.pretrained_checkout,
            device=args.device,
            precision=args.precision,
            max_length=args.max_length,
            image_size=args.image_size,
            task_start_token=args.task_start_token,
            prompt_end_token=args.prompt_end_token
        )

    logger.info(f"Загрузка предобученной модели: {args.model_name_or_path}")
    config = VisionEncoderDecoderConfig.from_pretrained(args.model_name_or_path)
    config.encoder.image_size = args.image_size
    config.decoder.max_length = args.max_length

    processor = DonutProcessor.from_pretrained(args.model_name_or_path, use_fast=True)
    processor.image_processor.size = args.image_size[::-1]  # (width, height)
    processor.image_processor.do_align_long_axis = False

    model = VisionEncoderDecoderModel.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    return DonutModel(
        model=model,
        processor=processor,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        task_start_token=args.task_start_token,
        prompt_end_token=args.prompt_end_token
    )


def setup_data_module(args, processor):
    logger.info(f"Настройка модуля данных из {args.data_dir}")
    
    return DonutDataModule(
        processor=processor,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        task_start_token=args.task_start_token,
        prompt_end_token=args.prompt_end_token,
        sort_json_key=args.sort_json_key,
        image_size=args.image_size,
        apply_augmentation=args.apply_augmentation,
        distributed=args.distributed,
        pin_memory=True,
        cache_images=args.cache_images,
        train_limit_samples=args.train_limit_samples,
        val_limit_samples=args.val_limit_samples,
        augmentation_prob=args.augmentation_prob,
        max_rotation=args.max_rotation,
        brightness_range=(args.brightness_min, args.brightness_max),
        contrast_range=(args.contrast_min, args.contrast_max),
        blur_range=(args.blur_min, args.blur_max),
        noise_level=args.noise_level,
        sharpness_range=(args.sharpness_min, args.sharpness_max)
    )


def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    start_epoch = 0
    donut_model = setup_model(args)
    data_module = setup_data_module(args, donut_model.processor)
    dataset_info = data_module.get_sample_info()
    logger.info(f"Информация о датасете: {dataset_info}")

    trainer = DonutTrainer(
        model=donut_model,
        data_module=data_module,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
        enable_distributed=args.distributed,
        report_to=args.report_to,
        memory_efficient=True,
        evaluate_during_training=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    if args.pretrained_checkout and Path(args.pretrained_checkout).exists():
        start_epoch = trainer.load_checkpoint(args.pretrained_checkout)
        logger.info(f"Продолжаем обучение с эпохи {start_epoch}")
    
    try:
        metrics = trainer.train(start_epoch=start_epoch)

        logger.info("Обучение завершено. Финальные метрики:")
        for metric, values in metrics.items():
            if values:
                logger.info(f"{metric}: {values[-1]}")

        trainer._plot_metrics(metrics, args.num_epochs)
        
        logger.info(f"Модель сохранена в {output_dir}")
        
        return 0
    except KeyboardInterrupt:
        logger.info("Обучение прервано пользователем")
        if hasattr(trainer, "model"):
            try:
                trainer.model.save_pretrained(output_dir / "interrupted_model")
                logger.info(f"Прерванная модель сохранена в {output_dir / 'interrupted_model'}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении прерванной модели: {e}")
        return 1
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
