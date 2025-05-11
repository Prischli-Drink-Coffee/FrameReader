import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import set_seed

from dataset import TrOCRDataModule
from model import TrOCRModel
from trainer import Trainer
from utils import MemoryOptimizer, TrainingSpeedup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение модели TrOCR для GPU A100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Группа аргументов данных
    data_group = parser.add_argument_group("Аргументы данных")
    data_group.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Путь к директории с данными с папками train/validation/test"
    )
    data_group.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="Путь к директории для сохранения моделей и журналов"
    )
    data_group.add_argument(
        "--pretrained_model_name", 
        type=str, 
        default="raxtemur/trocr-base-ru", # vit-rugpt2-image-captioning # raxtemur/trocr-base-ru
        help="Имя предобученной модели из HuggingFace"
    )
    data_group.add_argument(
        "--encoder_model", 
        type=str, 
        default=None,
        help="Имя модели энкодера (если не указано, используется значение по умолчанию)"
    )
    data_group.add_argument(
        "--decoder_model", 
        type=str, 
        default=None,
        help="Имя модели декодера (если не указано, используется значение по умолчанию)"
    )
    data_group.add_argument(
        "--encoder_size", 
        type=str, 
        choices=["base", "large", "xlarge"],
        default=None,
        help="Размер предопределенного энкодера, если не задано конкретное имя"
    )
    data_group.add_argument(
        "--fraction", 
        type=float, 
        default=1.0,
        help="Доля данных для использования (0.0-1.0). Полезно для быстрых экспериментов."
    )

    # Группа аргументов обучения
    train_group = parser.add_argument_group("Аргументы обучения")
    train_group.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Размер пакета для обучения и оценки"
    )
    train_group.add_argument(
        "--num_workers", 
        type=int, 
        default=8, 
        help="Количество рабочих процессов загрузчика данных"
    )
    train_group.add_argument(
        "--max_length", 
        type=int, 
        default=64, 
        help="Максимальная длина последовательности для токенизации"
    )
    train_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Пиковая скорость обучения для оптимизатора"
    )
    train_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.001, 
        help="Затухание весов для регуляризации"
    )
    train_group.add_argument(
        "--num_epochs", 
        type=int, 
        default=50,
        help="Количество эпох обучения"
    )
    train_group.add_argument(
        "--log_interval", 
        type=int, 
        default=1, 
        help="Количество шагов между обновлениями журнала"
    )
    train_group.add_argument(
        "--save_interval", 
        type=int, 
        default=1, 
        help="Количество эпох между сохранением контрольных точек"
    )
    train_group.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.0005, 
        help="Доля общих шагов для линейного разогрева"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=32, 
        help="Количество шагов для накопления градиентов"
    )
    train_group.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=100.0, 
        help="Максимальная норма для отсечения градиентов"
    )
    train_group.add_argument(
        "--report_to", 
        type=str, 
        choices=["none", "tensorboard", "wandb"],
        default="tensorboard",
        help="Платформа для отчетов об обучении"
    )
    
    # Группа аргументов оптимизации
    optim_group = parser.add_argument_group("Аргументы оптимизации")
    optim_group.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp16", "fp32"], 
        default="bf16", 
        help="Режим точности для вычислений (рекомендуется bf16 для A100)"
    )
    optim_group.add_argument(
        "--image_size", 
        type=int, 
        nargs=2, 
        default=[384, 384], 
        help="Размер изображения (высота, ширина) - меньшие значения экономят память"
    )
    optim_group.add_argument(
        "--freeze_encoder", 
        action="store_true", 
        default=False,
        help="Заморозить параметры энкодера для экономии памяти и ускорения обучения"
    )
    optim_group.add_argument(
        "--enable_gradient_checkpointing", 
        action="store_true", 
        default=True,
        help="Включить проверку градиентов для экономии памяти"
    )
    optim_group.add_argument(
        "--enable_torch_compile", 
        action="store_true", 
        default=False,
        help="Включить torch.compile для ускорения (требуется PyTorch 2.0+)"
    )
    optim_group.add_argument(
        "--flash_attention", 
        action="store_true", 
        default=False,
        help="Использовать FlashAttention если доступно"
    )
    optim_group.add_argument(
        "--use_8bit_decoder", 
        action="store_true", 
        default=False,
        help="Использовать 8-битный декодер для экономии памяти"
    )
    optim_group.add_argument(
        "--memory_efficient", 
        action="store_true", 
        default=True,
        help="Включить оптимизации памяти"
    )
    optim_group.add_argument(
        "--cache_images", 
        action="store_true", 
        default=False,
        help="Кэшировать изображения в памяти для быстрого доступа (только для небольших наборов данных)"
    )

    # Группа разных аргументов
    misc_group = parser.add_argument_group("Разные аргументы")
    misc_group.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Случайное начальное число для воспроизводимости"
    )
    misc_group.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Устройство для использования (по умолчанию cuda, если доступно)"
    )
    misc_group.add_argument(
        "--eval_only", 
        action="store_true", 
        help="Запустить только оценку, без обучения"
    )
    misc_group.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Возобновить обучение с указанной контрольной точки"
    )
    misc_group.add_argument(
        "--enable_distributed", 
        action="store_true", 
        default=False,
        help="Включить распределенное обучение на нескольких GPU"
    )
    
    args = parser.parse_args()
    
    if args.batch_size < 1:
        parser.error("Размер пакета должен быть не меньше 1")

    if args.fraction <= 0.0 or args.fraction > 1.0:
        parser.error("Доля должна быть в диапазоне (0.0, 1.0]")

    if args.precision == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            logger.warning("Запрошена точность BF16, но она не поддерживается вашим GPU. Откат к FP16.")
            args.precision = "fp16"
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_properties(0).name
        if "A100" in device_name and args.batch_size == 128:
            if args.encoder_size == "xlarge":
                suggested_bs = 64
            elif args.encoder_size == "large":
                suggested_bs = 96
            else:
                suggested_bs = 128
            
            logger.info(f"Обнаружен GPU A100 с размером энкодера {args.encoder_size}, рекомендуемый размер пакета: {suggested_bs}")
            if args.batch_size > suggested_bs:
                logger.warning(f"Автоматическое снижение размера пакета с {args.batch_size} до {suggested_bs} для энкодера {args.encoder_size} на A100")
                args.batch_size = suggested_bs
    
    if args.enable_torch_compile and not hasattr(torch, "compile"):
        logger.warning("Запрошено torch.compile, но он не доступен в текущей версии PyTorch. Отключено.")
        args.enable_torch_compile = False
    
    return args


def setup_environment(seed: int) -> None:
    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        device_count = torch.cuda.device_count()
        logger.info(f"Найдено {device_count} устройств(о) CUDA")
        
        total_memory = 0
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            total_memory += memory_gb
            logger.info(f"Устройство CUDA {i}: {props.name} с {memory_gb:.1f}ГБ памяти")
        
        logger.info(f"Общая доступная видеопамять: {total_memory:.1f}ГБ")
 
        if torch.cuda.is_bf16_supported():
            logger.info("Обнаружена поддержка BF16 - рекомендуется для обучения")
        else:
            logger.info("Поддержка BF16 не обнаружена - рекомендуется FP16")
    else:
        logger.warning("CUDA не доступна. Обучение будет выполняться на CPU, что может быть очень медленно.")


def setup_output_directory(output_dir: Union[str, Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "samples").mkdir(exist_ok=True)
    
    logger.info(f"Создана выходная директория: {output_path}")
    return output_path


def main() -> None:

    if torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.info("Включен режим CUDA_LAUNCH_BLOCKING для точной локализации ошибок CUDA")

    args = parse_args()
    setup_environment(args.seed)
    output_dir = setup_output_directory(args.output_dir)
    
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Инициализация модели с предобученной моделью: {args.pretrained_model_name}")
    
    if args.encoder_model or args.decoder_model:
        logger.info(f"Использование кастомной модели с энкодером: {args.encoder_model or f'[{args.encoder_size}]'} "
                   f"и декодером: {args.decoder_model or f'[{args.encoder_size}]'}")
    
    model = TrOCRModel(
        pretrained_model_name=args.pretrained_model_name,
        encoder_model_name=args.encoder_model,
        decoder_model_name=args.decoder_model,
        encoder_size=args.encoder_size,
        max_length=args.max_length,
        device=args.device,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        freeze_encoder=args.freeze_encoder,
        precision=args.precision,
        img_size=tuple(args.image_size),
        enable_torch_compile=args.enable_torch_compile,
        flash_attention=args.flash_attention,
        use_8bit_decoder=args.use_8bit_decoder
    )

    if hasattr(model, 'tokenizer') and hasattr(model, 'decoder'):
        decoder_vocab_size = getattr(model.decoder.config, "vocab_size", None)
        tokenizer_vocab_size = len(model.tokenizer)
        logger.info(f"Размер словаря декодера: {decoder_vocab_size}, размер словаря токенизатора: {tokenizer_vocab_size}")
        
        if decoder_vocab_size is not None and decoder_vocab_size < tokenizer_vocab_size:
            logger.warning(f"Размер словаря декодера ({decoder_vocab_size}) меньше размера словаря токенизатора ({tokenizer_vocab_size})")
            logger.info("Попытка расширения словаря декодера")
            try:
                model.decoder.resize_token_embeddings(tokenizer_vocab_size)
                logger.info(f"Словарь декодера успешно расширен до {tokenizer_vocab_size}")
            except Exception as e:
                logger.error(f"Не удалось расширить словарь декодера: {e}")
    
    logger.info(f"Загрузка данных из {args.data_dir} с оптимизированным размером изображения {args.image_size}")
    data_module = TrOCRDataModule(
        processor=model.processor,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        image_size=tuple(args.image_size),
        fraction=args.fraction,
        apply_augmentation=True,
        distributed=args.enable_distributed,
        pin_memory=True,
        cache_images=args.cache_images,
        seed=args.seed
    )
    
    if args.memory_efficient:
        MemoryOptimizer.optimize_memory_usage()
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats()
            allocated = stats.get("allocated_bytes.all.current", 0) / (1024**3)
            logger.info(f"Используемая память GPU после инициализации: {allocated:.2f}ГБ")

    trainer = Trainer(
        model=model,
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
        enable_distributed=args.enable_distributed,
        report_to=args.report_to,
        memory_efficient=args.memory_efficient
    )
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Возобновление обучения из контрольной точки: {checkpoint_path}")
            
            if (checkpoint_path / "pytorch_model.bin").exists() or (checkpoint_path / "model.safetensors").exists():
                model = TrOCRModel.from_pretrained(
                    checkpoint_path,
                    device=args.device,
                    precision=args.precision,
                    flash_attention=args.flash_attention,
                    enable_torch_compile=args.enable_torch_compile
                )
                trainer.model = model
                logger.info(f"Модель успешно загружена из {checkpoint_path}")
            else:
                logger.warning(f"Файлы модели не найдены в {checkpoint_path}, загружаем только состояния обучения")
            
            trainer_state_path = checkpoint_path / "trainer_state.pt"
            if trainer_state_path.exists():
                try:
                    trainer_state = torch.load(trainer_state_path, map_location=trainer.device)
                    
                    if 'optimizer_state' in trainer_state:
                        trainer.optimizer.load_state_dict(trainer_state['optimizer_state'])
                        logger.info("Состояние оптимизатора восстановлено")
                    
                    if 'epoch' in trainer_state:
                        start_epoch = trainer_state['epoch'] + 1
                        logger.info(f"Обучение продолжится с эпохи {start_epoch}")
                    
                    if 'metrics' in trainer_state:
                        trainer.metrics = trainer_state['metrics']
                        logger.info("Метрики восстановлены")
                    
                    if 'global_step' in trainer_state:
                        trainer.global_step = trainer_state['global_step']
                        logger.info(f"Глобальный шаг обучения восстановлен: {trainer.global_step}")
                    
                    if 'scheduler_state' in trainer_state:
                        total_steps = len(data_module.train_dataloader()) * args.num_epochs // args.gradient_accumulation_steps
                        trainer.scheduler = trainer._create_scheduler(total_steps)
                        trainer.scheduler.load_state_dict(trainer_state['scheduler_state'])
                        logger.info("Состояние планировщика восстановлено")
                        
                except Exception as e:
                    logger.error(f"Ошибка при загрузке состояния тренера: {e}")
                    start_epoch = 0
            else:
                logger.warning(f"Файл состояния тренера не найден в {checkpoint_path}, начинаем обучение с эпохи 0")
        else:
            logger.warning(f"Контрольная точка не найдена: {checkpoint_path}")
    
    if args.eval_only:
        perform_evaluation(trainer, data_module, output_dir)
    else:
        perform_training(trainer, output_dir, start_epoch)
    
    try:
        if args.eval_only:
            perform_evaluation(trainer, data_module, output_dir)
        else:
            perform_training(trainer, output_dir)
    except KeyboardInterrupt:
        logger.info("\nОбучение прервано пользователем. Сохранение промежуточных результатов...")
        try:
            model.save_pretrained(output_dir / "interrupted_model")
            logger.info(f"Промежуточная модель сохранена в {output_dir / 'interrupted_model'}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении промежуточной модели: {e}")


def perform_evaluation(
    trainer: Trainer, 
    data_module: TrOCRDataModule,
    output_dir: Path
) -> None:

    logger.info("Запуск только оценки")
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    
    results = {}
    
    if val_dataloader:
        logger.info("Оценка на валидационном наборе")
        val_loss = trainer._evaluate(val_dataloader)
        results["validation_loss"] = float(val_loss)
        logger.info(f"Потери валидации: {val_loss:.4f}")
    else:
        logger.warning("Загрузчик валидационных данных недоступен")
    
    if test_dataloader:
        logger.info("Оценка на тестовом наборе")
        test_loss = trainer._evaluate(test_dataloader)
        results["test_loss"] = float(test_loss)
        logger.info(f"Потери теста: {test_loss:.4f}")
    else:
        logger.warning("Загрузчик тестовых данных недоступен")
    
    with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты оценки сохранены в {output_dir / 'evaluation_results.json'}")
    
    if hasattr(trainer.model, "generate") and test_dataloader:
        logger.info("Генерация образцов из тестового набора")
        generate_samples(trainer.model, test_dataloader, output_dir / "samples", limit=10)


def generate_samples(
    model: TrOCRModel, 
    dataloader: DataLoader, 
    output_dir: Path,
    limit: int = 10
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= limit:
                break
                
            generated_ids = model.model.generate(
                pixel_values=batch["pixel_values"].to(model.device),
                max_length=model.max_length,
                num_beams=4,
                early_stopping=True,
            )
            generated_text = model.processor.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            decoded_labels = batch["labels"].clone()
            decoded_labels[decoded_labels == -100] = model.processor.tokenizer.pad_token_id
            reference_text = model.processor.tokenizer.batch_decode(
                decoded_labels, 
                skip_special_tokens=True
            )
            
            for j, (pred, ref) in enumerate(zip(generated_text, reference_text)):
                results.append({
                    "sample_id": i * dataloader.batch_size + j,
                    "reference": ref,
                    "prediction": pred,
                    "match": pred.strip() == ref.strip()
                })
    
    with open(output_dir / "generation_samples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    matches = sum(1 for r in results if r["match"])
    accuracy = matches / len(results) if results else 0
    logger.info(f"Точность на {len(results)} образцах: {accuracy:.2%}")


def perform_training(
    trainer: Trainer, 
    output_dir: Path,
    start_epoch: int = 0
) -> None:
    logger.info(f"Запуск оптимизированного обучения с эпохи {start_epoch}")
    start_time = time.time()
    
    try:
        metrics = trainer.train(start_epoch=start_epoch)

        total_time = time.time() - start_time
        logger.info(f"Общее время обучения: {total_time/60:.2f} минут")

        if "val_loss" in metrics and metrics["val_loss"]:
            logger.info(f"Финальные потери валидации: {metrics['val_loss'][-1]:.4f}")
        
        logger.info(f"Финальные потери обучения: {metrics['train_loss'][-1]:.4f}")

        if "time_per_epoch" in metrics and metrics["time_per_epoch"]:
            avg_time = sum(metrics["time_per_epoch"]) / len(metrics["time_per_epoch"])
            logger.info(f"Среднее время на эпоху: {avg_time/60:.2f} минут")
        
        if hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(output_dir / "final_model")
            logger.info(f"Финальная модель сохранена в {output_dir / 'final_model'}")
        
        if hasattr(trainer.model, "generate") and trainer.data_module.val_dataloader():
            logger.info("Генерация образцов из валидационного набора")
            generate_samples(
                trainer.model, 
                trainer.data_module.val_dataloader(), 
                output_dir / "samples"
            )
        
        logger.info(f"Обучение успешно завершено. Модель сохранена в {output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Необработанное исключение: {e}", exc_info=True)
        sys.exit(1)