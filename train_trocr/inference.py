import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import TrOCRModel
from utils import MemoryOptimizer

log_filename = f"trocr_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инференс с моделью TrOCR для OCR в русском языке",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default='/home/student/projects/RusTitW/recognition/result/best_model', 
        help="Путь к директории модели"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Путь к входному изображению или директории с изображениями"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default='./output.json', 
        help="Путь к выходному файлу или директории"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Устройство для использования (по умолчанию cuda, если доступно)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Размер пакета для инференса"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        nargs=2, 
        default=[384, 384],
        help="Размер изображения (высота, ширина) для предобработки"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=64, 
        help="Максимальная длина генерируемого текста"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=10, 
        help="Количество лучей для лучевого поиска"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp16", "fp32"], 
        default="bf16", 
        help="Режим точности для вычислений"
    )
    parser.add_argument(
        "--save_visualizations", 
        action="store_true",
        help="Сохранять визуализации результатов"
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.0,
        help="Порог уверенности для фильтрации результатов (0.0-1.0)"
    )
    parser.add_argument(
        "--limit_samples", 
        type=int, 
        default=None,
        help="Ограничение количества обрабатываемых образцов (для отладки)"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.0,
        help="Штраф за повторение для генерации текста"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Температура для генерации текста"
    )
    
    args = parser.parse_args()
    
    if args.batch_size < 1:
        parser.error("Размер пакета должен быть не меньше 1")
    
    if args.precision == "bf16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            logger.warning("Запрошена точность BF16, но она не поддерживается вашим GPU. Откат к FP16.")
            args.precision = "fp16"
    
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        parser.error("Порог уверенности должен быть в диапазоне [0.0, 1.0]")
    
    if not Path(args.model_dir).exists():
        parser.error(f"Директория модели не существует: {args.model_dir}")
    
    if not Path(args.input).exists():
        parser.error(f"Входной путь не существует: {args.input}")
    
    output_path = Path(args.output)
    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return args


def setup_autocast(precision: str) -> Tuple[Any, bool]:
    autocast_enabled = precision in ["bf16", "fp16"]
    
    if not autocast_enabled:
        return (lambda: torch.no_grad(), False)
    
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return (
            lambda: torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"), 
            True
        )
    elif precision == "fp16":
        return (
            lambda: torch.amp.autocast(enabled=True, dtype=torch.float16, device_type="cuda"), 
            True
        )
    else:
        return (lambda: torch.no_grad(), False)


def load_images(
    image_paths: List[Path], 
    image_size: Tuple[int, int]
) -> List[Optional[Image.Image]]:
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
            images.append(img)
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {path}: {e}")
            images.append(None)
    return images


def process_batch(
    model: TrOCRModel, 
    image_batch: List[Image.Image], 
    image_paths: List[Path],
    generation_params: Dict[str, Any],
    autocast_fn: Any,
    autocast_enabled: bool
) -> List[Dict[str, Any]]:

    results = []
    valid_images = [img for img in image_batch if img is not None]
    valid_indices = [i for i, img in enumerate(image_batch) if img is not None]
    
    if not valid_images:
        for path in image_paths:
            results.append({
                "image_path": str(path),
                "text": "",
                "error": "Ошибка загрузки изображения"
            })
        return results

    try:
        processor_outputs = model.processor(
            images=valid_images, 
            return_tensors="pt", 
            padding=True
        )
        pixel_values = processor_outputs.pixel_values.to(model.device)
    except Exception as e:
        logger.error(f"Ошибка при предобработке изображений: {e}")
        for path in image_paths:
            results.append({
                "image_path": str(path),
                "text": "",
                "error": f"Ошибка предобработки: {str(e)}"
            })
        return results
    
    try:
        with autocast_fn() if autocast_enabled else torch.no_grad():
            if hasattr(model, "generate") and callable(model.generate):
                logger.info("Использование model.generate()")
                
                safe_params = generation_params.copy()
                safe_params.pop('decoder_input_ids', None)
                
                generated_texts = model.generate(
                    pixel_values=pixel_values,
                    max_length=safe_params.pop('max_length', 64),
                    num_beams=safe_params.pop('num_beams', 4),
                    early_stopping=safe_params.pop('early_stopping', True),
                    temperature=safe_params.pop('temperature', 1.0),
                    top_k=safe_params.pop('top_k', 50),
                    top_p=safe_params.pop('top_p', 1.0),
                    repetition_penalty=safe_params.pop('repetition_penalty', 1.0),
                    length_penalty=safe_params.pop('length_penalty', 1.0),
                    no_repeat_ngram_size=safe_params.pop('no_repeat_ngram_size', 0)
                )
            
            elif hasattr(model, "model") and hasattr(model.model, "generate") and callable(model.model.generate):
                logger.info("Использование model.model.generate()")
                
                generated_ids = model.model.generate(
                    pixel_values=pixel_values,
                    **generation_params
                )
                
                generated_texts = model.processor.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            
            else:
                logger.info("Использование метода forward и ручное декодирование")
                
                if hasattr(model, "model") and hasattr(model.model, "encoder"):
                    batch_size = pixel_values.size(0)
                    if hasattr(model.model, "config") and hasattr(model.model.config, "decoder_start_token_id") and \
                       model.model.config.decoder_start_token_id is not None:
                        start_token_id = model.model.config.decoder_start_token_id
                        decoder_input_ids = torch.full(
                            (batch_size, 1),
                            start_token_id,
                            dtype=torch.long,
                            device=model.device
                        )
                    elif hasattr(model, "tokenizer") and model.tokenizer.bos_token_id is not None:
                        decoder_input_ids = torch.full(
                            (batch_size, 1),
                            model.tokenizer.bos_token_id,
                            dtype=torch.long,
                            device=model.device
                        )
                    else:
                        decoder_input_ids = torch.zeros(
                            (batch_size, 1),
                            dtype=torch.long,
                            device=model.device
                        )
                    
                    max_length = generation_params.get("max_length", 64)
                    eos_token_id = model.processor.tokenizer.eos_token_id
                    
                    all_generated_ids = decoder_input_ids.clone()
                    for _ in range(max_length - 1):
                        outputs = model.model(
                            pixel_values=pixel_values,
                            decoder_input_ids=all_generated_ids,
                            return_dict=True
                        )
                        
                        next_token_logits = outputs.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        all_generated_ids = torch.cat([all_generated_ids, next_tokens], dim=1)
                        
                        if eos_token_id is not None and (next_tokens == eos_token_id).all():
                            break
                    
                    generated_texts = model.processor.tokenizer.batch_decode(
                        all_generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                
                else:
                    outputs = model.model(pixel_values=pixel_values, return_dict=True)
                    
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    generated_texts = []
                    for ids in predicted_ids:
                        text = model.processor.tokenizer.decode(
                            ids.tolist(), 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        generated_texts.append(text)
    
    except Exception as e:
        logger.error(f"Ошибка при генерации текста: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        for path in image_paths:
            results.append({
                "image_path": str(path),
                "text": "",
                "error": f"Ошибка генерации: {str(e)}"
            })
        return results

    for i, path in enumerate(image_paths):
        if i in valid_indices:
            idx = valid_indices.index(i)
            if idx < len(generated_texts):
                text = generated_texts[idx]
                results.append({
                    "image_path": str(path),
                    "text": text,
                    "confidence": 1.0
                })
            else:
                results.append({
                    "image_path": str(path),
                    "text": "",
                    "error": "Индекс за пределами сгенерированных текстов"
                })
        else:
            results.append({
                "image_path": str(path),
                "text": "",
                "error": "Недействительное изображение"
            })
    
    return results


def save_visualization(
    image_path: Path,
    text: str,
    output_dir: Path,
    conf: float = 1.0
) -> None:
    try:
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(f"Распознано (уверенность: {conf:.2f})")
        
        ax.text(
            0.5, -0.12, text,
            size=12, ha="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9)
        )
        
        ax.axis('off')
        plt.tight_layout()
        viz_filename = image_path.stem + "_ocr.png"
        plt.savefig(output_dir / viz_filename, dpi=100, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Не удалось создать визуализацию для {image_path}: {e}")


def create_summary(results: List[Dict[str, Any]], output_path: Path) -> None:
    summary = {
        "total_images": len(results),
        "successful_recognitions": sum(1 for r in results if "error" not in r),
        "recognition_errors": sum(1 for r in results if "error" in r),
        "empty_texts": sum(1 for r in results if "error" not in r and len(r["text"].strip()) == 0),
        "average_text_length": np.mean([len(r["text"]) for r in results if "error" not in r]),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Сводка инференса: всего изображений: {summary['total_images']}, "
                f"успешно распознано: {summary['successful_recognitions']}, "
                f"ошибок: {summary['recognition_errors']}, "
                f"пустых текстов: {summary['empty_texts']}")


def main() -> None:
    args = parse_args()
    image_size = tuple(args.image_size)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Загрузка модели TrOCR из {args.model_dir}")
    
    MemoryOptimizer.optimize_memory_usage()
    
    try:
        model = TrOCRModel.from_pretrained(
            args.model_dir,
            device=device,
            precision=args.precision,
            flash_attention=False
        )
        model.eval()
        
        logger.info(f"Информация о модели:")
        logger.info(f"  Тип модели: {type(model)}")
        if hasattr(model, 'model'):
            logger.info(f"  Внутренняя модель: {type(model.model)}")
        if hasattr(model, 'processor'):
            logger.info(f"  Процессор: {type(model.processor)}")
            logger.info(f"  Токенизатор: {type(model.processor.tokenizer)}")
            
        has_generate = hasattr(model, 'generate')
        has_model_generate = hasattr(model.model, 'generate') if hasattr(model, 'model') else False
        logger.info(f"  Наличие метода generate: model.generate={has_generate}, model.model.generate={has_model_generate}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)
    
    autocast_fn, autocast_enabled = setup_autocast(args.precision)
    
    generation_params = {
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "early_stopping": True,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        # "return_dict_in_generate": True, 
        # "output_scores": True
    }
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    all_results = []
    
    if input_path.is_file():
        logger.info(f"Обработка изображения: {input_path}")
        
        images = load_images([input_path], image_size)
        results = process_batch(
            model,
            images,
            [input_path],
            generation_params,
            autocast_fn,
            autocast_enabled
        )
        
        all_results.extend(results)
        
        if results and not "error" in results[0]:
            logger.info(f"Распознанный текст: {results[0]['text']}")
        else:
            logger.error(f"Ошибка при обработке изображения: {results[0].get('error', 'Неизвестная ошибка')}")
            
        if args.save_visualizations and results:
            viz_dir = output_path.parent / "visualizations"
            save_visualization(
                input_path,
                results[0].get("text", ""),
                viz_dir,
                results[0].get("confidence", 1.0)
            )
        
    elif input_path.is_dir():
        logger.info(f"Обработка изображений в директории: {input_path}")
        
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_paths = sorted([
            p for p in input_path.glob("**/*") 
            if p.suffix.lower() in extensions
        ])
        
        if not image_paths:
            logger.warning(f"Изображения не найдены в {input_path}")
            sys.exit(0)
        
        if args.limit_samples and args.limit_samples > 0:
            image_paths = image_paths[:args.limit_samples]
            logger.info(f"Ограничено до {len(image_paths)} образцов")
        else:
            logger.info(f"Найдено {len(image_paths)} изображений")
        
        for batch_start in tqdm(range(0, len(image_paths), args.batch_size), desc="Обработка пакетов"):
            batch_paths = image_paths[batch_start:batch_start + args.batch_size]
            
            batch_images = load_images(batch_paths, image_size)
            batch_results = process_batch(
                model,
                batch_images,
                batch_paths,
                generation_params,
                autocast_fn,
                autocast_enabled
            )
            
            all_results.extend(batch_results)
            
            if args.save_visualizations:
                viz_dir = output_path.parent / "visualizations"
                for path_idx, result in enumerate(batch_results):
                    if "error" not in result:
                        save_visualization(
                            batch_paths[path_idx],
                            result["text"],
                            viz_dir,
                            result.get("confidence", 1.0)
                        )
            
            if (batch_start // args.batch_size) % 10 == 0 and batch_start > 0:
                success_rate = sum(1 for r in all_results if "error" not in r) / len(all_results)
                logger.info(f"Обработано {len(all_results)}/{len(image_paths)} изображений. "
                            f"Успешно: {success_rate:.1%}")
    else:
        logger.error(f"Входной путь не существует: {input_path}")
        sys.exit(1)
    
    if args.confidence_threshold > 0:
        filtered_results = [
            r for r in all_results 
            if "confidence" not in r or r["confidence"] >= args.confidence_threshold
        ]
        logger.info(f"Отфильтровано {len(all_results) - len(filtered_results)} результатов "
                    f"по порогу уверенности {args.confidence_threshold}")
        all_results = filtered_results
    
    if output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {output_path}")
    else:
        results_file = output_path / "ocr_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {results_file}")
    
    summary_path = output_path.parent / "summary.json" if output_path.suffix == ".json" else output_path / "summary.json"
    create_summary(all_results, summary_path)
    
    if len(all_results) > 1:
        txt_path = output_path.with_suffix('.txt') if output_path.suffix == ".json" else output_path / "ocr_text.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for result in all_results:
                if "error" not in result:
                    f.write(result["text"] + "\n\n")
        logger.info(f"Связный текст сохранен в {txt_path}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        elapsed_time = time.time() - start_time
        logger.info(f"Инференс завершен за {elapsed_time:.2f} секунд")
    except Exception as e:
        logger.error(f"Необработанное исключение: {e}", exc_info=True)
        sys.exit(1)