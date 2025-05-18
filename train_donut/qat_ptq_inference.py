import logging
import os
import sys
import argparse
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nltk import edit_distance
import random

from inference import TextCleanup, EditDistanceMetric
from model import DonutModel

try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Torch-TensorRT не установлен. Для использования TensorRT установите библиотеку.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TRTInferenceEngine:
    
    def __init__(
        self,
        model_path: Union[str, Path],
        processor_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        image_size: Optional[tuple] = (384, 384),
        max_length: int = 64,
        num_beams: int = 5,
        task_start_token: str = "<s_500k>",
        prompt_end_token: Optional[str] = "<s_prompt>"
    ):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        if processor_path is None:
            processor_path = self.model_path.parent
        else:
            processor_path = Path(processor_path) if isinstance(processor_path, str) else processor_path
            
        self.processor_path = processor_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = max_length
        self.num_beams = num_beams
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token or task_start_token

        if not TENSORRT_AVAILABLE:
            raise ImportError("Torch-TensorRT не установлен. Установите его для использования TRT моделей.")
        
        logger.info(f"Инициализация TRT движка инференса из {model_path}")
        
        start_time = time.time()
        try:
            self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            logger.info(f"TRT модель загружена за {time.time() - start_time:.2f} с")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели TensorRT: {e}")
            raise
        
        try:
            temp_model = DonutModel.from_pretrained(
                self.processor_path,
                device="cpu",
                max_length=self.max_length,
                task_start_token=self.task_start_token,
                prompt_end_token=self.prompt_end_token
            )
            self.processor = temp_model.processor
            self.processor.image_processor.size = image_size
            self.tokenizer = self.processor.tokenizer

            self.eos_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
            
            logger.info(f"Процессор загружен из {self.processor_path}")

            del temp_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке процессора: {e}")
            raise
        
        logger.info(f"TRT движок инициализирован на устройстве {self.device}")
    
    def prepare_prompt(self, prompt: Optional[str] = None) -> str:
        if prompt is None or prompt.strip() == "":
            return self.task_start_token
        else:
            return f"{self.task_start_token}{prompt}{self.prompt_end_token}"
    
    def process_image(
        self, 
        image: Union[str, Path, Image.Image],
        max_length: int = 64,
        prompt: Optional[str] = None,
        return_json: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Dict[str, Any]]:

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        input_prompt = self.prepare_prompt(prompt)
        decoder_input_ids = self.tokenizer(
            input_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(self.device)
        
        padded_input_ids = torch.full(
            (decoder_input_ids.size(0), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        seq_length = min(decoder_input_ids.size(1), max_length)
        padded_input_ids[:, :seq_length] = decoder_input_ids[:, :seq_length]

        start_time = time.time()
        with torch.no_grad():
                outputs = self.model(pixel_values, padded_input_ids)
                
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'last_hidden_state' in outputs:
                        logits = outputs['last_hidden_state']
                    else:
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                logits = value
                                break
                        else:
                            raise ValueError(f"Не удалось найти тензор в выходных данных модели: {outputs.keys()}")
                else:
                    logits = outputs
                
                generated_sequence = decoder_input_ids[0].cpu().tolist()
                for i in range(seq_length, max_length):
                    pos_logits = logits[0, i-1, :]
                    next_token_id = torch.argmax(pos_logits).item()
                    generated_sequence.append(next_token_id)
                    if next_token_id == self.eos_token_id:
                        break
        
        inference_time = time.time() - start_time
        logger.debug(f"Время инференса: {inference_time:.4f} с")
        decoded_output = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        result = None
        if return_json:
            try:
                result = TextCleanup.extract_fields_from_donut_output(decoded_output)
            except Exception as e:
                logger.warning(f"Ошибка при преобразовании вывода в JSON: {e}")
                result = {"text_sequence": TextCleanup.cleanup_donut_output(decoded_output)}
        else:
            result = TextCleanup.cleanup_donut_output(decoded_output)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                if isinstance(result, dict):
                    json.dump(result, f, ensure_ascii=False, indent=2)
                else:
                    f.write(result)
            
            logger.debug(f"Результат сохранен в {save_path}")
        
        return result
    
    def process_batch(
        self, 
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        batch_size: int = 1,
        max_length: int = 64,
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        return_json: bool = True
    ) -> List[Dict[str, Any]]:
 
        results = []

        if save_results and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Обработка пакетов"):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for image_path in batch_paths:
                try:
                    save_path = None
                    if save_results and output_dir:
                        image_name = Path(image_path).stem
                        save_path = output_dir / f"{image_name}_result.json"
                    
                    result = self.process_image(
                        image_path, 
                        prompt=prompt, 
                        max_length=max_length,
                        return_json=return_json,
                        save_path=save_path
                    )
                    
                    batch_results.append({
                        "image_path": str(image_path),
                        "result": result
                    })
                            
                except Exception as e:
                    logger.error(f"Ошибка при обработке {image_path}: {e}")
                    batch_results.append({
                        "image_path": str(image_path),
                        "error": str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def visualize_prediction(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        return_json: bool = True
    ) -> None:

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
        else:
            pil_image = image

        if save_path is None and output_dir is not None and isinstance(image, (str, Path)):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = Path(image).stem
            save_path = output_dir / f"{image_name}_visualized.png"

        result = self.process_image(pil_image, prompt=prompt, return_json=return_json)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        ax.imshow(pil_image)
        ax.axis('off')

        if isinstance(result, str):
            text_result = result
        elif isinstance(result, dict):
            text_result = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            text_result = str(result)

        if isinstance(result, dict):
            clean_result = text_result
        else:
            clean_result = TextCleanup.cleanup_donut_output(text_result)
        
        plt.figtext(0.5, 0.01, clean_result, wrap=True, horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            
            # Сохраняем также JSON результат
            if output_dir is not None:
                json_path = output_dir / f"{save_path.stem.replace('_visualized', '')}_result.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    if isinstance(result, dict):
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(result)
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_on_dataset(
        self,
        dataset_path: Union[str, Path],
        ground_truth_file: Optional[Union[str, Path]] = None,
        prompt: Optional[str] = None,
        batch_size: int = 1,
        max_length: int = 64,
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        return_json: bool = True
    ) -> Dict[str, Any]:

        dataset_path = Path(dataset_path)
        ground_truth = {}
        if ground_truth_file is not None:
            ground_truth_file = Path(ground_truth_file)
            if ground_truth_file.exists():
                with open(ground_truth_file, "r", encoding="utf-8") as f:
                    ground_truth = json.load(f)
            else:
                logger.warning(f"Файл с эталонными данными не найден: {ground_truth_file}")

        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(list(dataset_path.glob(ext)))
    
        results = self.process_batch(
            image_paths,
            prompt=prompt,
            batch_size=batch_size,
            max_length=max_length,
            save_results=save_results,
            output_dir=output_dir,
            return_json=return_json
        )

        metrics = {
            "total_images": len(image_paths),
            "processed_images": len(results),
            "errors": sum(1 for r in results if "error" in r),
            "edit_distances": []
        }

        if ground_truth:
            for result in results:
                if "error" in result:
                    continue
                
                image_path = result["image_path"]
                image_name = Path(image_path).stem
                
                if image_name in ground_truth:
                    prediction = json.dumps(result["result"], ensure_ascii=False)
                    reference = json.dumps(ground_truth[image_name], ensure_ascii=False)
                    
                    edit_distance = EditDistanceMetric.calculate(prediction, reference)
                    metrics["edit_distances"].append({
                        "image_name": image_name,
                        "edit_distance": edit_distance
                    })

            if metrics["edit_distances"]:
                avg_edit_distance = sum(item["edit_distance"] for item in metrics["edit_distances"]) / len(metrics["edit_distances"])
                metrics["avg_edit_distance"] = avg_edit_distance
                
                logger.info(f"Среднее расстояние редактирования: {avg_edit_distance:.4f}")
        
        if save_results and output_dir:
            metrics_path = Path(output_dir) / "evaluation_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Метрики сохранены в {metrics_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Donut TensorRT Inference Engine")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Путь к модели TensorRT (.pt файл)")
    parser.add_argument("--processor_path", type=str, default=None,
                        help="Путь к оригинальной модели Donut или директории с процессором")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Путь к изображению для обработки")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Путь к директории с изображениями для пакетной обработки")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Путь к файлу с эталонными данными для оценки")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Директория для сохранения результатов")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Максимальная длина генерируемой последовательности")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Количество лучей для поиска по лучам")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Размер пакета для пакетной обработки")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Промпт для модели (инструкция, которая будет вставлена)")
    parser.add_argument("--visualize", action="store_true",
                        help="Визуализировать результаты")
    parser.add_argument("--save_results", action="store_true",
                        help="Сохранять результаты в файлы")
    parser.add_argument("--no_json", action="store_true",
                        help="Не преобразовывать вывод в JSON")
    parser.add_argument("--task_start_token", type=str, default="<s_500k>",
                        help="Токен начала задачи")
    parser.add_argument("--prompt_end_token", type=str, default="<s_prompt>",
                        help="Токен конца промпта")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Максимальное количество изображений для обработки (ограничивает выборку)")
    
    args = parser.parse_args()

    if not args.output_dir and args.save_results:
        args.output_dir = "./output_trt"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    engine = TRTInferenceEngine(
        model_path=args.model_path,
        processor_path=args.processor_path,
        device=args.device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        task_start_token=args.task_start_token,
        prompt_end_token=args.prompt_end_token
    )
    
    return_json = not args.no_json

    if args.image_path:
        if args.visualize:
            engine.visualize_prediction(
                image=args.image_path,
                prompt=args.prompt,
                output_dir=output_dir if args.save_results else None,
                return_json=return_json
            )
        else:
            save_path = None
            if args.save_results:
                save_path = output_dir / f"{Path(args.image_path).stem}_result.json"
            
            result = engine.process_image(
                args.image_path, 
                prompt=args.prompt,
                max_length=args.max_length,
                return_json=return_json,
                save_path=save_path
            )
            
            print("\nРезультат обработки:")
            if isinstance(result, dict):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result)

    if args.dataset_path:
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(list(Path(args.dataset_path).glob(ext)))
        
        if args.max_images is not None and args.max_images > 0:
            logger.info(f"Ограничиваем выборку до {args.max_images} изображений (из {len(image_paths)} найденных)")
            random.shuffle(image_paths)
            image_paths = image_paths[:args.max_images]
        
        logger.info(f"Будет обработано {len(image_paths)} изображений")
            
        if args.visualize:
            for image_path in tqdm(image_paths, desc="Визуализация изображений"):
                try:
                    save_path = output_dir / f"{Path(image_path).stem}_visualized.png" if args.save_results else None
                    engine.visualize_prediction(
                        image=image_path,
                        prompt=args.prompt,
                        save_path=save_path,
                        output_dir=output_dir if args.save_results else None,
                        return_json=return_json
                    )
                except Exception as e:
                    logger.error(f"Ошибка при визуализации {image_path}: {e}")
            
            print(f"\nВизуализировано {len(image_paths)} изображений")
            if args.save_results:
                logger.info(f"Визуализации сохранены в {output_dir}")
        
        if args.ground_truth:
            metrics = engine.evaluate_on_dataset(
                dataset_path=args.dataset_path,
                ground_truth_file=args.ground_truth,
                prompt=args.prompt,
                batch_size=args.batch_size,
                max_length=args.max_length,
                save_results=args.save_results,
                output_dir=output_dir,
                return_json=return_json
            )
            
            print("\nРезультаты оценки:")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            results = engine.process_batch(
                image_paths=image_paths,
                prompt=args.prompt,
                batch_size=args.batch_size,
                max_length=args.max_length,
                save_results=args.save_results,
                output_dir=output_dir,
                return_json=return_json
            )
            
            print(f"\nОбработано {len(results)} изображений")
            if args.save_results:
                logger.info(f"Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    main()