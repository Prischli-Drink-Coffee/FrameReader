import os
import sys
import json
import logging
import argparse
from pathlib import Path
import time
import re
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from tqdm.auto import tqdm

# Импортируем необходимые модули из Ultralytics
from ultralytics import YOLO
import torch.backends.cudnn as cudnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOInference:
    """Класс для инференса и оценки моделей YOLO."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        save_visualizations: bool = True,
        batch_size: int = 1,
        image_size: Optional[int] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        name: str = "pytorch_model"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_visualizations = save_visualizations
        self.batch_size = batch_size
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.name = name
        
        # Загружаем модель YOLO
        logger.info(f"Загрузка модели YOLO из {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.model_type = "PyTorch" if str(model_path).endswith('.pt') else "TensorRT"
            logger.info(f"Модель успешно загружена: {self.model_type}")
            
            # Если задан image_size, устанавливаем его
            if image_size is not None:
                logger.info(f"Установка размера изображения: {image_size}")
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
        
        # Настраиваем CUDA для оптимальной производительности
        if self.device.startswith('cuda'):
            cudnn.benchmark = True
            cudnn.deterministic = False
        
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        logger.info(f"Инициализирован модуль инференса YOLO '{name}', устройство: {self.device}")
    
    def warmup(self, num_iterations: int = 100) -> None:
        """Разогрев модели для стабилизации измерений времени."""
        logger.info(f"Разогрев модели {self.name} ({num_iterations} итераций)...")
        
        try:
            # Определяем размер батча для разогрева, который может отличаться для TensorRT
            batch_size = 1
            if self.model_type == "TensorRT":
                # Проверяем настройки модели для TensorRT
                if hasattr(self.model.model, "batch_size"):
                    batch_size = self.model.model.batch_size
                elif hasattr(self.model.model, "max_batch_size"):
                    batch_size = self.model.model.max_batch_size
                else:
                    # Для TensorRT по умолчанию используем батч размером 16, если не указано иное
                    batch_size = 16
            
            logger.info(f"Используется размер батча для разогрева: {batch_size}")
            
            # Создаем случайное изображение нужного размера для разогрева
            if batch_size > 1:
                # Для батчей > 1 создаем соответствующий размер тензора
                dummy_input = torch.rand(batch_size, 3, 640, 640).to(self.device)
                
                # Для TensorRT может потребоваться преобразование тензора в список изображений
                if self.model_type == "TensorRT":
                    dummy_images = []
                    for i in range(batch_size):
                        img = dummy_input[i].permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        dummy_images.append(Image.fromarray(img))
                    
                    for _ in range(num_iterations):
                        _ = self.model.predict(
                            source=dummy_images,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
                else:
                    # Для PyTorch моделей используем прямой вызов с тензором
                    for _ in range(num_iterations):
                        _ = self.model.predict(
                            source=dummy_input,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
            else:
                # Обычный случай для батча размером 1
                dummy_input = torch.rand(3, 640, 640).to(self.device)
                if isinstance(dummy_input, torch.Tensor):
                    dummy_input = dummy_input.permute(1, 2, 0).cpu().numpy()
                    dummy_input = (dummy_input * 255).astype(np.uint8)
                    dummy_input = Image.fromarray(dummy_input)
                
                for _ in range(num_iterations):
                    _ = self.model.predict(
                        source=dummy_input,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )
        
        except Exception as e:
            logger.warning(f"Ошибка при разогреве модели: {e}")
            logger.warning("Продолжаем без разогрева. Первые измерения времени могут быть неточными.")
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[Any]:
        """Генерирует предсказания для пакета изображений."""
        results = self.model.predict(
            source=images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            batch=self.batch_size,
            verbose=False
        )
        return results
    
    def process_sample(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Обрабатывает одно изображение и возвращает результаты."""
        try:
            # Засекаем время
            start_time = time.time()
            
            # Проверяем, является ли модель TensorRT и требует специальной обработки
            if self.model_type == "TensorRT":
                # Определяем требуемый размер батча
                batch_size = 1
                if hasattr(self.model.model, "batch_size"):
                    batch_size = self.model.model.batch_size
                elif hasattr(self.model.model, "max_batch_size"):
                    batch_size = self.model.model.max_batch_size
                else:
                    # По умолчанию для TensorRT используем 16
                    batch_size = 16
                
                if batch_size > 1:
                    # Загружаем исходное изображение
                    original_image = Image.open(image_path).convert("RGB")
                    
                    # Дублируем изображение batch_size раз
                    dummy_images = [original_image] * batch_size
                    
                    # Запускаем предсказание на батче
                    results = self.model.predict(
                        source=dummy_images,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )[0]  # Берем первый результат (для первого изображения)
                else:
                    # Стандартный случай для батча 1
                    results = self.model.predict(
                        source=image_path,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )[0]
            else:
                # Стандартный путь для не-TensorRT моделей
                results = self.model.predict(
                    source=image_path,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )[0]  # Берем первый результат
            
            inference_time = time.time() - start_time
            
            # Извлекаем данные о детекции
            num_detections = len(results.boxes)
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            
            # Создаем структуру результата
            result = {
                "image_path": str(image_path),
                "inference_time": inference_time,
                "num_detections": num_detections,
                "detections": [
                    {
                        "class_id": int(classes[i]),
                        "class_name": results.names[int(classes[i])],
                        "confidence": float(confidences[i]),
                        "box": boxes[i].tolist()
                    }
                    for i in range(num_detections)
                ]
            }
            
            # Сохраняем визуализацию если нужно
            if self.save_visualizations:
                save_path = self.output_dir / "samples"
                save_path.mkdir(exist_ok=True, parents=True)
                
                # Сохраняем изображение с результатами
                result_image = results.plot()
                result_image = Image.fromarray(result_image)
                image_name = Path(image_path).stem
                result_image.save(save_path / f"{image_name}_result.jpg")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {image_path}: {e}")
            return {"image_path": str(image_path), "error": str(e)}
    
    def evaluate_dataset(
        self, 
        data_path: Union[str, Path],
        split: str = "val",
        save_individual_results: bool = True
    ) -> Dict[str, Any]:
        """Оценивает модель на датасете и возвращает метрики."""
        data_path = Path(data_path)
        
        start_time_total = time.time()
        
        # Определяем путь к данным
        if not data_path.exists():
            raise FileNotFoundError(f"Путь к данным не существует: {data_path}")
        
        # Запускаем валидацию на указанном датасете
        logger.info(f"Оценка модели на датасете: {data_path}")
        try:
            # Прогон валидации
            validation_results = self.model.val(
                data=data_path,
                batch=self.batch_size,
                split=split,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=300,
                plots=self.save_visualizations
            )
            
            metrics = validation_results.results_dict
            
            # Находим путь к тестовым изображениям
            par_dir = data_path.parent
            images_dir = par_dir / split / "images"

            # Собираем пути к изображениям
            image_paths = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_paths.extend(list(images_dir.glob(f"*{ext}")))
            
            if len(image_paths) == 0:
                logger.warning(f"Не найдены изображения в {images_dir}")
                inference_times = []
            else:
                # Определяем требуемый размер батча для модели
                batch_size = self.batch_size
                if self.model_type == "TensorRT":
                    if hasattr(self.model.model, "batch_size"):
                        batch_size = self.model.model.batch_size
                    elif hasattr(self.model.model, "max_batch_size"):
                        batch_size = self.model.model.max_batch_size
                    else:
                        # По умолчанию для TensorRT используем 16
                        batch_size = 16
                
                logger.info(f"Размер батча для измерения производительности: {batch_size}")
                
                # Разогреваем модель перед измерением времени
                self.warmup()
                
                # Измеряем время инференса батчами
                inference_times = []
                batch_results = []
                
                # Создаем батчи изображений
                num_images = len(image_paths)
                num_batches = (num_images + batch_size - 1) // batch_size  # Округление вверх
                
                for i in tqdm(range(num_batches), desc=f"Измерение времени инференса батчами ({split})"):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_images)
                    
                    current_batch = image_paths[start_idx:end_idx]
                    current_batch_size = len(current_batch)
                    
                    # Если размер батча меньше требуемого (последний батч), дополняем его
                    if current_batch_size < batch_size and self.model_type == "TensorRT":
                        # Дополняем батч копиями последнего изображения до нужного размера
                        padding = [current_batch[-1]] * (batch_size - current_batch_size)
                        padded_batch = current_batch + padding
                        
                        # Засекаем время
                        start_time = time.time()
                        
                        # Загружаем изображения как PIL объекты
                        batch_images = [Image.open(img_path).convert("RGB") for img_path in padded_batch]
                        
                        # Запускаем предсказание
                        results = self.model.predict(
                            source=batch_images,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
                        
                        # Берем только результаты для реальных изображений из батча
                        results = results[:current_batch_size]
                    else:
                        # Для полного батча или не-TensorRT моделей
                        # Загружаем изображения как PIL объекты
                        batch_images = [Image.open(img_path).convert("RGB") for img_path in current_batch]
                        
                        # Засекаем время
                        start_time = time.time()
                        
                        # Запускаем предсказание
                        results = self.model.predict(
                            source=batch_images,
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
                    
                    # Рассчитываем время на весь батч
                    inference_time = time.time() - start_time
                    
                    # Рассчитываем время на одно изображение
                    time_per_image = inference_time / current_batch_size
                    
                    # Добавляем время для каждого изображения в батче
                    inference_times.extend([time_per_image] * current_batch_size)
                    
                    # Если нужно сохранять визуализации, обрабатываем каждый результат
                    if self.save_visualizations:
                        for j, (img_path, result) in enumerate(zip(current_batch, results)):
                            save_path = self.output_dir / "samples"
                            save_path.mkdir(exist_ok=True, parents=True)
                            
                            # Сохраняем изображение с результатами
                            result_image = result.plot()
                            result_image = Image.fromarray(result_image)
                            image_name = Path(img_path).stem
                            result_image.save(save_path / f"{image_name}_result.jpg")
                    
                    # Сохраняем детализированные результаты для каждого изображения в батче
                    for j, (img_path, result) in enumerate(zip(current_batch, results)):
                        num_detections = len(result.boxes)
                        classes = result.boxes.cls.cpu().numpy() if num_detections > 0 else []
                        confidences = result.boxes.conf.cpu().numpy() if num_detections > 0 else []
                        boxes = result.boxes.xyxy.cpu().numpy() if num_detections > 0 else []
                        
                        batch_result = {
                            "image_path": str(img_path),
                            "inference_time": time_per_image,
                            "num_detections": num_detections,
                            "detections": [
                                {
                                    "class_id": int(classes[k]),
                                    "class_name": result.names[int(classes[k])],
                                    "confidence": float(confidences[k]),
                                    "box": boxes[k].tolist()
                                }
                                for k in range(num_detections)
                            ]
                        }
                        batch_results.append(batch_result)
            
            # Расчет статистики по временам инференса
            if inference_times:
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                median_time = np.median(inference_times)
                fps = 1.0 / avg_time
            else:
                avg_time = 0
                std_time = 0
                median_time = 0
                fps = 0
            
            # Получаем количество изображений из атрибутов validation_results
            # Атрибут nt_per_image содержит число целей на изображение
            num_images = len(validation_results.nt_per_image) if hasattr(validation_results, 'nt_per_image') else 0
            
            # Получаем количество классов
            num_classes = len(validation_results.names) if hasattr(validation_results, 'names') else 0
            
            # Собираем все метрики
            all_metrics = {
                # Метрики из валидации
                "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
                "precision": metrics.get("metrics/precision(B)", 0.0),
                "recall": metrics.get("metrics/recall(B)", 0.0),
                "f1": metrics.get("metrics/F1(B)", 0.0),
                
                # Метрики времени инференса
                "avg_inference_time": avg_time,
                "std_inference_time": std_time,
                "median_inference_time": median_time,
                "fps": fps,
                "num_samples_inference": len(inference_times),
                
                # Общая информация
                "model_name": self.name,
                "model_type": self.model_type,
                "num_images": num_images,
                "num_classes": num_classes,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "total_evaluation_time": time.time() - start_time_total,
                "batch_size": batch_size
            }
            
            # Сохраняем метрики
            metrics_file = self.output_dir / f"{split}_metrics_{self.name}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            
            # Сохраняем детализированные результаты по изображениям
            if save_individual_results and batch_results:
                detailed_results_file = self.output_dir / f"{split}_detailed_results_{self.name}.json"
                with open(detailed_results_file, "w", encoding="utf-8") as f:
                    json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            # Сохраняем визуализацию метрик
            if self.save_visualizations:
                self._save_metrics_visualization(all_metrics, inference_times, split)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Ошибка при оценке датасета: {e}", exc_info=True)
            raise
    
    def _save_metrics_visualization(self, metrics: Dict[str, Any], inference_times: List[float], split: str) -> None:
        """Сохраняет визуализацию метрик."""
        if not self.save_visualizations:
            return
            
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 10)
        
        # График времени инференса
        if inference_times:
            plt.figure(figsize=(10, 6))
            sns.histplot(inference_times, kde=True, color="blue")
            plt.axvline(x=metrics["avg_inference_time"], color='r', linestyle='--', 
                        label=f'Среднее: {metrics["avg_inference_time"]:.4f} с')
            plt.axvline(x=metrics["median_inference_time"], color='g', linestyle='--', 
                        label=f'Медиана: {metrics["median_inference_time"]:.4f} с')
            plt.title(f"Распределение времени инференса ({metrics['model_name']})", fontsize=14)
            plt.xlabel("Время (сек)", fontsize=12)
            plt.ylabel("Частота", fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            inference_time_file = plots_dir / f"{split}_inference_time_{self.name}.png"
            plt.savefig(inference_time_file, dpi=100, bbox_inches='tight')
            plt.close()
        
        # График метрик детекции
        plt.figure(figsize=(12, 8))
        
        metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
        metrics_values = [metrics.get(name, 0.0) for name in metrics_names]
        
        bars = plt.bar(metrics_names, metrics_values, color='skyblue')
        
        # Добавляем значения над столбцами
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, 
                     bar.get_height() + 0.01, 
                     f'{value:.4f}', 
                     ha='center', va='bottom', fontsize=11)
        
        plt.title(f"Метрики детекции ({metrics['model_name']})", fontsize=14)
        plt.ylabel("Значение", fontsize=12)
        plt.ylim(0, 1.1)  # Устанавливаем предел по y от 0 до 1.1
        plt.tight_layout()
        
        detection_metrics_file = plots_dir / f"{split}_detection_metrics_{self.name}.png"
        plt.savefig(detection_metrics_file, dpi=100, bbox_inches='tight')
        plt.close()


def create_comparison_table(results_1: Dict[str, Any], results_2: Dict[str, Any], output_dir: Path):
    """Создает таблицу сравнения двух моделей и сохраняет как изображение и JSON."""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Данные для таблицы
    model_1_name = results_1.get('model_name', 'Model 1')
    model_2_name = results_2.get('model_name', 'Model 2')
    
    data = [
        ['Метрика', model_1_name, model_2_name, 'Разница'],
        ['mAP50', f"{results_1['mAP50']:.4f}", f"{results_2['mAP50']:.4f}", f"{results_2['mAP50'] - results_1['mAP50']:.4f}"],
        ['mAP50-95', f"{results_1['mAP50-95']:.4f}", f"{results_2['mAP50-95']:.4f}", f"{results_2['mAP50-95'] - results_1['mAP50-95']:.4f}"],
        ['Precision', f"{results_1['precision']:.4f}", f"{results_2['precision']:.4f}", f"{results_2['precision'] - results_1['precision']:.4f}"],
        ['Recall', f"{results_1['recall']:.4f}", f"{results_2['recall']:.4f}", f"{results_2['recall'] - results_1['recall']:.4f}"],
        ['Avg Time (ms)', f"{results_1['avg_inference_time']*1000:.2f}", f"{results_2['avg_inference_time']*1000:.2f}", f"{(results_2['avg_inference_time'] - results_1['avg_inference_time'])*1000:.2f}"],
        ['Total Time (s)', f"{results_1['total_evaluation_time']:.2f}", f"{results_2['total_evaluation_time']:.2f}", f"{results_2['total_evaluation_time'] - results_1['total_evaluation_time']:.2f}"],
        ['FPS', f"{results_1['fps']:.2f}", f"{results_2['fps']:.2f}", f"{results_2['fps'] - results_1['fps']:.2f}"],
        ['Samples', str(results_1['num_samples_inference']), str(results_2['num_samples_inference']), '-']
    ]
    
    # Создаем структурированные данные для JSON
    comparison_data = {
        "metadata": {
            "comparison_timestamp": datetime.now().isoformat(),
            "model_1": {
                "name": model_1_name,
                "type": results_1.get('model_type', 'Unknown'),
                "samples": results_1['num_samples_inference']
            },
            "model_2": {
                "name": model_2_name,
                "type": results_2.get('model_type', 'Unknown'),
                "samples": results_2['num_samples_inference']
            }
        },
        "detection_metrics": {
            "mAP50": {
                "model_1": results_1['mAP50'],
                "model_2": results_2['mAP50'],
                "difference": results_2['mAP50'] - results_1['mAP50'],
                "improvement": "model_2" if results_2['mAP50'] > results_1['mAP50'] else "model_1",
                "improvement_percentage": abs((results_2['mAP50'] - results_1['mAP50']) / max(0.001, results_1['mAP50']) * 100)
            },
            "mAP50-95": {
                "model_1": results_1['mAP50-95'],
                "model_2": results_2['mAP50-95'],
                "difference": results_2['mAP50-95'] - results_1['mAP50-95'],
                "improvement": "model_2" if results_2['mAP50-95'] > results_1['mAP50-95'] else "model_1",
                "improvement_percentage": abs((results_2['mAP50-95'] - results_1['mAP50-95']) / max(0.001, results_1['mAP50-95']) * 100)
            },
            "precision": {
                "model_1": results_1['precision'],
                "model_2": results_2['precision'],
                "difference": results_2['precision'] - results_1['precision'],
                "improvement": "model_2" if results_2['precision'] > results_1['precision'] else "model_1",
                "improvement_percentage": abs((results_2['precision'] - results_1['precision']) / max(0.001, results_1['precision']) * 100)
            },
            "recall": {
                "model_1": results_1['recall'],
                "model_2": results_2['recall'],
                "difference": results_2['recall'] - results_1['recall'],
                "improvement": "model_2" if results_2['recall'] > results_1['recall'] else "model_1",
                "improvement_percentage": abs((results_2['recall'] - results_1['recall']) / max(0.001, results_1['recall']) * 100)
            }
        },
        "performance_metrics": {
            "avg_time_ms": {
                "model_1": results_1['avg_inference_time'] * 1000,
                "model_2": results_2['avg_inference_time'] * 1000,
                "difference": (results_2['avg_inference_time'] - results_1['avg_inference_time']) * 1000,
                "improvement": "model_1" if results_1['avg_inference_time'] < results_2['avg_inference_time'] else "model_2",
                "speedup_factor": results_1['avg_inference_time'] / max(0.001, results_2['avg_inference_time'])
            },
            "total_time_s": {
                "model_1": results_1['total_evaluation_time'],
                "model_2": results_2['total_evaluation_time'],
                "difference": results_2['total_evaluation_time'] - results_1['total_evaluation_time'],
                "improvement": "model_1" if results_1['total_evaluation_time'] < results_2['total_evaluation_time'] else "model_2"
            },
            "fps": {
                "model_1": results_1['fps'],
                "model_2": results_2['fps'],
                "difference": results_2['fps'] - results_1['fps'],
                "improvement": "model_2" if results_2['fps'] > results_1['fps'] else "model_1",
                "speedup_factor": results_2['fps'] / max(0.001, results_1['fps'])
            }
        },
        "summary": {
            "better_accuracy_model": None,
            "faster_model": None,
            "overall_winner": None
        }
    }
    
    # Определяем лучшую модель по точности (средний балл по основным метрикам)
    # Исключаем F1 из расчетов
    model_1_accuracy_score = (
        results_1['mAP50'] +
        results_1['mAP50-95'] +
        results_1['precision'] +
        results_1['recall']
    ) / 4
    
    model_2_accuracy_score = (
        results_2['mAP50'] +
        results_2['mAP50-95'] +
        results_2['precision'] +
        results_2['recall']
    ) / 4
    
    comparison_data["summary"]["better_accuracy_model"] = model_1_name if model_1_accuracy_score > model_2_accuracy_score else model_2_name
    
    # Определяем более быструю модель на основе времени инференса, избегаем деления на ноль
    if results_1['fps'] > 0 or results_2['fps'] > 0:
        comparison_data["summary"]["faster_model"] = model_1_name if results_1['fps'] > results_2['fps'] else model_2_name
    else:
        # Если FPS у обеих моделей равен 0, сравниваем по времени инференса
        comparison_data["summary"]["faster_model"] = model_1_name if results_1['avg_inference_time'] < results_2['avg_inference_time'] else model_2_name
    
    # Определяем общего победителя (с весами: 70% точность, 30% скорость)
    # Используем нормализованные значения для скорости, чтобы избежать деления на ноль
    max_fps = max(0.001, max(results_1['fps'], results_2['fps']))
    
    model_1_speed_score = results_1['fps'] / max_fps if max_fps > 0 else 0.5
    model_2_speed_score = results_2['fps'] / max_fps if max_fps > 0 else 0.5
    
    model_1_overall_score = model_1_accuracy_score * 0.7 + model_1_speed_score * 0.3
    model_2_overall_score = model_2_accuracy_score * 0.7 + model_2_speed_score * 0.3
    
    comparison_data["summary"]["overall_winner"] = model_1_name if model_1_overall_score > model_2_overall_score else model_2_name
    
    # Сохраняем JSON файл
    comparison_json_file = output_dir / "model_comparison.json"
    with open(comparison_json_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Данные сравнения сохранены в {comparison_json_file}")
    
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)
    
    # Стилизация
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', weight='bold')
        elif col == 3 and row > 0:  # Колонка разницы
            if data[row][3] == '-':
                continue
            
            diff_value = float(data[row][3])
            metric_name = data[row][0]
            
            # Для разных метрик разная логика "лучше"
            if metric_name in ['Avg Time (ms)', 'Total Time (s)']:
                # Для этих метрик меньше = лучше
                if diff_value < 0:
                    cell.set_facecolor('#C6EFCE')  # Зеленый для улучшения
                elif diff_value > 0:
                    cell.set_facecolor('#FFC7CE')  # Красный для ухудшения
            else:
                # Для остальных метрик больше = лучше
                if diff_value > 0:
                    cell.set_facecolor('#C6EFCE')  # Зеленый для улучшения
                elif diff_value < 0:
                    cell.set_facecolor('#FFC7CE')  # Красный для ухудшения
        elif row % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    
    comparison_file = output_dir / "model_comparison.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Таблица сравнения сохранена в {comparison_file}")
    
    return comparison_file, comparison_json_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Оценка и сравнение моделей YOLO")
    
    # Параметры модели
    model_group = parser.add_argument_group("Параметры модели")
    model_group.add_argument("--model_path", type=str, required=True,
                        help="Путь к основной модели YOLO (.pt)")
    model_group.add_argument("--image_size", type=int, default=640,
                        help="Размер изображения для модели")
    model_group.add_argument("--conf_threshold", type=float, default=0.2,
                        help="Порог уверенности для детекций")
    model_group.add_argument("--iou_threshold", type=float, default=0.2,
                        help="Порог IoU для NMS")
    
    # Параметры вывода
    output_group = parser.add_argument_group("Параметры вывода")
    output_group.add_argument("--output_dir", type=str, default="./output/detection_comparison",
                        help="Директория для сохранения результатов")
    output_group.add_argument("--save_visualizations", action="store_true", default=True,
                        help="Сохранять визуализации результатов")
    output_group.add_argument("--batch_size", type=int, default=16,
                        help="Размер пакета для инференса")
    
    # Режимы работы
    mode_group = parser.add_argument_group("Режимы работы")
    mode_group.add_argument("--mode", type=str, choices=["single_image", "dataset", "both", "compare"], default="dataset",
                        help="Режим работы: одиночное изображение, датасет, оба или сравнение моделей")
    mode_group.add_argument("--image_path", type=str, default=None,
                        help="Путь к изображению для одиночной оценки")
    mode_group.add_argument("--data_path", type=str, default='../dataset/detection/yolo/data.yaml',
                        help="Путь к файлу данных YAML для оценки")
    mode_group.add_argument("--split", type=str, choices=["train", "val", "test"], default="test",
                        help="Раздел данных для оценки")
    
    # Параметры сравнения моделей
    compare_group = parser.add_argument_group("Параметры сравнения моделей")
    compare_group.add_argument("--model_path_2", type=str, default=None,
                        help="Путь ко второй модели для сравнения (TensorRT .engine)")
    compare_group.add_argument("--model_name_1", type=str, default="PyTorch Model",
                        help="Название первой модели для отображения в сравнении")
    compare_group.add_argument("--model_name_2", type=str, default="TensorRT Model",
                        help="Название второй модели для отображения в сравнении")
    
    # Параметры вычислений
    compute_group = parser.add_argument_group("Параметры вычислений")
    compute_group.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    compute_group.add_argument("--half", action="store_true", default=False,
                        help="Использовать вычисления с половинной точностью (FP16)")
    
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if args.mode == "single_image" and args.image_path is None:
        parser.error("Для режима 'single_image' требуется указать --image_path")
    
    if args.mode == "compare" and args.model_path_2 is None:
        parser.error("Для режима 'compare' требуется указать --model_path_2")
    
    return args


def main():
    args = parse_arguments()

    # Создаем директорию для вывода
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем аргументы
    with open(output_dir / "inference_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Создаем экземпляр для первой модели
    yolo_inference = YOLOInference(
        model_path=args.model_path,
        output_dir=output_dir / "model_1",
        device=args.device,
        save_visualizations=args.save_visualizations,
        batch_size=args.batch_size,
        image_size=args.image_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        name=args.model_name_1
    )

    if args.mode in ["single_image", "both"] and args.image_path is not None:
        try:
            result = yolo_inference.process_sample(args.image_path)
            
            result_file = output_dir / "single_image_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Результат для {args.image_path}:")
            logger.info(f"Время инференса: {result['inference_time']:.4f} сек")
            logger.info(f"Обнаружено объектов: {result['num_detections']}")
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {args.image_path}: {e}")
    
    if args.mode in ["dataset", "both", "compare"] and args.data_path is not None:
        try:
            metrics_1 = yolo_inference.evaluate_dataset(
                data_path=args.data_path,
                split=args.split,
                save_individual_results=args.save_visualizations
            )

            logger.info(f"Метрики для {args.model_name_1}:")
            logger.info(f"mAP50: {metrics_1['mAP50']:.4f}, mAP50-95: {metrics_1['mAP50-95']:.4f}")
            logger.info(f"Precision: {metrics_1['precision']:.4f}, Recall: {metrics_1['recall']:.4f}, F1: {metrics_1['f1']:.4f}")
            logger.info(f"Среднее время на изображение: {metrics_1['avg_inference_time']*1000:.2f} мс, FPS: {metrics_1['fps']:.2f}")
            
            # Если режим сравнения, загружаем вторую модель
            if args.mode == "compare" and args.model_path_2:
                yolo_inference_2 = YOLOInference(
                    model_path=args.model_path_2,
                    output_dir=output_dir / "model_2",
                    device=args.device,
                    save_visualizations=args.save_visualizations,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    name=args.model_name_2
                )
                
                metrics_2 = yolo_inference_2.evaluate_dataset(
                    data_path=args.data_path,
                    split=args.split,
                    save_individual_results=args.save_visualizations
                )

                logger.info(f"Метрики для {args.model_name_2}:")
                logger.info(f"mAP50: {metrics_2['mAP50']:.4f}, mAP50-95: {metrics_2['mAP50-95']:.4f}")
                logger.info(f"Precision: {metrics_2['precision']:.4f}, Recall: {metrics_2['recall']:.4f}, F1: {metrics_2['f1']:.4f}")
                logger.info(f"Среднее время на изображение: {metrics_2['avg_inference_time']*1000:.2f} мс, FPS: {metrics_2['fps']:.2f}")
                
                # Создаем сравнительную таблицу
                comparison_file, json_file = create_comparison_table(metrics_1, metrics_2, output_dir)
                logger.info(f"Сравнение моделей сохранено в {comparison_file} и {json_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при оценке датасета: {e}", exc_info=True)
            return 1

    logger.info(f"Результаты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())