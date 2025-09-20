import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Tuple, List, Optional
import random
import math

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from transformers import DonutProcessor
from PIL import Image, ImageFont, ImageDraw

from dataset import DonutDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("visualize")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Визуализация семплов из набора данных Donut",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Путь к директории с данными (содержащей папки train/valid/test)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./visualization.png", 
        help="Путь к выходному файлу визуализации"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "valid", "test"],
        default="train", 
        help="Раздел данных для визуализации"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=16, 
        help="Количество образцов для визуализации"
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        nargs=2, 
        default=[4, 4], 
        help="Размеры сетки [rows, cols]"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        nargs=2, 
        default=[384, 384], 
        help="Размер изображения (высота, ширина) для отображения"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Случайное зерно для выбора образцов"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300, 
        help="DPI для сохранения визуализации"
    )
    parser.add_argument(
        "--font_size", 
        type=int, 
        default=12, 
        help="Размер шрифта для подписей"
    )
    parser.add_argument(
        "--max_text_length", 
        type=int, 
        default=64, 
        help="Максимальная длина текста для отображения"
    )
    parser.add_argument(
        "--with_augmentation", 
        action="store_true", 
        help="Применять ли аугментацию к изображениям"
    )
    parser.add_argument(
        "--shuffle", 
        action="store_true", 
        help="Случайно перемешать семплы перед выбором"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Имя модели из HuggingFace или путь к локальному чекпоинту для инициализации процессора"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Путь к чекпоинту модели для загрузки процессора (имеет приоритет над model_name)"
    )
    parser.add_argument(
        "--task_start_token",
        type=str,
        default="<s>",
        help="Токен начала задания"
    )
    parser.add_argument(
        "--prompt_end_token",
        type=str,
        default="",
        help="Токен конца промпта (если отличается от task_start_token)"
    )
    parser.add_argument(
        "--visualization_backend", 
        type=str, 
        choices=["matplotlib", "pil", "auto"], 
        default="auto",
        help="Бэкенд для визуализации (matplotlib, PIL или автоматический выбор)"
    )
    parser.add_argument(
        "--style", 
        type=str, 
        choices=["default", "dark", "light", "elegant", "technical"],
        default="default", 
        help="Стиль визуализации"
    )
    parser.add_argument(
        "--show_metadata", 
        action="store_true", 
        help="Показывать дополнительные метаданные об образцах"
    )
    parser.add_argument(
        "--text_bg_color",
        type=str,
        default="white",
        help="Цвет фона для текста"
    )
    parser.add_argument(
        "--text_color",
        type=str,
        default="black",
        help="Цвет текста"
    )
    parser.add_argument(
        "--border_width",
        type=int,
        default=2,
        help="Ширина рамки для изображений"
    )
    parser.add_argument(
        "--border_color",
        type=str,
        default="#888888",
        help="Цвет рамки для изображений"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина последовательности для токенизатора"
    )
    
    args = parser.parse_args()
    
    # Убираем обязательность указания модели или чекпоинта
    # if not args.model_name and not args.checkpoint_path:
    #    parser.error("Необходимо указать либо --model_name, либо --checkpoint_path")
        
    if not args.prompt_end_token:
        args.prompt_end_token = args.task_start_token
    
    if args.grid_size[0] * args.grid_size[1] < args.samples:
        logger.warning(
            f"Размер сетки {args.grid_size[0]}x{args.grid_size[1]} меньше, чем запрошенное число образцов {args.samples}. "
            f"Установка samples={args.grid_size[0] * args.grid_size[1]}"
        )
        args.samples = args.grid_size[0] * args.grid_size[1]
    
    return args


def init_processor(model_name_or_path: str) -> DonutProcessor:
    """
    Инициализирует процессор Donut из модели или чекпоинта.
    
    Args:
        model_name_or_path: Имя модели из HuggingFace или путь к локальному чекпоинту
    
    Returns:
        DonutProcessor: Инициализированный процессор
    """
    try:
        if not model_name_or_path:
            raise ValueError("Не указано имя модели или путь к чекпоинту")
        
        logger.info(f"Попытка загрузки процессора из {model_name_or_path}")
        
        # Проверяем, является ли это путем к чекпоинту
        path = Path(model_name_or_path)
        if path.exists() and path.is_dir():
            try:
                from transformers import DonutProcessor
                processor = DonutProcessor.from_pretrained(model_name_or_path)
                logger.info(f"Процессор успешно загружен из чекпоинта {model_name_or_path}")
                return processor
            except Exception as checkpoint_e:
                logger.warning(f"Не удалось загрузить процессор из чекпоинта: {checkpoint_e}")
                # Пробуем загрузить отдельные компоненты из чекпоинта
                try:
                    from transformers import AutoImageProcessor, AutoTokenizer
                    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                    processor = DonutProcessor(image_processor=image_processor, tokenizer=tokenizer)
                    logger.info(f"Процессор собран из отдельных компонентов чекпоинта {model_name_or_path}")
                    return processor
                except Exception as e:
                    logger.warning(f"Не удалось собрать процессор из компонентов: {e}")
                    raise
        
        # Загрузка из HuggingFace Hub
        from transformers import AutoImageProcessor, AutoTokenizer
        image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        processor = DonutProcessor(image_processor=image_processor, tokenizer=tokenizer)
        logger.info(f"Процессор успешно загружен из модели {model_name_or_path}")
        return processor
    except Exception as e:
        logger.error(f"Ошибка при инициализации процессора: {e}")
        logger.info("Создание минимального процессора...")
        
        class MinimalProcessor:
            def __init__(self):
                self.image_processor = None
                
                # Минимальный токенизатор
                class MinimalTokenizer:
                    def __init__(self):
                        self.pad_token_id = 0
                        self.bos_token_id = 1
                        self.eos_token_id = 2
                    
                    def batch_decode(self, *args, **kwargs):
                        return ["[PLACEHOLDER]"]
                    
                    def tokenize(self, text):
                        return [text]
                    
                    def convert_tokens_to_ids(self, token):
                        return 1 if token == "<s>" else 0
                
                self.tokenizer = MinimalTokenizer()
            
            def __call__(self, images, **kwargs):
                return type('obj', (object,), {'pixel_values': torch.zeros(1, 3, 224, 224)})
        
        return MinimalProcessor()


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_style_settings(style: str):
    """
    Получение настроек стиля для визуализации.
    
    Args:
        style: Название стиля ('default', 'dark', 'light', 'elegant', 'technical')
        
    Returns:
        dict: Словарь с настройками стиля
    """
    styles = {
        "default": {
            "bg_color": "white",
            "text_color": "black",
            "title_color": "black",
            "grid_color": "#cccccc",
            "border_color": "#888888",
            "accent_color": "#4472C4",
            "matplotlib_style": "default"
        },
        "dark": {
            "bg_color": "#2d2d2d",
            "text_color": "#e6e6e6",
            "title_color": "#ffffff",
            "grid_color": "#3a3a3a",
            "border_color": "#555555",
            "accent_color": "#5cb2ff",
            "matplotlib_style": "dark_background"
        },
        "light": {
            "bg_color": "#f9f9f9",
            "text_color": "#333333",
            "title_color": "#000000",
            "grid_color": "#e6e6e6",
            "border_color": "#cccccc",
            "accent_color": "#1976D2",
            "matplotlib_style": "seaborn-v0_8-whitegrid"
        },
        "elegant": {
            "bg_color": "#fffff8",
            "text_color": "#222222",
            "title_color": "#000000",
            "grid_color": "#e1e1e1",
            "border_color": "#d4af37",
            "accent_color": "#8B4513",
            "matplotlib_style": "seaborn-v0_8-paper"
        },
        "technical": {
            "bg_color": "white",
            "text_color": "#222222",
            "title_color": "#000066",
            "grid_color": "#cccccc",
            "border_color": "#0066cc",
            "accent_color": "#ff6600",
            "matplotlib_style": "seaborn-v0_8-ticks"
        }
    }
    
    return styles.get(style, styles["default"])


def apply_border_to_image(image: Image.Image, width: int = 2, color: str = "#888888") -> Image.Image:
    """
    Применяет рамку к изображению.
    
    Args:
        image: Исходное изображение
        width: Ширина рамки
        color: Цвет рамки
        
    Returns:
        Image.Image: Изображение с рамкой
    """
    if width <= 0:
        return image
        
    width = max(1, min(width, 10))  # Ограничиваем ширину рамки
    
    img_with_border = Image.new('RGB', (image.width + 2*width, image.height + 2*width), color)
    img_with_border.paste(image, (width, width))
    
    return img_with_border


def extract_metadata_from_sample(sample: dict) -> dict:
    """
    Извлекает метаданные из образца.
    
    Args:
        sample: Словарь с данными образца
        
    Returns:
        dict: Словарь с метаданными
    """
    metadata = {}
    
    if "image_path" in sample:
        metadata["file"] = os.path.basename(sample["image_path"])
    
    if "ground_truth" in sample:
        gt = sample["ground_truth"]
        if "height" in gt and "width" in gt:
            metadata["size"] = f"{gt.get('width', '?')}x{gt.get('height', '?')}"

        if "text_length" in gt:
            metadata["text_length"] = gt.get("text_length", "?")
        elif "gt_parse" in gt and isinstance(gt["gt_parse"], dict) and "text" in gt["gt_parse"]:
            metadata["text_length"] = len(gt["gt_parse"]["text"])
            
        if "source" in gt:
            metadata["source"] = gt["source"]
            
    return metadata


def visualize_samples(
    dataset: DonutDataset,
    output_path: str,
    num_samples: int,
    grid_size: Tuple[int, int],
    dpi: int = 150,
    font_size: int = 12,
    max_text_length: int = 50,
    shuffle: bool = False,
    style: str = "default",
    show_metadata: bool = False,
    text_bg_color: str = "white",
    text_color: str = "black",
    border_width: int = 2,
    border_color: str = "#888888"
) -> None:
    """
    Визуализирует выборку семплов из датасета с использованием matplotlib.
    
    Args:
        dataset: Датасет для визуализации
        output_path: Путь к файлу для сохранения
        num_samples: Количество образцов для визуализации
        grid_size: Размер сетки (строки, столбцы)
        dpi: Разрешение изображения
        font_size: Размер шрифта
        max_text_length: Максимальная длина текста
        shuffle: Флаг перемешивания индексов
        style: Стиль визуализации
        show_metadata: Показывать дополнительные метаданные
        text_bg_color: Цвет фона для текста
        text_color: Цвет текста
        border_width: Ширина рамки
        border_color: Цвет рамки
    """
    if num_samples > len(dataset):
        logger.warning(f"Запрошено {num_samples} образцов, но доступно только {len(dataset)}. "
                      f"Будет использовано {len(dataset)} образцов.")
        num_samples = len(dataset)
    
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    selected_indices = indices[:num_samples]

    rows, cols = grid_size
    
    # Настраиваем стиль визуализации
    style_settings = get_style_settings(style)
    plt.style.use(style_settings["matplotlib_style"])
    
    # Рассчитываем размеры фигуры
    figsize = (cols * 4, rows * (5 if show_metadata else 4))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Создаем сетку с учетом метаданных
    if show_metadata:
        gs = GridSpec(rows * 3, cols, figure=fig, height_ratios=[3, 1, 1] * rows)
    else:
        gs = GridSpec(rows * 2, cols, figure=fig, height_ratios=[3, 1] * rows)
    
    fig.patch.set_facecolor(style_settings["bg_color"])
    
    # Настраиваем шрифты для корректного отображения кириллицы
    font_candidates = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'FreeSans']
    font_prop = None
    
    for font_name in font_candidates:
        try:
            font_prop = FontProperties(family=font_name, size=font_size)
            logger.info(f"Используется шрифт: {font_name}")
            break
        except:
            continue
            
    if font_prop is None:
        font_prop = FontProperties(size=font_size)
        logger.warning("Не удалось установить шрифт с поддержкой кириллицы.")
    
    # Рендеринг образцов
    for i, idx in enumerate(selected_indices):
        if i >= rows * cols:
            break
            
        try:
            image, text, token_sequence = dataset.visualize_sample(idx)
            metadata = {}
            
            if hasattr(dataset, 'samples') and 0 <= idx < len(dataset.samples):
                metadata = extract_metadata_from_sample(dataset.samples[idx])
            
            row, col = divmod(i, cols)
            
            # Рендерим изображение
            if show_metadata:
                ax_img = fig.add_subplot(gs[row * 3, col])
            else:
                ax_img = fig.add_subplot(gs[row * 2, col])
                
            ax_img.imshow(image)
            ax_img.set_title(f"Образец #{idx}", fontproperties=font_prop, color=style_settings["title_color"])
            
            # Добавляем рамку
            for spine in ax_img.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                
            ax_img.axis('on' if border_width > 0 else 'off')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Рендерим текст
            if show_metadata:
                ax_text = fig.add_subplot(gs[row * 3 + 1, col])
            else:
                ax_text = fig.add_subplot(gs[row * 2 + 1, col])
                
            text_truncated = truncate_text(text, max_text_length)
            ax_text.text(0.5, 0.5, text_truncated, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontproperties=font_prop,
                        color=text_color,
                        wrap=True)
            
            # Настройка фона для текстового блока
            ax_text.set_facecolor(text_bg_color)
            ax_text.axis('on')
            ax_text.set_xticks([])
            ax_text.set_yticks([])
            
            # Рендерим метаданные, если нужно
            if show_metadata:
                ax_meta = fig.add_subplot(gs[row * 3 + 2, col])
                
                meta_text = ""
                for key, value in metadata.items():
                    meta_text += f"{key}: {value}\n"
                    
                if not meta_text:
                    meta_text = "Нет метаданных"
                
                ax_meta.text(0.5, 0.5, meta_text,
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontproperties=font_prop,
                           fontsize=font_size - 2,
                           color=style_settings["accent_color"],
                           wrap=True)
                
                ax_meta.set_facecolor(style_settings["bg_color"])
                ax_meta.axis('on')
                for spine in ax_meta.spines.values():
                    spine.set_edgecolor(style_settings["grid_color"])
                    spine.set_linestyle(':')
                ax_meta.set_xticks([])
                ax_meta.set_yticks([])
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации образца {idx}: {e}")
            if show_metadata:
                ax_error = fig.add_subplot(gs[row * 3:(row * 3 + 3), col])
            else:
                ax_error = fig.add_subplot(gs[row * 2:(row * 2 + 2), col])
            
            ax_error.text(0.5, 0.5, f"Ошибка:\n{str(e)[:100]}", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='red',
                        fontproperties=font_prop,
                        wrap=True)
            ax_error.axis('off')
    
    plt.suptitle(f"Визуализация {num_samples} образцов из {dataset.split} набора данных",
                fontsize=font_size + 4, fontproperties=font_prop, color=style_settings["title_color"])
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=style_settings["bg_color"])
    plt.close(fig)
    
    logger.info(f"Визуализация сохранена в {output_path}")


def visualize_samples_with_pil(
    dataset: DonutDataset,
    output_path: str,
    num_samples: int,
    grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
    font_size: int = 12,
    max_text_length: int = 50,
    shuffle: bool = False,
    style: str = "default",
    show_metadata: bool = False,
    text_bg_color: str = "white",
    text_color: str = "black",
    border_width: int = 2,
    border_color: str = "#888888"
) -> None:
    """
    Визуализирует выборку семплов из датасета с использованием PIL.
    
    Args:
        dataset: Датасет для визуализации
        output_path: Путь к файлу для сохранения
        num_samples: Количество образцов для визуализации
        grid_size: Размер сетки (строки, столбцы)
        image_size: Размер изображения (высота, ширина) для отображения
        font_size: Размер шрифта
        max_text_length: Максимальная длина текста
        shuffle: Флаг перемешивания индексов
        style: Стиль визуализации
        show_metadata: Показывать дополнительные метаданные
        text_bg_color: Цвет фона для текста
        text_color: Цвет текста
        border_width: Ширина рамки
        border_color: Цвет рамки
    """
    if num_samples > len(dataset):
        logger.warning(f"Запрошено {num_samples} образцов, но доступно только {len(dataset)}. "
                      f"Будет использовано {len(dataset)} образцов.")
        num_samples = len(dataset)
    
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    rows, cols = grid_size
    img_height, img_width = image_size
    
    # Получаем настройки стиля
    style_settings = get_style_settings(style)
    bg_color = style_settings["bg_color"]
    title_color = style_settings["title_color"]
    accent_color = style_settings["accent_color"]
    
    # Определяем высоту для текста и метаданных
    text_height = 40
    metadata_height = 40 if show_metadata else 0
    cell_height = img_height + text_height + metadata_height + 30  # Дополнительное пространство для заголовка
    
    full_width = cols * img_width
    full_height = rows * cell_height + 60  # Дополнительное пространство для основного заголовка
    
    full_image = Image.new('RGB', (full_width, full_height), color=bg_color)
    draw = ImageDraw.Draw(full_image)
    
    try:
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',             # Linux
            '/usr/share/fonts/TTF/DejaVuSans.ttf',                         # Linux
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',        # macOS
            'C:\\Windows\\Fonts\\arial.ttf',                               # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # Другой вариант для Linux
        ]
        
        font = None
        for font_path in font_paths:
            if (os.path.exists(font_path)):
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"Используется шрифт: {font_path}")
                break
                
        if font is None:
            font = ImageFont.load_default()
            logger.warning("Не найден шрифт с поддержкой кириллицы. Используется шрифт по умолчанию.")
            
        # Шрифт для заголовка (крупнее)
        title_font = None
        for font_path in font_paths:
            if (os.path.exists(font_path)):
                title_font = ImageFont.truetype(font_path, font_size + 4)
                break
                
        if title_font is None:
            title_font = font
            
        # Шрифт для метаданных (мельче)
        small_font = None
        for font_path in font_paths:
            if (os.path.exists(font_path)):
                small_font = ImageFont.truetype(font_path, max(font_size - 2, 8))
                break
                
        if small_font is None:
            small_font = font
    except Exception as e:
        font = ImageFont.load_default()
        title_font = font
        small_font = font
        logger.warning(f"Ошибка при загрузке шрифта: {e}. Используется шрифт по умолчанию.")

    title = f"Визуализация {num_samples} образцов из {dataset.split} набора данных"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((full_width - title_width) // 2, 20), title, fill=title_color, font=title_font)

    for i, idx in enumerate(selected_indices):
        if i >= rows * cols:
            break
            
        row, col = divmod(i, cols)
        x = col * img_width
        y = row * cell_height + 60  # Учитываем отступ для основного заголовка
        
        try:
            image, text, token_sequence = dataset.visualize_sample(idx)
            metadata = {}
            
            if hasattr(dataset, 'samples') and 0 <= idx < len(dataset.samples):
                metadata = extract_metadata_from_sample(dataset.samples[idx])
            
            if image.size != (img_width, img_height):
                image = image.resize((img_width, img_height), Image.BILINEAR)
            
            # Добавляем рамку
            if border_width > 0:
                image = apply_border_to_image(image, width=border_width, color=border_color)
            
            full_image.paste(image, (x, y))
            
            # Добавляем заголовок образца
            sample_title = f"Образец #{idx}"
            draw.text((x + 5, y - 20), sample_title, fill=title_color, font=font)
            
            # Добавляем текст образца на фоне заданного цвета
            text_truncated = truncate_text(text, max_text_length)
            text_width = draw.textlength(text_truncated, font=font)
            
            # Создаем фон для текста
            text_bg = Image.new('RGB', (img_width, text_height), color=text_bg_color)
            full_image.paste(text_bg, (x, y + img_height))
            
            # Добавляем текст
            text_x = x + (img_width - text_width) // 2
            text_y = y + img_height + (text_height - font_size) // 2
            draw.text((text_x, text_y), text_truncated, fill=text_color, font=font)
            
            # Добавляем метаданные, если нужно
            if show_metadata and metadata:
                meta_text = ""
                for key, value in metadata.items():
                    meta_text += f"{key}: {value}  "
                
                meta_width = draw.textlength(meta_text, font=small_font)
                meta_x = x + (img_width - meta_width) // 2
                meta_y = y + img_height + text_height + 5
                draw.text((meta_x, meta_y), meta_text, fill=accent_color, font=small_font)
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации образца {idx}: {e}")
            
            # Создаем пустую область с сообщением об ошибке
            error_bg = Image.new('RGB', (img_width, img_height), color='#ffeeee')
            full_image.paste(error_bg, (x, y))
            
            error_text = f"Ошибка: {str(e)[:100]}"
            error_width = draw.textlength(error_text, font=font)
            error_x = x + (img_width - error_width) // 2
            error_y = y + (img_height - font_size) // 2
            draw.text((error_x, error_y), error_text, fill='red', font=font)
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_image.save(output_path)
    logger.info(f"Визуализация сохранена в {output_path}")


def main() -> None:
    args = parse_args()
    
    # Добавляем расширение .png к выходному пути, если расширение отсутствует
    if not os.path.splitext(args.output_path)[1]:
        args.output_path = f"{args.output_path}.png"
        logger.info(f"Добавлено расширение файла, новый путь: {args.output_path}")
    
    # Выбираем источник для загрузки процессора
    processor_path = args.checkpoint_path if args.checkpoint_path else args.model_name
    
    if processor_path:
        logger.info(f"Инициализация процессора из {processor_path}")
        processor = init_processor(processor_path)
    else:
        # Для простой визуализации создаем минимальный процессор, если не указан путь к модели/чекпоинту
        logger.info("Создание минимального процессора для визуализации")
        processor = init_processor("")
    
    logger.info(f"Загрузка датасета из {args.data_dir}, раздел '{args.split}'")
    try:
        dataset = DonutDataset(
            processor=processor,
            data_dir=args.data_dir,
            split=args.split,
            max_length=args.max_length,
            image_size=args.image_size,
            task_start_token=args.task_start_token,
            prompt_end_token=args.prompt_end_token,
            apply_augmentation=args.with_augmentation
        )
        
        logger.info(f"Загружено {len(dataset)} образцов")
        if len(dataset) == 0:
            logger.error(f"Датасет пуст. Проверьте путь к данным и название раздела.")
            sys.exit(1)
            
        random.seed(args.random_seed)
        
        # Выбор бэкенда для визуализации
        if args.visualization_backend == "auto":
            logger.info("Используется автоматический выбор бэкенда визуализации")
            try:
                visualize_samples_with_pil(
                    dataset=dataset,
                    output_path=args.output_path,
                    num_samples=args.samples,
                    grid_size=args.grid_size,
                    image_size=args.image_size,
                    font_size=args.font_size,
                    max_text_length=args.max_text_length,
                    shuffle=args.shuffle,
                    style=args.style,
                    show_metadata=args.show_metadata,
                    text_bg_color=args.text_bg_color,
                    text_color=args.text_color,
                    border_width=args.border_width,
                    border_color=args.border_color
                )
            except Exception as e:
                logger.warning(f"Не удалось создать визуализацию с PIL: {e}. Пробуем matplotlib...")
                visualize_samples(
                    dataset=dataset,
                    output_path=args.output_path,
                    num_samples=args.samples,
                    grid_size=args.grid_size,
                    dpi=args.dpi,
                    font_size=args.font_size,
                    max_text_length=args.max_text_length,
                    shuffle=args.shuffle,
                    style=args.style,
                    show_metadata=args.show_metadata,
                    text_bg_color=args.text_bg_color,
                    text_color=args.text_color,
                    border_width=args.border_width,
                    border_color=args.border_color
                )
        elif args.visualization_backend == "pil":
            logger.info("Используется бэкенд визуализации PIL")
            try:
                visualize_samples_with_pil(
                    dataset=dataset,
                    output_path=args.output_path,
                    num_samples=args.samples,
                    grid_size=args.grid_size,
                    image_size=args.image_size,
                    font_size=args.font_size,
                    max_text_length=args.max_text_length,
                    shuffle=args.shuffle,
                    style=args.style,
                    show_metadata=args.show_metadata,
                    text_bg_color=args.text_bg_color,
                    text_color=args.text_color,
                    border_width=args.border_width,
                    border_color=args.border_color
                )
            except Exception as e:
                logger.error(f"Ошибка при визуализации с PIL: {e}", exc_info=True)
                sys.exit(1)
        else:  # matplotlib
            logger.info("Используется бэкенд визуализации matplotlib")
            try:
                visualize_samples(
                    dataset=dataset,
                    output_path=args.output_path,
                    num_samples=args.samples,
                    grid_size=args.grid_size,
                    dpi=args.dpi,
                    font_size=args.font_size,
                    max_text_length=args.max_text_length,
                    shuffle=args.shuffle,
                    style=args.style,
                    show_metadata=args.show_metadata,
                    text_bg_color=args.text_bg_color,
                    text_color=args.text_color,
                    border_width=args.border_width,
                    border_color=args.border_color
                )
            except Exception as e:
                logger.error(f"Ошибка при визуализации с matplotlib: {e}", exc_info=True)
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке или визуализации датасета: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()