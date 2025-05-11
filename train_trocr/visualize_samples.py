import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Tuple, List, Optional
import random
import math

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем не-интерактивный бэкенд
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from transformers import TrOCRProcessor
from PIL import Image, ImageFont, ImageDraw

from dataset import TrOCRDataset

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("visualize")

def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Визуализация семплов из тренировочного набора данных TrOCR",
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
        default=150, 
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
        default=50, 
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
        default="raxtemur/trocr-base-ru",
        help="Имя модели из HuggingFace для инициализации процессора (только для токенизатора)"
    )
    
    args = parser.parse_args()
    
    # Проверка согласованности grid_size и samples
    if args.grid_size[0] * args.grid_size[1] < args.samples:
        logger.warning(
            f"Размер сетки {args.grid_size[0]}x{args.grid_size[1]} меньше, чем запрошенное число образцов {args.samples}. "
            f"Установка samples={args.grid_size[0] * args.grid_size[1]}"
        )
        args.samples = args.grid_size[0] * args.grid_size[1]
    
    return args


def init_processor(model_name: str) -> TrOCRProcessor:
    """Инициализация процессора TrOCR для работы с набором данных."""
    try:
        from transformers import AutoImageProcessor, AutoTokenizer
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        return processor
    except Exception as e:
        logger.error(f"Ошибка при инициализации процессора: {e}")
        logger.info("Создание минимального процессора...")
        
        # Минимальная реализация процессора для работы с TrOCRDataset
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
                
                self.tokenizer = MinimalTokenizer()
            
            def __call__(self, images, **kwargs):
                return type('obj', (object,), {'pixel_values': torch.zeros(1, 3, 224, 224)})
        
        return MinimalProcessor()


def truncate_text(text: str, max_length: int) -> str:
    """Усечение текста до максимальной длины с добавлением многоточия."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def visualize_samples(
    dataset: TrOCRDataset,
    output_path: str,
    num_samples: int,
    grid_size: Tuple[int, int],
    dpi: int = 150,
    font_size: int = 12,
    max_text_length: int = 50,
    shuffle: bool = False
) -> None:
    """
    Создание визуализации сетки образцов из датасета.
    
    Args:
        dataset: Датасет TrOCR
        output_path: Путь для сохранения визуализации
        num_samples: Количество образцов для визуализации
        grid_size: Размеры сетки [строки, столбцы]
        dpi: DPI для сохранения изображения
        font_size: Размер шрифта для подписей
        max_text_length: Максимальная длина отображаемого текста
        shuffle: Случайно выбирать образцы
    """
    if num_samples > len(dataset):
        logger.warning(f"Запрошено {num_samples} образцов, но доступно только {len(dataset)}. "
                      f"Будет использовано {len(dataset)} образцов.")
        num_samples = len(dataset)
    
    # Выбор индексов образцов
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    # Создание фигуры и сетки
    rows, cols = grid_size
    figsize = (cols * 4, rows * 4)  # Размер фигуры пропорционален сетке
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(rows * 2, cols, figure=fig, height_ratios=[3, 1] * rows)
    
    # Настройка шрифта для поддержки кириллицы
    try:
        font_prop = FontProperties(family="DejaVu Sans", size=font_size)
    except:
        font_prop = None
        logger.warning("Не удалось установить шрифт DejaVu Sans. Могут возникнуть проблемы с отображением кириллицы.")
    
    # Отображение образцов в сетке
    for i, idx in enumerate(selected_indices):
        if i >= rows * cols:
            break
            
        # Получение изображения и текста
        try:
            image, text = dataset.visualize_sample(idx)
            row, col = divmod(i, cols)
            
            # Размещение изображения
            ax_img = fig.add_subplot(gs[row * 2, col])
            ax_img.imshow(image)
            ax_img.set_title(f"Образец #{idx}", fontproperties=font_prop)
            ax_img.axis('off')
            
            # Размещение текста
            ax_text = fig.add_subplot(gs[row * 2 + 1, col])
            text_truncated = truncate_text(text, max_text_length)
            ax_text.text(0.5, 0.5, text_truncated, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontproperties=font_prop,
                        wrap=True)
            ax_text.axis('off')
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации образца {idx}: {e}")
    
    # Добавление общей информации
    plt.suptitle(f"Визуализация {num_samples} образцов из {dataset.split} набора данных",
                fontsize=font_size + 4, fontproperties=font_prop)
    
    # Настройка отступов
    plt.tight_layout(rect=[0, 0, 1, 1.05])
    
    # Сохранение визуализации
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Визуализация сохранена в {output_path}")


def visualize_samples_with_pil(
    dataset: TrOCRDataset,
    output_path: str,
    num_samples: int,
    grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
    font_size: int = 12,
    max_text_length: int = 50,
    shuffle: bool = False
) -> None:
    """
    Создание визуализации сетки образцов с использованием PIL вместо matplotlib.
    Это более надежный вариант для систем без графического интерфейса.
    
    Args:
        dataset: Датасет TrOCR
        output_path: Путь для сохранения визуализации
        num_samples: Количество образцов для визуализации
        grid_size: Размеры сетки [строки, столбцы]
        image_size: Размер отдельного изображения (высота, ширина)
        font_size: Размер шрифта для подписей
        max_text_length: Максимальная длина отображаемого текста
        shuffle: Случайно выбирать образцы
    """
    if num_samples > len(dataset):
        logger.warning(f"Запрошено {num_samples} образцов, но доступно только {len(dataset)}. "
                      f"Будет использовано {len(dataset)} образцов.")
        num_samples = len(dataset)
    
    # Выбор индексов образцов
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    rows, cols = grid_size
    img_height, img_width = image_size
    
    # Рассчитываем размеры полного изображения
    cell_height = img_height + 40  # Дополнительное место для текста
    full_width = cols * img_width
    full_height = rows * cell_height
    
    # Создаем полное изображение
    full_image = Image.new('RGB', (full_width, full_height), color='white')
    draw = ImageDraw.Draw(full_image)
    
    # Пытаемся загрузить шрифт, который поддерживает кириллицу
    try:
        # Пробуем разные шрифты, которые могут поддерживать кириллицу
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',             # Linux
            '/usr/share/fonts/TTF/DejaVuSans.ttf',                         # Linux
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',        # macOS
            'C:\\Windows\\Fonts\\arial.ttf',                               # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # Другой вариант для Linux
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"Используется шрифт: {font_path}")
                break
                
        if font is None:
            # Если не нашли ни один шрифт, используем шрифт по умолчанию
            font = ImageFont.load_default()
            logger.warning("Не найден шрифт с поддержкой кириллицы. Используется шрифт по умолчанию.")
    except Exception as e:
        font = ImageFont.load_default()
        logger.warning(f"Ошибка при загрузке шрифта: {e}. Используется шрифт по умолчанию.")
    
    # Добавление заголовка
    title = f"Визуализация {num_samples} образцов из {dataset.split} набора данных"
    draw.text((10, 10), title, fill="black", font=font)
    
    # Отображение образцов в сетке
    for i, idx in enumerate(selected_indices):
        if i >= rows * cols:
            break
            
        # Расчет позиции в сетке
        row, col = divmod(i, cols)
        x = col * img_width
        y = row * cell_height + 40  # Смещение вниз из-за заголовка
        
        try:
            # Получение изображения и текста
            image, text = dataset.visualize_sample(idx)
            
            # Преобразование к нужному размеру
            if image.size != (img_width, img_height):
                image = image.resize((img_width, img_height), Image.BILINEAR)
            
            # Вставка изображения
            full_image.paste(image, (x, y))
            
            # Добавление номера образца
            sample_title = f"Образец #{idx}"
            draw.text((x + 5, y - 15), sample_title, fill="black", font=font)
            
            # Добавление текста
            text_truncated = truncate_text(text, max_text_length)
            text_width = draw.textlength(text_truncated, font=font)
            text_x = x + (img_width - text_width) // 2
            text_y = y + img_height + 5
            draw.text((text_x, text_y), text_truncated, fill="black", font=font)
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации образца {idx}: {e}")
            # Добавляем сообщение об ошибке в сетку
            draw.text((x + 5, y + img_height // 2), f"Ошибка: {e}", fill="red", font=font)
    
    # Сохранение визуализации
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_image.save(output_path)
    logger.info(f"Визуализация сохранена в {output_path}")


def main() -> None:
    """Основная функция скрипта."""
    args = parse_args()
    
    logger.info(f"Инициализация процессора для модели {args.model_name}")
    processor = init_processor(args.model_name)
    
    logger.info(f"Загрузка датасета из {args.data_dir}, раздел '{args.split}'")
    try:
        dataset = TrOCRDataset(
            processor=processor,
            data_dir=args.data_dir,
            split=args.split,
            max_length=512,  # Большое значение, так как нам нужен только просмотр
            image_size=args.image_size,
            apply_augmentation=args.with_augmentation
        )
        
        logger.info(f"Загружено {len(dataset)} образцов")
        if len(dataset) == 0:
            logger.error(f"Датасет пуст. Проверьте путь к данным и название раздела.")
            sys.exit(1)
            
        random.seed(args.random_seed)
        
        # Выбор метода визуализации
        try:
            # Сначала пробуем PIL (более надежный вариант)
            visualize_samples_with_pil(
                dataset=dataset,
                output_path=args.output_path,
                num_samples=args.samples,
                grid_size=args.grid_size,
                image_size=args.image_size,
                font_size=args.font_size,
                max_text_length=args.max_text_length,
                shuffle=args.shuffle
            )
        except Exception as e:
            logger.warning(f"Не удалось создать визуализацию с PIL: {e}. Пробуем matplotlib...")
            # Если PIL не сработал, пробуем matplotlib
            visualize_samples(
                dataset=dataset,
                output_path=args.output_path,
                num_samples=args.samples,
                grid_size=args.grid_size,
                dpi=args.dpi,
                font_size=args.font_size,
                max_text_length=args.max_text_length,
                shuffle=args.shuffle
            )
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке или визуализации датасета: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()