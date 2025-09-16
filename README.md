# 🧠 FrameReader Training

🚀 Обучение моделей компьютерного зрения для детекции и распознавания текста на русском языке.
Часть проекта FrameReader.

---

## 📋 Поддерживаемые модели

| Модель | Назначение | Особенности | Размер изображений |
|--------|------------|-------------|-------------------|
| **YOLOv12** | Детекция текста | Обучение через Jupyter | 640x640 |
| **Donut** | OCR документов | TensorRT | 384x384 |

---

## 🛠️ Требования

- **Python** >= 3.10
- **NVIDIA GPU** (рекомендуется A100 80GB)
- **CUDA** >= 12.1
- **Датасет** [RusTitW](https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition)

### Установка зависимостей

```bash
# Установка через uv
uv sync
```

---

## 📦 Подготовка данных

### 1. Загрузка датасета RusTitW

```bash
# Скачивание и подготовка данных
jupyter notebook notebooks/download_dataset.ipynb
```

### 2. Валидация данных

```bash
# Проверка и исправление метаданных
python donut/valid_metadata_jsonl.py \
  --dir path/to/metadata \
  --fix
```

### 3. Визуализация датасета

```bash
# Просмотр образцов данных
python donut/visualize_samples.py \
  --data_dir path/to/dataset \
  --split train \
  --samples 25
```

---

## 🚀 Обучение моделей

### 1. YOLOv12 (Детекция текста)

```bash
# Запуск через Jupyter Notebook
jupyter notebook notebooks/text_detection.ipynb
```

**Основные настройки:**
- Размер изображения: 640x640
- Аугментации: поворот, масштабирование, цветокоррекция
- Экспорт в TensorRT для продакшена

### 2. Donut (OCR текста)

```bash
python donut/train.py \
  --data_dir path/to/documents \
  --output_dir ./output \
  --batch_size 80 \
  --image_size 384 384 \
  --apply_augmentation \
  --num_epochs 30
```

**Параметры аугментации:**

```bash
# Настройка аугментаций
python donut/train.py \
  --augmentation_prob 0.3 \
  --max_rotation 8.0 \
  --noise_level 0.05
```

---

## 🔮 Инференс моделей

### Donut

```bash
# Распознавание документов
python donut/inference.py \
  --model_path Akajackson/donut_rus \
  --dataset_path path/to/documents \
  --batch_size 8 \
  --precision fp16
```

### Конвертация Donut

```bash
# Конвертация модели
python donut/convert_to_tensorrt.py \
  --model_path ./output/checkpoint-10 \
  --output_dir ./output/engine
  --precision fp16
  --image_size 384,384
  --max_batch_size 1
```

### Оптимизированный Donut (после конвертации)

```bash
# Инференс конвертированной модели
python donut/inference.py \
  --model_path ./output/engine \
```

---

### Конвертация Yolo

```bash
# Конвертация YOLO в TensorRT
python -c "
from ultralytics import YOLO
model = YOLO('./yolo_weights.pt')
model.export(format='engine', imgsz=640)
"
```

---

## 🎥 Обработка видео

### Трекинг текста

```bash
# Анализ видео с помощью YOLO + SAHI
python tracker/process_video.py \
  --source video.mp4 \
  --yolo_model ./yolo_weights.pt \
  --output analyzed_video.mp4 \
  --conf_threshold 0.5 \
  --tracker bytetrack
```

**Поддерживаемые форматы:**
- Входные: MP4, AVI, MOV, MKV
- Выходные: MP4 с аннотациями

---

## 📊 Структура ветки

```
train/
├── notebooks/
│   ├── download_dataset.ipynb     
│   └── text_detection.ipynb       
├── donut/
│   ├── train.py                   
│   ├── inference.py               
│   ├── convert_to_tensorrt.py         
│   ├── dataset.py       
│   ├── visualize_samples.py
│   ├── trainer.py
│   ├── utils.py 
│   └── valid_metadata_jsonl.py         
├── tracker/
│   └── process_video.py           
├── requirements.txt               
└── README.md                      
```

---

## 🔧 Конфигурация обучения

### Рекомендуемые параметры

#### Donut (6 vRAM)

*Двустадийное обучение*

```python
{
    "batch_size": 32,
    "learning_rate": 1e-4,
    "image_size": [384, 384],
    "apply_augmentation": True,
    "augmentation_prob": 0.3,
    "max_rotation": 8.0,
    "noise_level": 0.05,
    "num_epochs": 25
}
```

#### YOLO (детекция)

```python
{
    "imgsz": 640,
    "epochs": 100,
    "batch": 512,
    "lr0": 0.01,
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.1
}
```

---

## 🔍 Отладка и решение проблем

### Частые проблемы

#### 1. Нехватка памяти GPU

**Симптомы:**
```
CUDA out of memory. Tried to allocate X MiB
```

**Решения:**
```bash
* Уменьшить batch_size
* Использовать накопление градиентов
* Заморозить энкодер
```

#### 2. Проблемы с данными

**Симптомы:**
```
KeyError: 'text'
FileNotFoundError: image not found
```

**Решения:**
```bash
# Валидация и исправление
python donut/valid_metadata_jsonl.py --dir data --fix

# Проверка путей к изображениям
python donut/visualize_samples.py --data_dir data --samples 5
```

#### 3. Низкая точность модели

**Решения:**
- Увеличить количество эпох: `--num_epochs 50`
- Включить аугментацию: `--apply_augmentation`
- Настроить learning rate: `--learning_rate 1e-4`

### Мониторинг обучения

```bash
# Отслеживание через TensorBoard
tensorboard --logdir ./logs

# Просмотр логов в реальном времени
tail -f train_output.log

# Проверка использования GPU
watch -n 1 nvidia-smi
```

---

## 📚 Дополнительные ресурсы

### Документация

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

### Предобученные модели

| Модель | HuggingFace Hub | Описание |
|--------|----------------|----------|
| Donut-Ru | [Akajackson/donut_rus](https://huggingface.co/Akajackson/donut_rus) | Donut для русских документов |
| YOLOv12 | [ultralytics/yolov12](https://github.com/ultralytics/ultralytics) | Базовые веса YOLO |

### Датасеты

- [RusTitW](https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition) - Русский текст в изображениях
- [Synthetic Russian OCR](https://huggingface.co/datasets/sberbank-ai/synthetic_russian_ocr) - Синтетические данные

---

## 🤝 Поддержка

### Получение помощи

При возникновении проблем:

1. **Проверьте системные требования:**
   ```bash
   python --version  # >= 3.10
   nvidia-smi       # CUDA доступна
   ```

2. **Проверьте логи:**
   ```bash
   # Последние ошибки обучения
   tail -n 50 train_output.log
   ```

3. **Создайте issue** с информацией:
   - Версия Python и CUDA
   - Конфигурация GPU
   - Полный лог ошибки
   - Используемые параметры

---

**🔗 Связанные проекты:**
- [FrameReader Triton Server](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/triton-server) - Развертывание обученных моделей
- [FrameReader Server](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/server) - Бекенд