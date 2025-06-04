# 🧠 FrameReader Training

🚀 Обучение моделей компьютерного зрения для детекции и распознавания текста на русском языке.
Часть проекта FrameReader.

---

## 📋 Поддерживаемые модели

| Модель | Назначение | Особенности | Размер изображений |
|--------|------------|-------------|-------------------|
| **YOLOv12** | Детекция текста | Обучение через Jupyter | 640x640 |
| **Donut** | OCR документов | QAT/PTQ квантизация, TensorRT | 384x384 |
| **TrOCR** | OCR текста | Оптимизация A100, FlashAttention | 384x384 |

---

## 🛠️ Требования

- **Python** >= 3.10
- **NVIDIA GPU** (рекомендуется A100 80GB)
- **CUDA** >= 12.1
- **Датасет** [RusTitW](https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition)

### Установка зависимостей

```bash
# Установка через uv (рекомендуется)
uv pip install -r requirements.txt

# Или стандартным способом
pip install -r requirements.txt
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
python train_donut/valid_metadata_jsonl.py \
  --dir path/to/metadata \
  --fix
```

### 3. Визуализация датасета

```bash
# Просмотр образцов данных
python train_trocr/visualize_samples.py \
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

### 2. TrOCR (Распознавание текста)

```bash
python train_trocr/train.py \
  --data_dir path/to/dataset \
  --output_dir ./trocr_output \
  --batch_size 32 \
  --precision bf16 \
  --image_size 384 384 \
  --learning_rate 5e-5 \
  --num_epochs 10
```

**Дополнительные параметры:**

```bash
# Для экономии памяти GPU
python train_trocr/train.py \
  --freeze_encoder \
  --gradient_accumulation_steps 32 \
  --flash_attention \
  --batch_size 16
```

### 3. Donut (OCR текста)

```bash
python train_donut/train.py \
  --data_dir path/to/documents \
  --output_dir ./donut_output \
  --batch_size 80 \
  --image_size 384 384 \
  --apply_augmentation \
  --num_epochs 30
```

**Параметры аугментации:**

```bash
# Настройка аугментаций
python train_donut/train.py \
  --augmentation_prob 0.3 \
  --max_rotation 8.0 \
  --noise_level 0.05
```

---

## 🔮 Инференс моделей

### TrOCR

```bash
# Пакетное распознавание изображений
python train_trocr/inference.py \
  --model_dir ./trocr_output/best_model \
  --input path/to/images \
  --batch_size 16 \
  --image_size 384 384
```

### Donut

```bash
# Распознавание документов
python train_donut/inference.py \
  --model_path Akajackson/donut_rus \
  --dataset_path path/to/documents \
  --batch_size 8 \
  --precision fp16
```

### Оптимизированный Donut (после квантизации)

```bash
# Инференс квантизованной модели
python train_donut/qat_ptq_inference.py \
  --model_path ./donut_output/quantized \
  --input_dir path/to/test_images \
  --engine_type trt_int8
```

---

## ⚡️ Оптимизация моделей

### Квантизация Donut

```bash
# QAT (Quantization Aware Training)
python train_donut/qat_ptq_trainer.py \
  --pretrained_model_path ./donut_output \
  --data_dir path/to/dataset \
  --optimization_type qat \
  --num_epochs 5

# PTQ (Post Training Quantization) + TensorRT
python train_donut/qat_ptq_trainer.py \
  --pretrained_model_path ./donut_output \
  --data_dir path/to/dataset \
  --optimization_type trt_int8 \
  --calibration_batches 32
```

### Экспорт в TensorRT

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
├── train_donut/
│   ├── train.py                   
│   ├── inference.py               
│   ├── qat_ptq_trainer.py         
│   ├── qat_ptq_inference.py       
│   ├── visualize_samples.py       
│   └── valid_metadata_jsonl.py    
├── train_trocr/
│   ├── train.py                   
│   ├── inference.py               
│   └── visualize_samples.py       
├── tracker/
│   └── process_video.py           
├── requirements.txt               
└── README.md                      
```

---

## 🔧 Конфигурация обучения

### Рекомендуемые параметры

#### TrOCR (A100 80GB)

```python
{
    "batch_size": 32,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.0005,
    "gradient_accumulation_steps": 32,
    "precision": "bf16",
    "image_size": [384, 384],
    "freeze_encoder": True,
    "flash_attention": True
}
```

#### Donut (документы)

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
# Уменьшить batch_size
python train_trocr/train.py --batch_size 8

# Использовать накопление градиентов
python train_trocr/train.py --gradient_accumulation_steps 64

# Заморозить энкодер
python train_trocr/train.py --freeze_encoder
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
python train_donut/valid_metadata_jsonl.py --dir data --fix

# Проверка путей к изображениям
python train_trocr/visualize_samples.py --data_dir data --samples 5
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
| TrOCR-Ru | [raxtemur/trocr-base-ru](https://huggingface.co/raxtemur/trocr-base-ru) | TrOCR для русского текста |
| Donut-Ru | [Akajackson/donut_rus](https://huggingface.co/Akajackson/donut_rus) | Donut для русских документов |
| YOLOv8 | [ultralytics/yolov12](https://github.com/ultralytics/ultralytics) | Базовые веса YOLO |

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