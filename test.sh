#!/bin/bash

set -e

echo "Тестирование Triton Server с Ray Serve..."

mkdir -p /workspace/docs

TEST_IMAGE="/workspace/docs/test.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "Создание тестового изображения..."
    
    if command -v convert >/dev/null 2>&1; then
        convert -size 512x512 canvas:white -font Arial -pointsize 40 -draw "text 50,250 'Hello Triton Server!'" $TEST_IMAGE
    else
        echo "ImageMagick не найден, создаем изображение через Python..."
        python3 -c "
from PIL import Image, ImageDraw, ImageFont
import os
# Создаем белое изображение 512x512
image = Image.new('RGB', (512, 512), color='white')
# Получаем объект для рисования
draw = ImageDraw.Draw(image)
# Пытаемся использовать шрифт Arial, если он есть
try:
    font = ImageFont.truetype('Arial', 40)
except:
    # Если Arial не найден, используем шрифт по умолчанию
    font = ImageFont.load_default()
# Добавляем текст на изображение
draw.text((50, 250), 'Hello Triton Server!', fill='black', font=font)
# Сохраняем изображение
image.save('$TEST_IMAGE')
print('Тестовое изображение создано: $TEST_IMAGE')
        "
    fi
else
    echo "Используем существующее тестовое изображение: $TEST_IMAGE"
fi

echo "Проверка доступности сервиса..."
if ! curl -s http://localhost:8000 > /dev/null; then
    echo "ОШИБКА: Сервис не доступен на порту 8000. Убедитесь, что Ray Serve запущен."
    exit 1
fi

echo "Тестирование YOLO модели..."
echo "POST http://localhost:8000/generate/yolo"
YOLO_RESPONSE=$(curl -s -X POST -F "image=@$TEST_IMAGE" http://localhost:8000/generate/yolo)
echo "Ответ YOLO модели:"
echo "$YOLO_RESPONSE" | python3 -m json.tool

echo -e "\nТестирование Donut модели..."
echo "POST http://localhost:8000/generate/donut"
DONUT_RESPONSE=$(curl -s -X POST -F "image=@$TEST_IMAGE" http://localhost:8000/generate/donut)
echo "Ответ Donut модели:"
echo "$DONUT_RESPONSE" | python3 -m json.tool

YOLO_STATUS=$(echo "$YOLO_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'error'))")
DONUT_STATUS=$(echo "$DONUT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'error'))")

if [ "$YOLO_STATUS" != "success" ] || [ "$DONUT_STATUS" != "success" ]; then
    echo -e "\n❌ Тестирование НЕ пройдено. Некоторые модели вернули ошибку."
    exit 1
else
    echo -e "\n✅ Тестирование успешно пройдено. Обе модели работают корректно."
fi