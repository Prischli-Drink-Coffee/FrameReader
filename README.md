# TritonServer с Ray Serve

🚀 Проект по развертыванию моделей компьютерного зрения и OCR с помощью NVIDIA Triton Server и Ray Serve.

---

## 📋 Поддерживаемые модели

| Модель | Описание | Назначение |
|--------|----------|------------|
| **YOLO** | Модель обнаружения объектов | Детекция и классификация объектов на изображениях |
| **Donut** | OCR модель | Извлечение текста из документов и изображений |

---

## 🛠️ Требования

- **Docker** >= 20.10
- **NVIDIA Docker Runtime** (для GPU поддержки)
- **Linux/WSL2** (рекомендуется Ubuntu 20.04+)

---

## 📦 Установка

### 1. Проверка Docker

Убедитесь, что Docker установлен и запущен:

```bash
docker --version
docker info
```

### 2. Установка NVIDIA Docker Runtime

```bash
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 🚀 Быстрый старт

### 1. Сборка Docker-образа

```bash
./build.sh
```

Дополнительные опции:

```bash
./build.sh --framework OCR
./build.sh --no-cache
./build.sh --tag my-triton:latest
./build.sh --dry-run
```

### 2. Запуск контейнера

```bash
./run.sh
```

Дополнительные опции:

```bash
./run.sh --image my-triton:latest
./run.sh --framework OCR
./run.sh --dry-run
```

### 3. Проверка состояния

После запуска сервисы будут доступны на:

* Triton Server: `http://localhost:8080`
* Ray Serve: `http://localhost:8000`
* Metrics: `http://localhost:8002/metrics`
* DashBoard: `http://localhost:8265`

---

## 📊 Мониторинг и управление

### Вход в контейнер

```bash
docker exec -it tritonserver /bin/bash
```

### Просмотр логов

```bash
pm2 logs
pm2 logs triton
pm2 logs deploy
```

### Управление сервисами

```bash
pm2 status
pm2 restart all
pm2 stop all
```

### Остановка контейнера

```bash
docker stop tritonserver
```

### Полная очистка

```bash
docker stop tritonserver
docker rm tritonserver
```

---

## 🔧 Конфигурация

### Структура проекта

```
├── backend/
│   ├── donut/
│   │   ├── engine.py
│   │   └── model.py
│   └── yolo/
│       └── model.py
├── models/
│   ├── donut/
│   │   └── config.pbtxt
│   └── yolo/
│       └── config.pbtxt
├── docker/
├── scripts/
├── build.sh
├── run.sh
└── tritonserver_deployment.py
```

### Параметры моделей

#### YOLO

* Входные данные: RGB изображения 640x640
* Выходные данные: JSON с обнаруженными объектами
* Параметры: confidence, IoU threshold, max detections

#### Donut

* Входные данные: RGB изображения 384x384
* Выходные данные: JSON с извлеченным текстом
* Параметры: max_length, num_beams, prompt

---

## 🧪 Тестирование

### Проверка работоспособности

```bash
curl -X GET http://localhost:8080/v2/health/ready
curl -X GET http://localhost:8080/v2/models
curl -X GET http://localhost:8000/-/healthz
```

### Пример запроса

```
import requests
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/yolo",
    json={"image": image_data}
)

print(response.json())
```

---

## 🔍 Отладка

### Общие проблемы

#### Контейнер не запускается

```bash
docker logs tritonserver
```

#### Нет доступа к GPU

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### Порты заняты

```bash
netstat -tulpn | grep :8080
```

#### Проблемы с моделями

```bash
pm2 logs
```

### Логи системы

* Triton Server: `./logs/triton_out.log`, `./logs/triton_err.log`
* Ray Serve: `./logs/deploy_out.log`, `./logs/deploy_err.log`

---

## 📚 API документация

После запуска документация доступна по адресам:

* Triton API: `http://localhost:8080/docs`
* Ray Serve: `http://localhost:8000/docs`

---

## 🤝 Поддержка

При возникновении проблем:

* Проверьте логи: `pm2 logs`
* Убедитесь в наличии GPU: `nvidia-smi`
* Проверьте статус сервисов: `pm2 status`
* Перезапустите контейнер: `docker restart tritonserver`

---


