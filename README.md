# TritonServer с Ray Serve

🚀 Часть проекта FrameReader, развертывание моделей компьютерного зрения с помощью NVIDIA Triton Server и Ray Serve.

---

## 📋 Поддерживаемые модели

| Модель | Хабы | Назначение | Размеры изображений |
|--------|----------|------------|------------|
| **YOLO** | Ultralytics | 640x640 | Детекция и классификация текста на кадрах |
| **Donut** | HuggingFace | 384x384 | OCR текста на кропнутых кадрах |

---

## 🛠️ Требования

- **Docker** >= 20.10
- **NVIDIA Docker Runtime** (для GPU поддержки)
- **Linux/WSL2** (рекомендуется Ubuntu 24.04+)
- **WeightsModels** ([ссылка на веса моделей](https://drive.google.com/drive/folders/15ujiDLGJ-_BtJrISSBLypJY3FtRFLTL2?usp=sharing))

```
Веса необходимо поместить по пути *./models/{donut|yolo}/1*
```

*Модели в формате tensorrt, если проблем с yolo не будет, достаточно конвертировать через model.export(dormat="engine"), то donut может не завестись, код обучения в этой [ветке](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/train), там есть конвертор в tensorrt, по вопросам создавайте issues*

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
│   │   ├── 1/
│   │   └── config.pbtxt
│   └── yolo/
│   │   ├── 1/
│       └── config.pbtxt
├── docker/
├── scripts/
├── build.sh
├── run.sh
└── tritonserver_deployment.py
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

### Пример ответов Main

```
{
  "status": "success",
  "model": "donut",
  "processed_images": 1,
  "results": [
    {
      "text_sequence": "А"
    }
  ]
}
```

### Пример ответов Streaming

```
event: start
data: {"total_images": 1, "model": "yolo", "batch_size": 1, "chunk_size": 1}

event: processing
data: {"chunk": 0, "images_in_chunk": 1, "progress": "1/1", "percentage": 100.0}

event: result
data: {"chunk": 0, "results": [{"boxes": [[393.4386901855469, 344.0882568359375, 507.4948425292969, 364.98529052734375], [178.00167846679688, 240.47451782226562, 287.5937805175781, 262.80584716796875], [425.69769287109375, 488.5166015625, 566.0052490234375, 510.41778564453125], [147.9241943359375, 325.3453369140625, 249.95932006835938, 350.301513671875], [344.65374755859375, 205.58306884765625, 494.43841552734375, 230.40255737304688]], "confidences": [0.9903075695037842, 0.9888029098510742, 0.9884862899780273, 0.9752254486083984, 0.9406743049621582], "classes": [0, 0, 0, 0, 0]}], "images_processed": 1, "successful": 1}

event: complete
data: {"status": "success", "total_processed": 1, "model": "yolo"}
```

---

### Поддержка WebSockets

*Бета версия, еще тестируется*

## 🧪 Тестирование

### Проверка работоспособности

```bash
curl -X GET http://localhost:8080/v2/health/ready
curl -X GET http://localhost:8080/v2/models
curl -X GET http://localhost:8000/-/healthz
```

### Тесты

```
uv sync
uv run ./tritonserver_test_main.py
uv run ./tritonserver_test_streaming.py
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

После запуска документация (Swagger) доступна по адресам:

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

**🔗 Связанные проекты:**
- [FrameReader Train](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/train) - Обучение моделей
- [FrameReader Server](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/server) - Бекенд