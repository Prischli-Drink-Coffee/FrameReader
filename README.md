# FrameReader Backend

🚀 Основной сервер проекта FrameReader, реализующий логику работы с базой данных, трекинг видео и интеграцию с Triton Server.

---

## 📋 Поддерживаемые функции

| Функция | Описание |
|---------|----------|
| **Трекинг видео** | Реализован с использованием YOLO для детекции текста на кадрах видео. |
| **Распознавание текста** | Использует Donut для OCR на кропнутых кадрах. |
| **WebSocket API** | Позволяет передавать видео в реальном времени и получать результаты трекинга и распознавания. |
| **База данных** | Хранение данных о пользователях, сессиях и результатах обработки видео. |

---

## 🛠️ Требования

- **Docker** >= 20.10
- **MySQL** (локально или в контейнере)
- **Python** >= 3.9
- **Triton Server** (для интеграции с моделями YOLO и Donut)

---

## 🚀 Быстрый старт

### 1. Сборка Docker-образа

```bash
./build.sh
```

Дополнительные опции:

```bash
./build.sh --no-cache
./build.sh --tag my-backend:latest
```

### 2. Запуск контейнера

```bash
./run.sh
```

Дополнительные опции:

```bash
./run.sh --image my-backend:latest
./run.sh --env-file .env
```

### 3. Проверка состояния

После запуска сервисы будут доступны на:

* **FastAPI**: `http://localhost:8000`
* **Swagger UI**: `http://localhost:8000/docs`

---

## 📂 Структура проекта

```
server/
├── src/
│   ├── database/          # Модели и подключение к базе данных
│   ├── pipeline/          # Основной FastAPI сервер
│   ├── repository/        # Репозитории для работы с базой данных
│   ├── scripts/           # Скрипты для трекинга и распознавания
│   ├── services/          # Сервисный слой
│   └── utils/             # Вспомогательные утилиты
├── docker/                # Docker-конфигурации
├── build.sh               # Скрипт сборки
├── run.sh                 # Скрипт запуска
├── db.sh                  # Скрипт инициализации база данных
├── tracker.sh             # Скрипт запуска трекера
├── recognizer.sh          # Скрипт запуска распознавателя
├── server.sh              # Локальный запуск сервера
└── test.sh                # Тесты
```

---

## 📄 Пример запросов

### WebSocket API

```python
import asyncio
import websockets

async def send_video():
    async with websockets.connect("ws://localhost:8000/ws/video_recognition/1") as websocket:
        await websocket.send(json.dumps({
            "video_url": "https://rutube.ru/video/12345/"
        }))
        async for message in websocket:
            print(message)

asyncio.get_event_loop().run_until_complete(send_video())
```

### REST API

```python
import requests

response = requests.post(
    "http://localhost:8000/api/video_sessions",
    json={"user_id": 1, "video_url": "https://rutube.ru/video/12345/"}
)

print(response.json())
```

---

## 🐛 Отладка

### Логи

```bash
docker logs framereader-backend
```

### Проверка состояния базы данных

```bash
docker exec -it mysql-container mysql -u user -p
```

### Перезапуск контейнера

```bash
docker restart framereader-backend
```

---

## 📚 Документация

После запуска сервера документация доступна по адресу:

* **Swagger UI**: `http://localhost:8000/docs`
* **ReDoc**: `http://localhost:8000/redoc`

---

## 🤝 Поддержка

При возникновении проблем:

1. Проверьте логи: `docker logs framereader-backend`
2. Убедитесь, что база данных запущена и доступна
3. Перезапустите контейнер: `docker restart framereader-backend`

---

**🔗 Связанные проекты:**
- [FrameReader Triton Server](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/triton-server) - Сервер моделей YOLO и Donut
- [FrameReader Train](https://github.com/Prischli-Drink-Coffee/FrameReader/tree/train) - Обучение моделей