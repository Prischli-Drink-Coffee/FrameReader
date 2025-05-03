import os
import uuid
from aiofiles import open as aio_open
from config import Config
config = Config()


async def write_file_into_server(name_object: str, file) -> None:
    # Получаем расширение файла
    file_extension = file.filename.split('.')[-1]
    # Создаем уникальное имя файла
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    # Записываем путь к файлу
    file_location = os.path.join(config.__getattr__("UPLOAD_DIR"), f"{name_object}", unique_filename)
    # Проверяем существует ли папка, в которой храняться файлы
    os.makedirs(os.path.join(config.__getattr__("UPLOAD_DIR"), f"{name_object}"), exist_ok=True)
    # Открывааем файл и записываем данные изображения
    async with aio_open(file_location, "wb") as buffer:
        await buffer.write(await file.read())
    return unique_filename
