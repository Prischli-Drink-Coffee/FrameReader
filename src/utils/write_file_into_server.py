import os
import uuid
from aiofiles import open as aio_open
from load_dotenv import load_dotenv

load_dotenv()


async def write_file_into_server(name_object: str, file) -> None:
    file_extension = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_location = os.path.join(os.getenv("UPLOAD_DIR"), f"{name_object}", unique_filename)
    os.makedirs(os.path.join(os.getenv("UPLOAD_DIR"), f"{name_object}"), exist_ok=True)
    async with aio_open(file_location, "wb") as buffer:
        await buffer.write(await file.read())
    return unique_filename
