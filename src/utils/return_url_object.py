from fastapi.responses import FileResponse, JSONResponse
from config import Config
config = Config()


def return_url_object(url: str) -> str:
    return (f"http://{config.__getattr__('HOST')}:{config.__getattr__('SERVER_PORT')}/"
            f"public{url}")
