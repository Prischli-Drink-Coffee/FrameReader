from fastapi.responses import FileResponse, JSONResponse
from load_dotenv import load_dotenv

load_dotenv()

def return_url_object(url: str) -> str:
    return (f"http://{os.getenv('HOST')}:{os.getenv('SERVER_PORT')}/"
            f"public{url}")
