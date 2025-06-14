from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings
from typing import Optional, ClassVar
from pathlib import Path
from src.utils.env import Env


env = Env()


class AuthJWT(BaseModel):
    private_key_path: Optional[str] = None
    public_key_path: Optional[str] = None
    _private_key_content: str = PrivateAttr()
    _public_key_content: str = PrivateAttr()
    access_token_expire_minutes: ClassVar[int] = int(env.__getattr__("ACCESS_TOKEN_EXPIRE_MINUTES"))
    refresh_token_expire_days: ClassVar[int] = int(env.__getattr__("REFRESH_TOKEN_EXPIRE_DAYS"))

    def __init__(self, **data):
        super().__init__(**data)
        self._load_keys()

    def _load_keys(self) -> None:
        script_dir = Path(__file__).resolve().parent
        default_private_key_path = script_dir.parent / "keys" / "jwt-private.pem"
        default_public_key_path = script_dir.parent / "keys" / "jwt-public.pem"

        self._private_key_content = self._read_key_file(default_private_key_path, "private")
        self._public_key_content = self._read_key_file(default_public_key_path, "public")

    def _read_key_file(self, path: Path, key_type: str) -> str:
        try:
            return path.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"{key_type.capitalize()} key file not found at {path}")
        except Exception as e:
            raise IOError(f"Error reading {key_type} key file {path}: {e}")

    @property
    def private_key_content(self) -> str:
        return self._private_key_content

    @property
    def public_key_content(self) -> str:
        return self._public_key_content


class Settings(BaseSettings):
    auth_jwt: AuthJWT = Field(default_factory=AuthJWT)
    algorithm: str = "RS256"