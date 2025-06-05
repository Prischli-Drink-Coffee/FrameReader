from pydantic import (BaseModel, Field, StrictStr, Json, condecimal, StrictFloat,
                      StrictInt, PrivateAttr, SecretBytes, StrictBytes, StrictBool)
from pydantic_settings import BaseSettings
from typing import Optional, List, ClassVar
from datetime import datetime
import os
from pathlib import Path
from src.utils.env import Env

env = Env()


ACCESS_TOKEN_EXPIRE_MINUTES = int(config.__getattr__("ACCESS_TOKEN_EXPIRE_MINUTES"))
REFRESH_TOKEN_EXPIRE_DAYS = int(config.__getattr__("REFRESH_TOKEN_EXPIRE_DAYS"))


class AuthJWT(BaseModel):
    private_key_path: Optional[str] = None
    public_key_path: Optional[str] = None
    _private_key_content: str = PrivateAttr()
    _public_key_content: str = PrivateAttr()
    access_token_expire_minutes: ClassVar[int] = ACCESS_TOKEN_EXPIRE_MINUTES
    refresh_token_expire_days: ClassVar[int] = REFRESH_TOKEN_EXPIRE_DAYS

    def __init__(self, **data):
        super().__init__(**data)
        self._private_key_content: Path = Path(__file__).resolve().parent.parent / "keys" / "jwt-private.pem"
        self._public_key_content: Path = Path(__file__).resolve().parent.parent / "keys" / "jwt-public.pem"

    @property
    def private_key_content(self):
        return self._private_key_content

    @property
    def public_key_content(self):
        return self._public_key_content



class Settings(BaseSettings):
    auth_jwt: AuthJWT = AuthJWT()
    algorithm: str = "RS256"

