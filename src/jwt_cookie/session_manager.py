from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status, Response
from fastapi.security import HTTPBearer
import jwt
import hashlib
import secrets
from src.jwt_cookie.settings import Settings
from src.utils.custom_logging import setup_logging

log = setup_logging()
settings = Settings()


class FingerprintCollector:
    @staticmethod
    def generate_fingerprint_hash(request: Request) -> str:
        user_agent = request.headers.get("user-agent", "")
        accept_language = request.headers.get("accept-language", "")
        accept_encoding = request.headers.get("accept-encoding", "")
        
        fingerprint_data = f"{user_agent}|{accept_language}|{accept_encoding}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()


class JWTCookieManager:
    def __init__(self):
        self.cookie_name = "session_token"
        self.cookie_max_age = 30 * 24 * 60 * 60
        self.settings = Settings()
        
    def create_user_token(self, user_id: int, fingerprint_hash: str, session_id: int) -> str:
        payload = {
            "user_id": user_id,
            "fingerprint_hash": fingerprint_hash,
            "session_id": session_id,
            "token_type": "user_session",
            "exp": datetime.utcnow() + timedelta(seconds=self.cookie_max_age),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(
            payload=payload,
            key=self.settings.auth_jwt.private_key_content,
            algorithm=self.settings.algorithm
        )
    
    def decode_token(self, token: str) -> Dict:
        try:
            payload = jwt.decode(
                token,
                self.settings.auth_jwt.public_key_content,
                algorithms=[self.settings.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )
    
    def set_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=self.cookie_max_age,
            httponly=True,
            secure=True,
            samesite="lax"
        )
    
    def get_token_from_request(self, request: Request) -> Optional[str]:
        return request.cookies.get(self.cookie_name)
    
    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(key=self.cookie_name)
