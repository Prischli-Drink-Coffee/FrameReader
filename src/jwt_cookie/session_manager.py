from typing import Dict, Optional, Any, Protocol
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status, Response
import jwt
import hashlib
from dataclasses import dataclass
from src.jwt_cookie.settings import Settings
from src.utils.custom_logging import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class TokenPayload:
    user_id: int
    fingerprint_hash: str
    session_id: int
    token_type: str = "user_session"

    def to_jwt_dict(self, expiration_delta: timedelta) -> Dict[str, Any]:
        now = datetime.utcnow()
        return {
            "user_id": self.user_id,
            "fingerprint_hash": self.fingerprint_hash,
            "session_id": self.session_id,
            "token_type": self.token_type,
            "exp": int((now + expiration_delta).timestamp()),
            "iat": int(now.timestamp())
        }


class FingerprintCollector:
    @staticmethod
    def generate_fingerprint_hash(request: Request) -> str:
        headers = [
            request.headers.get("user-agent", ""),
            request.headers.get("accept-language", ""),
            request.headers.get("accept-encoding", "")
        ]
        fingerprint_data = "|".join(headers)
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()


class JWTCookieManager:
    def __init__(self) -> None:
        self._cookie_name: str = "session_token"
        self._cookie_max_age: int = 30 * 24 * 60 * 60
        self._settings = Settings()
        
    def create_user_token(self, user_id: int, fingerprint_hash: str, session_id: int) -> str:
        payload = TokenPayload(
            user_id=user_id,
            fingerprint_hash=fingerprint_hash,
            session_id=session_id
        )
        
        jwt_payload = payload.to_jwt_dict(timedelta(seconds=self._cookie_max_age))
        
        return jwt.encode(
            payload=jwt_payload,
            key=self._settings.auth_jwt.private_key_content,
            algorithm=self._settings.algorithm
        )
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(
                jwt=token,
                key=self._settings.auth_jwt.public_key_content,
                algorithms=[self._settings.algorithm]
            )
        except jwt.ExpiredSignatureError:
            log.warning("JWT token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Token expired"
            )
        except jwt.InvalidTokenError as e:
            log.warning(f"Invalid JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )
    
    def validate_payload_structure(self, payload: Dict[str, Any]) -> bool:
        required_fields = {"user_id", "fingerprint_hash", "session_id", "token_type"}
        return required_fields.issubset(payload.keys())
    
    def extract_user_data(self, payload: Dict[str, Any]) -> Dict[str, int]:
        if not self.validate_payload_structure(payload):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload structure"
            )
        return {
            "user_id": int(payload["user_id"]),
            "session_id": int(payload["session_id"])
        }
    
    def verify_fingerprint(self, payload: Dict[str, Any], request: Request) -> bool:
        token_fingerprint = payload.get("fingerprint_hash")
        current_fingerprint = FingerprintCollector.generate_fingerprint_hash(request)
        return token_fingerprint == current_fingerprint
    
    def set_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            key=self._cookie_name,
            value=token,
            max_age=self._cookie_max_age,
            httponly=True,
            secure=True,
            samesite="lax"
        )
    
    def get_token_from_request(self, request: Request) -> Optional[str]:
        return request.cookies.get(self._cookie_name)
    
    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=self._cookie_name,
            httponly=True,
            secure=True,
            samesite="lax"
        )
    
    def refresh_token(self, old_token: str, request: Request) -> str:
        payload = self.decode_token(old_token)
        
        if not self.verify_fingerprint(payload, request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device fingerprint mismatch"
            )
        
        user_data = self.extract_user_data(payload)
        return self.create_user_token(
            user_data["user_id"],
            payload["fingerprint_hash"],
            user_data["session_id"]
        )