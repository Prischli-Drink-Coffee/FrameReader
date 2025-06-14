from typing import Dict, Any
from datetime import datetime, timedelta
import hashlib

from fastapi import Request, HTTPException, status, Response

from src.jwt_cookie.session_manager import JWTCookieManager, FingerprintCollector
from src.services import user_services, user_sessions_services


class UserSessionManager:
    def __init__(self) -> None:
        self._jwt_manager = JWTCookieManager()
        self._fingerprint_collector = FingerprintCollector()
        self._session_expiry_days = 30

    def create_or_get_user_session(self, request: Request, response: Response) -> Dict[str, Any]:
        fingerprint_hash = self._fingerprint_collector.generate_fingerprint_hash(request)
        existing_token = self._jwt_manager.get_token_from_request(request)

        if existing_token:
            try:
                payload = self._jwt_manager.decode_token(existing_token)
                if self._is_valid_fingerprint(payload, fingerprint_hash):
                    return self._refresh_existing_session(payload, response, request)
            except HTTPException:
                pass

        return self._create_new_session(request, response, fingerprint_hash)

    def _is_valid_fingerprint(self, payload: Dict[str, Any], fingerprint_hash: str) -> bool:
        return payload.get("fingerprint_hash") == fingerprint_hash

    def _refresh_existing_session(
        self, 
        payload: Dict[str, Any], 
        response: Response, 
        request: Request
    ) -> Dict[str, Any]:
        user_id = payload["user_id"]
        session_id = payload["session_id"]
        fingerprint_hash = payload["fingerprint_hash"]

        user_services.update_user_activity(user_id)

        new_token = self._jwt_manager.create_user_token(user_id, fingerprint_hash, session_id)
        self._jwt_manager.set_cookie(response, new_token)

        return {
            "user_id": user_id,
            "session_id": session_id,
            "is_new_user": False
        }

    def _create_new_session(
        self, 
        request: Request, 
        response: Response, 
        fingerprint_hash: str
    ) -> Dict[str, Any]:
        user = user_services.get_or_create_user(fingerprint_hash)
        jwt_token_hash = self._generate_jwt_hash(user.ID)

        session_id = user_sessions_services.create_user_session(
            user_id=user.ID,
            jwt_token_hash=jwt_token_hash,
            expires_at=self._calculate_expiry_date(),
            user_agent=request.headers.get("user-agent"),
            ip_address=self._extract_client_ip(request)
        )

        token = self._jwt_manager.create_user_token(user.ID, fingerprint_hash, session_id)
        self._jwt_manager.set_cookie(response, token)

        return {
            "user_id": user.ID,
            "session_id": session_id,
            "is_new_user": user.TotalSessions == 0
        }

    def _generate_jwt_hash(self, user_id: int) -> str:
        timestamp = datetime.utcnow().timestamp()
        hash_input = f"{user_id}_{timestamp}".encode()
        return hashlib.sha256(hash_input).hexdigest()

    def _calculate_expiry_date(self) -> datetime:
        return datetime.utcnow() + timedelta(days=self._session_expiry_days)

    def _extract_client_ip(self, request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def get_current_user_from_request(self, request: Request) -> Dict[str, Any]:
        token = self._jwt_manager.get_token_from_request(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No session token"
            )

        payload = self._jwt_manager.decode_token(token)
        self._validate_session_fingerprint(payload, request)
        
        session = self._get_active_session(payload["session_id"])
        
        return {
            "user_id": payload["user_id"],
            "session_id": payload["session_id"],
            "fingerprint_hash": payload["fingerprint_hash"]
        }

    def _validate_session_fingerprint(self, payload: Dict[str, Any], request: Request) -> None:
        current_fingerprint = self._fingerprint_collector.generate_fingerprint_hash(request)
        if payload.get("fingerprint_hash") != current_fingerprint:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )

    def _get_active_session(self, session_id: int):
        session = user_sessions_services.get_session_by_id(session_id)
        if not session or not session.IsActive:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid"
            )
        return session

    def logout_user(self, request: Request, response: Response) -> Dict[str, str]:
        try:
            current_user = self.get_current_user_from_request(request)
            user_sessions_services.deactivate_session(current_user["session_id"])
            self._jwt_manager.clear_cookie(response)
            return {"message": "Successfully logged out"}
        except HTTPException:
            self._jwt_manager.clear_cookie(response)
            return {"message": "Logged out"}


session_manager = UserSessionManager()