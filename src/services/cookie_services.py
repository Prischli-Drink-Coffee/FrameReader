from src.jwt_cookie.session_manager import JWTCookieManager
from src.fingerprint_collector import FingerprintCollector
from fastapi import Request, HTTPException, status, Response
from src.services import user_services, user_sessions_services
from src.database.models import UserSessions


class UserSessionManager:
    def __init__(self):
        self.jwt_manager = JWTCookieManager()
        self.fingerprint_collector = FingerprintCollector()
    
    def create_or_get_user_session(self, request: Request, response: Response) -> Dict:
        fingerprint_hash = self.fingerprint_collector.generate_fingerprint_hash(request)
        existing_token = self.jwt_manager.get_token_from_request(request)
        
        if existing_token:
            try:
                payload = self.jwt_manager.decode_token(existing_token)
                if payload.get("fingerprint_hash") == fingerprint_hash:
                    return self._refresh_existing_session(payload, response, request)
            except HTTPException:
                pass
        
        return self._create_new_session(request, response, fingerprint_hash)
    
    def _refresh_existing_session(self, payload: Dict, response: Response, request: Request) -> Dict:
        
        user_id = payload.get("user_id")
        session_id = payload.get("session_id")
        fingerprint_hash = payload.get("fingerprint_hash")
        
        user_services.update_user_activity(user_id)
        user_sessions_services.update_session_activity(session_id, datetime.utcnow())
        
        new_token = self.jwt_manager.create_user_token(user_id, fingerprint_hash, session_id)
        self.jwt_manager.set_cookie(response, new_token)
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "is_new_user": False
        }
    
    def _create_new_session(self, request: Request, response: Response, fingerprint_hash: str) -> Dict:
        
        user = user_services.get_or_create_user(fingerprint_hash)
        
        jwt_token_hash = hashlib.sha256(
            f"{user.ID}_{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()
        
        session_data = UserSessions(
            UserID=user.ID,
            JwtTokenHash=jwt_token_hash,
            ExpiresAt=datetime.utcnow() + timedelta(days=30),
            UserAgent=request.headers.get("user-agent"),
            IPAddress=request.client.host if request.client else None,
            IsActive=True
        )
        
        session_id = user_sessions_services.create_session(session_data)
        
        token = self.jwt_manager.create_user_token(user.ID, fingerprint_hash, session_id)
        self.jwt_manager.set_cookie(response, token)
        
        is_new_user = user.TotalSessions == 0
        
        return {
            "user_id": user.ID,
            "session_id": session_id,
            "is_new_user": is_new_user
        }
    
    def get_current_user_from_request(self, request: Request) -> Dict:
        token = self.jwt_manager.get_token_from_request(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="No session token"
            )
        
        payload = self.jwt_manager.decode_token(token)
        fingerprint_hash = self.fingerprint_collector.generate_fingerprint_hash(request)
        
        if payload.get("fingerprint_hash") != fingerprint_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid session"
            )
        
        from src.service import user_sessions_services
        
        session_id = payload.get("session_id")
        session = user_sessions_services.get_session_by_id(session_id)
        
        if not session or not session.IsActive:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Session expired or invalid"
            )
        
        return {
            "user_id": payload.get("user_id"),
            "session_id": session_id,
            "fingerprint_hash": fingerprint_hash
        }
    
    def logout_user(self, request: Request, response: Response) -> Dict:
        try:
            current_user = self.get_current_user_from_request(request)
            
            from src.service import user_sessions_services
            user_sessions_services.deactivate_session(current_user["session_id"])
            
            self.jwt_manager.clear_cookie(response)
            
            return {"message": "Successfully logged out"}
        except HTTPException:
            self.jwt_manager.clear_cookie(response)
            return {"message": "Logged out"}


session_manager = UserSessionManager()