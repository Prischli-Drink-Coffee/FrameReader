from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import HTTPException, status

from src.repository import user_sessions_repository
from src.database.models import UserSessions
from src.utils.exam_services import check_if_exists
from src.utils.custom_logging import get_logger

log = get_logger(__name__)


class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Session {session_id} not found'
        )


class SessionValidationError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )


def get_all_sessions() -> List[UserSessions]:
    sessions = user_sessions_repository.get_all_sessions()
    return [UserSessions(**session) for session in sessions]


def get_session_by_id(session_id: int) -> UserSessions:
    session = user_sessions_repository.get_session_by_id(session_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return UserSessions(**session)


def get_session_by_token_hash(token_hash: str) -> Optional[UserSessions]:
    session = user_sessions_repository.get_session_by_token_hash(token_hash)
    return UserSessions(**session) if session else None


def get_active_sessions_by_user(user_id: int) -> List[UserSessions]:
    sessions = user_sessions_repository.get_active_sessions_by_user(user_id)
    return [UserSessions(**session) for session in sessions]


def get_sessions_by_user(user_id: int) -> List[UserSessions]:
    sessions = user_sessions_repository.get_sessions_by_user(user_id)
    return [UserSessions(**session) for session in sessions]


def create_session(session: UserSessions) -> int:
    _validate_unique_jwt_hash(session.JwtTokenHash)
    return user_sessions_repository.create_session(session)


def create_user_session(
    user_id: int,
    jwt_token_hash: str,
    expires_at: datetime,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
) -> int:
    _validate_unique_jwt_hash(jwt_token_hash)
    
    current_time = datetime.utcnow()
    session = UserSessions(
        user_id=user_id,
        jwt_token_hash=jwt_token_hash,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address,
        is_active=1,
        created_at=current_time
    )
    
    return user_sessions_repository.create_session(session)


def _validate_unique_jwt_hash(jwt_token_hash: str) -> None:
    check_if_exists(
        get_all=get_all_sessions,
        attr_name="JwtTokenHash",
        attr_value=jwt_token_hash,
        exception_detail='Session with this token hash already exists'
    )


def update_session_fields(session_id: int, updates: Dict[str, Any]) -> Dict[str, str]:
    _ensure_session_exists(session_id)
    
    allowed_updates = _filter_allowed_updates(updates)
    if not allowed_updates:
        raise SessionValidationError("No valid fields to update")

    user_sessions_repository.update_session(session_id, allowed_updates)
    return {"message": "Session updated successfully"}


def _ensure_session_exists(session_id: int) -> None:
    get_session_by_id(session_id)


def _filter_allowed_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    allowed_fields = {
        "jwt_token_hash": str,
        "expires_at": datetime,
        "user_agent": str,
        "ip_address": str,
        "is_active": int
    }

    filtered_updates = {}
    for field, value in updates.items():
        if field in allowed_fields and value is not None:
            if field == "expires_at" and isinstance(value, str):
                filtered_updates[field] = datetime.fromisoformat(value)
            else:
                filtered_updates[field] = value

    return filtered_updates


def deactivate_session(session_id: int) -> Dict[str, str]:
    _ensure_session_exists(session_id)
    user_sessions_repository.deactivate_session(session_id)
    return {"message": "Session deactivated successfully"}


def deactivate_user_sessions(user_id: int) -> Dict[str, str]:
    user_sessions_repository.deactivate_user_sessions(user_id)
    return {"message": "All user sessions deactivated successfully"}


def delete_session(session_id: int) -> Dict[str, str]:
    _ensure_session_exists(session_id)
    user_sessions_repository.delete_session(session_id)
    return {"message": "Session deleted successfully"}


def cleanup_expired_sessions() -> Dict[str, Any]:
    deleted_count = user_sessions_repository.delete_expired_sessions()
    return {
        "message": "Expired sessions cleaned up successfully",
        "deleted_count": deleted_count
    }


def get_expired_sessions() -> List[UserSessions]:
    sessions = user_sessions_repository.get_expired_sessions()
    return [UserSessions(**session) for session in sessions]


def get_sessions_by_ip(ip_address: str) -> List[UserSessions]:
    sessions = user_sessions_repository.get_sessions_by_ip(ip_address)
    return [UserSessions(**session) for session in sessions]


def get_user_session_statistics(user_id: int) -> Dict[str, Any]:
    active_count = user_sessions_repository.count_active_sessions_by_user(user_id)
    all_sessions = get_sessions_by_user(user_id)

    total_count = len(all_sessions)
    expired_count = total_count - active_count

    return {
        "user_id": user_id,
        "total_sessions": total_count,
        "active_sessions": active_count,
        "expired_sessions": expired_count
    }


def extend_session(session_id: int, extend_days: int = 30) -> Dict[str, str]:
    _ensure_session_exists(session_id)
    new_expires_at = datetime.utcnow() + timedelta(days=extend_days)

    user_sessions_repository.update_session(
        session_id,
        {"expires_at": new_expires_at}
    )

    return {"message": f"Session extended by {extend_days} days"}


def validate_active_session(session_id: int) -> bool:
    try:
        session = get_session_by_id(session_id)
        return bool(session.IsActive and session.ExpiresAt > datetime.utcnow())
    except HTTPException:
        return False


def get_session_security_info(session_id: int) -> Dict[str, Any]:
    session = get_session_by_id(session_id)

    return {
        "session_id": session.ID,
        "user_id": session.UserID,
        "created_at": session.CreatedAt,
        "expires_at": session.ExpiresAt,
        "user_agent": session.UserAgent,
        "ip_address": session.IPAddress,
        "is_active": session.IsActive,
        "is_expired": session.ExpiresAt < datetime.utcnow() if session.ExpiresAt else False
    }