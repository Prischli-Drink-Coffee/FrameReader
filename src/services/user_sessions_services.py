from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from src.repository import user_sessions_repository
from src.database.models import UserSessions
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists
from src.utils.custom_logging import setup_logging

log = setup_logging()


def get_all_sessions() -> List[UserSessions]:
    sessions = user_sessions_repository.get_all_sessions()
    return [UserSessions(**session) for session in sessions]


def get_session_by_id(session_id: int) -> UserSessions:
    session = user_sessions_repository.get_session_by_id(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Session not found'
        )
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


def create_session(session: UserSessions) -> UserSessions:
    # Reverted to original logic, assuming check_if_exists handles it
    check_if_exists(
        get_all=get_all_sessions,
        attr_name="JwtTokenHash",
        attr_value=session.JwtTokenHash,
        exception_detail='Session with this token hash already exists'
    )

    session_id = user_sessions_repository.create_session(session)
    return get_session_by_id(session_id)


def update_session_fields(session_id: int, updates: Dict[str, Any]) -> Dict[str, str]:
    get_session_by_id(session_id) # Check if exists

    allowed_fields = {
        "jwt_token_hash": str,
        "expires_at": datetime,
        "user_agent": str,
        "ip_address": str,
        "is_active": bool,
        "last_activity": datetime
    }

    filtered_updates = {}
    for field, value in updates.items():
        if field in allowed_fields and value is not None:
            if field in ["expires_at", "last_activity"] and isinstance(value, str):
                filtered_updates[field] = datetime.fromisoformat(value)
            else:
                filtered_updates[field] = value

    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )

    user_sessions_repository.update_session(session_id, filtered_updates)
    return {"message": "Session updated successfully"}


def update_session_activity(session_id: int, last_activity: datetime) -> Dict[str, str]:
    get_session_by_id(session_id) # Check if exists
    user_sessions_repository.update_session_activity(session_id, last_activity)
    return {"message": "Session activity updated successfully"}


def deactivate_session(session_id: int) -> Dict[str, str]:
    get_session_by_id(session_id) # Check if exists
    user_sessions_repository.deactivate_session(session_id)
    return {"message": "Session deactivated successfully"}


def deactivate_user_sessions(user_id: int) -> Dict[str, str]:
    user_sessions_repository.deactivate_user_sessions(user_id)
    return {"message": "All user sessions deactivated successfully"}


def delete_session(session_id: int) -> Dict[str, str]:
    get_session_by_id(session_id) # Check if exists
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

    latest_activity = None
    if all_sessions:
        # Reverted to original logic, assuming LastActivity is always present
        latest_activity = max(
            session.LastActivity for session in all_sessions
            if session.LastActivity
        )

    return {
        "user_id": user_id,
        "total_sessions": total_count,
        "active_sessions": active_count,
        "expired_sessions": expired_count,
        "latest_activity": latest_activity
    }


def extend_session(session_id: int, extend_days: int = 30) -> Dict[str, str]:
    session = get_session_by_id(session_id)
    new_expires_at = datetime.utcnow() + timedelta(days=extend_days)

    user_sessions_repository.update_session(
        session_id,
        {"expires_at": new_expires_at}
    )

    return {"message": f"Session extended by {extend_days} days"}


def validate_active_session(session_id: int) -> bool:
    try:
        session = get_session_by_id(session_id)
        return (
            session.IsActive and
            session.ExpiresAt > datetime.utcnow()
        )
    except HTTPException:
        return False


def get_session_security_info(session_id: int) -> Dict[str, Any]:
    session = get_session_by_id(session_id)

    return {
        "session_id": session.ID,
        "user_id": session.UserID,
        "created_at": session.CreatedAt,
        "last_activity": session.LastActivity,
        "expires_at": session.ExpiresAt,
        "user_agent": session.UserAgent,
        "ip_address": session.IPAddress,
        "is_active": session.IsActive,
        "is_expired": session.ExpiresAt < datetime.utcnow() if session.ExpiresAt else False
    }