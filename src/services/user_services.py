from typing import Optional, Dict, Any, List
from datetime import datetime
from src.repository import user_repository
from src.database.models import Users
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists
from src.utils.custom_logging import get_logger

log = get_logger(__name__)


def get_all_users() -> List[Users]:
    users = user_repository.get_all_users()
    return [Users(**user) for user in users]


def get_user_by_id(user_id: int) -> Users:
    user = user_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User not found'
        )
    return Users(**user)


def get_user_by_fingerprint(fingerprint_hash: str) -> Optional[Users]:
    user = user_repository.get_user_by_fingerprint(fingerprint_hash)
    return Users(**user) if user else None


def create_user(fingerprint_hash: str) -> Users:
    # Reverted to original logic, assuming check_if_exists handles it
    check_if_exists(
        get_all=get_all_users,
        attr_name="FingerprintHash",
        attr_value=fingerprint_hash,
        exception_detail='User with this fingerprint already exists'
    )

    current_time = datetime.utcnow()
    user = Users(
        fingerprint_hash=fingerprint_hash,
        first_visit=current_time,
        last_activity=current_time,
        total_sessions=0,
        total_videos_processed=0
    )

    user_id = user_repository.create_user(user)
    return get_user_by_id(user_id)


def get_or_create_user(fingerprint_hash: str) -> Users:
    existing_user = get_user_by_fingerprint(fingerprint_hash)
    if existing_user:
        return existing_user
    return create_user(fingerprint_hash)


def update_user_activity(user_id: int) -> Dict[str, str]:
    get_user_by_id(user_id)
    user_repository.update_user_activity(user_id, datetime.utcnow())
    return {"message": "User activity updated successfully"}


def increment_user_videos(user_id: int) -> Dict[str, str]:
    get_user_by_id(user_id)
    user_repository.increment_videos_processed(user_id)
    return {"message": "User video count incremented successfully"}


def update_user_fields(user_id: int, updates: Dict[str, Any]) -> Dict[str, str]:
    get_user_by_id(user_id)

    allowed_fields = {
        "last_activity": datetime,
        "total_sessions": int,
        "total_videos_processed": int
    }

    filtered_updates = {}
    for field, value in updates.items():
        if field in allowed_fields:
            if field == "last_activity" and isinstance(value, str):
                filtered_updates[field] = datetime.fromisoformat(value)
            else:
                filtered_updates[field] = value

    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )

    user_repository.update_user(user_id, filtered_updates)
    return {"message": "User updated successfully"}


def delete_user(user_id: int) -> Dict[str, str]:
    get_user_by_id(user_id)
    user_repository.delete_user(user_id)
    return {"message": "User deleted successfully"}


def get_active_users() -> List[Users]:
    users = user_repository.get_users_by_activity_period(
        datetime.utcnow().replace(day=1),
        datetime.utcnow()
    )
    return [Users(**user) for user in users]


def get_user_statistics(user_id: int) -> Dict[str, Any]:
    user = get_user_by_id(user_id)
    return {
        "user_id": user.ID,
        "total_sessions": user.TotalSessions,
        "total_videos_processed": user.TotalVideosProcessed,
        "first_visit": user.FirstVisit,
        "last_activity": user.LastActivity,
        "days_since_first_visit": (datetime.utcnow() - user.FirstVisit).days
    }


def get_users_by_activity_range(start_date: datetime, end_date: datetime) -> List[Users]:
    users = user_repository.get_users_by_activity_period(start_date, end_date)
    return [Users(**user) for user in users]


def get_active_users_count() -> int:
    return user_repository.get_active_users_count()