from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from src.repository import video_sessions_repository
from src.database.models import VideoSessions, ProcessingStatusEnum
from fastapi import HTTPException, status
import json
from src.utils.custom_logging import setup_logging

log = setup_logging()


def get_all_video_sessions() -> List[VideoSessions]:
    sessions = video_sessions_repository.get_all_video_sessions()
    return [VideoSessions(**session) for session in sessions]


def get_video_session_by_id(session_id: int) -> VideoSessions:
    session = video_sessions_repository.get_video_session_by_id(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Video session not found'
        )
    return VideoSessions(**session)


def get_video_sessions_by_user(user_id: int) -> List[VideoSessions]:
    sessions = video_sessions_repository.get_video_sessions_by_user(user_id)
    return [VideoSessions(**session) for session in sessions]


def get_video_sessions_by_status(processing_status: ProcessingStatusEnum) -> List[VideoSessions]:
    sessions = video_sessions_repository.get_video_sessions_by_status(processing_status)
    return [VideoSessions(**session) for session in sessions]


def get_video_sessions_by_user_and_status(user_id: int, processing_status: ProcessingStatusEnum) -> List[VideoSessions]:
    sessions = video_sessions_repository.get_video_sessions_by_user_and_status(user_id, processing_status)
    return [VideoSessions(**session) for session in sessions]


def get_processing_sessions() -> List[VideoSessions]:
    sessions = video_sessions_repository.get_processing_sessions()
    return [VideoSessions(**session) for session in sessions]


def create_video_session(user_id: int, video_url: str) -> VideoSessions:
    session = VideoSessions(
        user_id=user_id,
        video_url=video_url,
        processing_status=ProcessingStatusEnum.PROCESSING,
        started_at=datetime.utcnow()
    )
    session_id = video_sessions_repository.create_video_session(session)
    return get_video_session_by_id(session_id)


def update_video_session_fields(session_id: int, updates: Dict[str, Any]) -> Dict[str, str]:
    get_video_session_by_id(session_id)
    allowed_fields = {
        "video_url": str,
        "processing_status": ProcessingStatusEnum,
        "started_at": datetime,
        "completed_at": datetime
    }
    filtered_updates = {}
    for field, value in updates.items():
        if field in allowed_fields and value is not None:
            if field in ["started_at", "completed_at"] and isinstance(value, str):
                filtered_updates[field] = datetime.fromisoformat(value)
            else:
                filtered_updates[field] = value

    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )
    video_sessions_repository.update_video_session(session_id, filtered_updates)
    return {"message": "Video session updated successfully"}


def update_session_status(session_id: int, status: Union[ProcessingStatusEnum, str], completed_at: Optional[datetime] = None) -> Dict[str, str]:
    get_video_session_by_id(session_id)
    video_sessions_repository.update_session_status(session_id, status, completed_at)
    status_value = status.value if hasattr(status, 'value') else status
    return {"message": f"Session status updated to {status_value}"}


def complete_video_session(session_id: int) -> Dict[str, str]:
    session = get_video_session_by_id(session_id)
    if session.ProcessingStatus == ProcessingStatusEnum.COMPLETED:
        return {"message": "Session already completed"}
    video_sessions_repository.complete_video_session(session_id)
    from src.services import user_services
    user_services.increment_user_videos(session.UserID)
    return {"message": "Video session completed successfully"}


def fail_video_session(session_id: int, error_message: Optional[str] = None) -> Dict[str, str]:
    session = get_video_session_by_id(session_id)
    if session.ProcessingStatus == ProcessingStatusEnum.FAILED:
        return {"message": "Session already marked as failed"}
    video_sessions_repository.fail_video_session(session_id)
    log.error(f"Video session {session_id} failed: {error_message or 'Unknown error'}")
    return {"message": "Video session marked as failed"}


def delete_video_session(session_id: int) -> Dict[str, str]:
    get_video_session_by_id(session_id)
    video_sessions_repository.delete_video_session(session_id)
    return {"message": "Video session deleted successfully"}


def get_user_video_history(user_id: int, limit: int = 10) -> List[VideoSessions]:
    sessions = video_sessions_repository.get_completed_sessions_by_user(user_id, limit)
    return [VideoSessions(**session) for session in sessions]


def get_user_session_statistics(user_id: int) -> Dict[str, Any]:
    total_count = video_sessions_repository.get_user_session_count(user_id)
    processing_sessions = get_video_sessions_by_user_and_status(user_id, ProcessingStatusEnum.PROCESSING)
    completed_sessions = get_video_sessions_by_user_and_status(user_id, ProcessingStatusEnum.COMPLETED)
    failed_sessions = get_video_sessions_by_user_and_status(user_id, ProcessingStatusEnum.FAILED)
    return {
        "user_id": user_id,
        "total_sessions": total_count,
        "processing_count": len(processing_sessions),
        "completed_count": len(completed_sessions),
        "failed_count": len(failed_sessions),
        "success_rate": len(completed_sessions) / total_count * 100 if total_count > 0 else 0
    }


def get_sessions_by_date_range(start_date: datetime, end_date: datetime) -> List[VideoSessions]:
    sessions = video_sessions_repository.get_sessions_by_date_range(start_date, end_date)
    return [VideoSessions(**session) for session in sessions]


def get_recent_sessions(days: int = 7) -> List[VideoSessions]:
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()
    return get_sessions_by_date_range(start_date, end_date)


def cleanup_old_failed_sessions(days_old: int = 30) -> Dict[str, Any]:
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    old_failed_sessions = get_video_sessions_by_status(ProcessingStatusEnum.FAILED)
    deleted_count = 0
    for session in old_failed_sessions:
        if session.StartedAt and session.StartedAt < cutoff_date:
            video_sessions_repository.delete_video_session(session.ID)
            deleted_count += 1
    return {
        "message": f"Cleaned up {deleted_count} old failed sessions",
        "deleted_count": deleted_count
    }


def get_session_duration(session_id: int) -> Optional[timedelta]:
    session = get_video_session_by_id(session_id)
    if not session.StartedAt:
        return None
    end_time = session.CompletedAt or datetime.utcnow()
    return end_time - session.StartedAt


def get_processing_queue_status() -> Dict[str, Any]:
    processing_sessions = get_processing_sessions()
    return {
        "queue_length": len(processing_sessions),
        "oldest_session": min(
            (session.StartedAt for session in processing_sessions if session.StartedAt),
            default=None
        ),
        "sessions": [
            {
                "id": session.ID,
                "user_id": session.UserID,
                "video_url": session.VideoURL,
                "started_at": session.StartedAt
            }
            for session in processing_sessions
        ]
    }