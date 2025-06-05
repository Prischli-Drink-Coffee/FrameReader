from typing import Optional, Dict, Any, List
from datetime import datetime
from src.database.my_connector import db
from src.database.models import VideoSessions, ProcessingStatus


def get_all_video_sessions() -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions"
    return db.fetch_all(query)


def get_video_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE id = %s"
    return db.fetch_one(query, (session_id,))


def get_video_sessions_by_user(user_id: int) -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE user_id = %s ORDER BY started_at DESC"
    return db.fetch_all(query, (user_id,))


def get_video_sessions_by_status(status: ProcessingStatus) -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE processing_status = %s"
    return db.fetch_all(query, (status.value,))


def get_video_sessions_by_user_and_status(user_id: int, status: ProcessingStatus) -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE user_id = %s AND processing_status = %s ORDER BY started_at DESC"
    return db.fetch_all(query, (user_id, status.value))


def get_processing_sessions() -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE processing_status = 'processing'"
    return db.fetch_all(query)


def create_video_session(session: VideoSessions) -> int:
    query = """
        INSERT INTO video_sessions (user_id, video_url, processing_status, started_at)
        VALUES (%s, %s, %s, %s)
    """
    params = (
        session.UserID,
        session.VideoURL,
        session.ProcessingStatus.value,
        session.StartedAt or datetime.utcnow()
    )
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_video_session(session_id: int, updates: Dict[str, Any]) -> None:
    set_clauses = []
    params = []
    
    field_mapping = {
        "video_url": "VideoURL",
        "processing_status": "ProcessingStatus",
        "started_at": "StartedAt",
        "completed_at": "CompletedAt"
    }
    
    for db_field, value in updates.items():
        if db_field in field_mapping and value is not None:
            set_clauses.append(f"{db_field} = %s")
            if db_field == "processing_status" and hasattr(value, 'value'):
                params.append(value.value)
            else:
                params.append(value)
    
    if not set_clauses:
        return
        
    params.append(session_id)
    query = f"UPDATE video_sessions SET {', '.join(set_clauses)} WHERE id = %s"
    db.execute_query(query, params)


def update_session_status(session_id: int, status: ProcessingStatus, completed_at: Optional[datetime] = None) -> None:
    if status == ProcessingStatus.COMPLETED and completed_at is None:
        completed_at = datetime.utcnow()
    
    if completed_at:
        query = "UPDATE video_sessions SET processing_status = %s, completed_at = %s WHERE id = %s"
        params = (status.value, completed_at, session_id)
    else:
        query = "UPDATE video_sessions SET processing_status = %s WHERE id = %s"
        params = (status.value, session_id)
    
    db.execute_query(query, params)


def complete_video_session(session_id: int) -> None:
    query = "UPDATE video_sessions SET processing_status = 'completed', completed_at = %s WHERE id = %s"
    db.execute_query(query, (datetime.utcnow(), session_id))


def fail_video_session(session_id: int) -> None:
    query = "UPDATE video_sessions SET processing_status = 'failed', completed_at = %s WHERE id = %s"
    db.execute_query(query, (datetime.utcnow(), session_id))


def delete_video_session(session_id: int) -> None:
    query = "DELETE FROM video_sessions WHERE id = %s"
    db.execute_query(query, (session_id,))


def get_user_session_count(user_id: int) -> int:
    query = "SELECT COUNT(*) as count FROM video_sessions WHERE user_id = %s"
    result = db.fetch_one(query, (user_id,))
    return result["count"] if result else 0


def get_sessions_by_date_range(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    query = "SELECT * FROM video_sessions WHERE started_at BETWEEN %s AND %s ORDER BY started_at DESC"
    return db.fetch_all(query, (start_date, end_date))


def get_completed_sessions_by_user(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM video_sessions 
        WHERE user_id = %s AND processing_status = 'completed' 
        ORDER BY completed_at DESC 
        LIMIT %s
    """
    return db.fetch_all(query, (user_id, limit))