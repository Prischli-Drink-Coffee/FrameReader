from typing import Optional, Dict, Any, List
from datetime import datetime
from src.database.my_connector import db
from src.database.models import UserSessions


def get_all_sessions() -> List[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions"
    return db.fetch_all(query)


def get_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE id = %s"
    return db.fetch_one(query, (session_id,))


def get_session_by_token_hash(token_hash: str) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE jwt_token_hash = %s"
    return db.fetch_one(query, (token_hash,))


def get_active_sessions_by_user(user_id: int) -> List[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE user_id = %s AND is_active = TRUE AND expires_at > NOW()"
    return db.fetch_all(query, (user_id,))


def get_sessions_by_user(user_id: int) -> List[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE user_id = %s ORDER BY created_at DESC"
    return db.fetch_all(query, (user_id,))


def create_session(session: UserSessions) -> int:
    query = """
        INSERT INTO user_sessions (user_id, jwt_token_hash, expires_at, user_agent, ip_address, is_active)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = (
        session.UserID,
        session.JwtTokenHash,
        session.ExpiresAt,
        session.UserAgent,
        session.IPAddress,
        session.IsActive
    )
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_session(session_id: int, updates: Dict[str, Any]) -> None:
    set_clauses = []
    params = []

    field_mapping = {
        "jwt_token_hash": "JwtTokenHash",
        "expires_at": "ExpiresAt",
        "user_agent": "UserAgent",
        "ip_address": "IPAddress",
        "is_active": "IsActive",
        "last_activity": "LastActivity"
    }

    for db_field, value in updates.items():
        if db_field in field_mapping and value is not None:
            set_clauses.append(f"{db_field} = %s")
            params.append(value)

    if not set_clauses:
        return

    params.append(session_id)
    query = f"UPDATE user_sessions SET {', '.join(set_clauses)} WHERE id = %s"
    db.execute_query(query, params)


def update_session_activity(session_id: int, last_activity: datetime) -> None:
    query = "UPDATE user_sessions SET last_activity = %s WHERE id = %s"
    db.execute_query(query, (last_activity, session_id))


def deactivate_session(session_id: int) -> None:
    query = "UPDATE user_sessions SET is_active = FALSE WHERE id = %s"
    db.execute_query(query, (session_id,))


def deactivate_user_sessions(user_id: int) -> None:
    query = "UPDATE user_sessions SET is_active = FALSE WHERE user_id = %s"
    db.execute_query(query, (user_id,))


def delete_session(session_id: int) -> None:
    query = "DELETE FROM user_sessions WHERE id = %s"
    db.execute_query(query, (session_id,))


def delete_expired_sessions() -> int:
    query = "DELETE FROM user_sessions WHERE expires_at < NOW()"
    cursor = db.execute_query(query)
    return cursor.rowcount


def get_expired_sessions() -> List[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE expires_at < NOW()"
    return db.fetch_all(query)


def get_sessions_by_ip(ip_address: str) -> List[Dict[str, Any]]:
    query = "SELECT * FROM user_sessions WHERE ip_address = %s ORDER BY created_at DESC"
    return db.fetch_all(query, (ip_address,))


def count_active_sessions_by_user(user_id: int) -> int:
    query = "SELECT COUNT(*) as count FROM user_sessions WHERE user_id = %s AND is_active = TRUE AND expires_at > NOW()"
    result = db.fetch_one(query, (user_id,))
    return result["count"] if result else 0