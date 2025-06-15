from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from decimal import Decimal
from src.database.my_connector import db
from src.database.models import FrameAnnotations
from src.utils.custom_logging import get_logger

log = get_logger(__name__)


def get_all_annotations() -> List[Dict[str, Any]]:
    query = "SELECT * FROM frame_annotations"
    return db.fetch_all(query)


def get_annotation_by_id(annotation_id: int) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM frame_annotations WHERE id = %s"
    return db.fetch_one(query, (annotation_id,))


def get_annotations_by_video_session(video_session_id: int) -> List[Dict[str, Any]]:
    query = "SELECT * FROM frame_annotations WHERE video_session_id = %s ORDER BY frame_timestamp ASC"
    return db.fetch_all(query, (video_session_id,))


def get_annotations_by_timestamp_range(video_session_id: int, start_time: Decimal, end_time: Decimal) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM frame_annotations
        WHERE video_session_id = %s AND frame_timestamp BETWEEN %s AND %s
        ORDER BY frame_timestamp ASC
    """
    return db.fetch_all(query, (video_session_id, start_time, end_time))


def get_latest_annotations(video_session_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM frame_annotations
        WHERE video_session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    return db.fetch_all(query, (video_session_id, limit))


def create_annotation(annotation: FrameAnnotations) -> int:
    query = """
        INSERT INTO frame_annotations (video_session_id, frame_timestamp, annotation_data)
        VALUES (%s, %s, %s)
    """
    params = (
        annotation.VideoSessionID,
        annotation.FrameTimestamp,
        annotation.AnnotationData
    )
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def create_annotation_batch(annotations: List[FrameAnnotations]) -> List[int]:

    query = """
        INSERT INTO frame_annotations (video_session_id, frame_timestamp, annotation_data)
        VALUES (%s, %s, %s)
    """
    
    params_list = []
    for i, ann in enumerate(annotations):
        try:
            annotation_data_json = json.dumps(ann.AnnotationData, ensure_ascii=False)
            
            params_list.append((
                ann.VideoSessionID, 
                ann.FrameTimestamp, 
                annotation_data_json
            ))
            
        except Exception as e:
            log.error(f"Error processing annotation {i}: {e}")
            log.error(f"Annotation data: {ann.AnnotationData}")
            continue

    inserted_ids = []
    try:
        for params in params_list:
            cursor = db.execute_query(query, params)
            inserted_ids.append(cursor.lastrowid)
        log.info(f"Successfully inserted {len(inserted_ids)} annotations")
    except Exception as e:
        log.error(f"Database error while inserting annotations: {e}")
        log.error(f"Sample params: {params_list[0] if params_list else 'None'}")
        raise
        
    return inserted_ids


def update_annotation(annotation_id: int, updates: Dict[str, Any]) -> None:
    set_clauses = []
    params = []

    field_mapping = {
        "frame_timestamp": "FrameTimestamp",
        "annotation_data": "AnnotationData"
    }

    for db_field, value in updates.items():
        if db_field in field_mapping and value is not None:
            set_clauses.append(f"{db_field} = %s")
            params.append(value)

    if not set_clauses:
        return

    params.append(annotation_id)
    query = f"UPDATE frame_annotations SET {', '.join(set_clauses)} WHERE id = %s"
    db.execute_query(query, params)


def delete_annotation(annotation_id: int) -> None:
    query = "DELETE FROM frame_annotations WHERE id = %s"
    db.execute_query(query, (annotation_id,))


def delete_annotations_by_video_session(video_session_id: int) -> int:
    query = "DELETE FROM frame_annotations WHERE video_session_id = %s"
    cursor = db.execute_query(query, (video_session_id,))
    return cursor.rowcount


def get_annotation_count_by_video_session(video_session_id: int) -> int:
    query = "SELECT COUNT(*) as count FROM frame_annotations WHERE video_session_id = %s"
    result = db.fetch_one(query, (video_session_id,))
    return result["count"] if result else 0


def get_annotations_by_date_range(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM frame_annotations
        WHERE created_at BETWEEN %s AND %s
        ORDER BY created_at DESC
    """
    return db.fetch_all(query, (start_date, end_date))


def get_video_session_timeline(video_session_id: int) -> List[Dict[str, Any]]:
    query = """
        SELECT frame_timestamp, annotation_data, created_at
        FROM frame_annotations
        WHERE video_session_id = %s
        ORDER BY frame_timestamp ASC
    """
    return db.fetch_all(query, (video_session_id,))


def search_annotations_by_content(video_session_id: int, search_term: str) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM frame_annotations
        WHERE video_session_id = %s AND JSON_SEARCH(annotation_data, 'all', %s) IS NOT NULL
        ORDER BY frame_timestamp ASC
    """
    return db.fetch_all(query, (video_session_id, f"%{search_term}%"))