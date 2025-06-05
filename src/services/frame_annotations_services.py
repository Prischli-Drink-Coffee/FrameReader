from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal
from src.repository import frame_annotations_repository
from src.database.models import FrameAnnotations
from fastapi import HTTPException, status
from src.utils.custom_logging import setup_logging

log = setup_logging()


def get_all_annotations() -> List[FrameAnnotations]:
    annotations = frame_annotations_repository.get_all_annotations()
    return [FrameAnnotations(**annotation) for annotation in annotations]


def get_annotation_by_id(annotation_id: int) -> FrameAnnotations:
    annotation = frame_annotations_repository.get_annotation_by_id(annotation_id)
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Frame annotation not found'
        )
    return FrameAnnotations(**annotation)


def get_annotations_by_video_session(video_session_id: int) -> List[FrameAnnotations]:
    annotations = frame_annotations_repository.get_annotations_by_video_session(video_session_id)
    return [FrameAnnotations(**annotation) for annotation in annotations]


def get_annotations_by_timestamp_range(
    video_session_id: int, 
    start_time: Decimal, 
    end_time: Decimal
) -> List[FrameAnnotations]:
    annotations = frame_annotations_repository.get_annotations_by_timestamp_range(
        video_session_id, start_time, end_time
    )
    return [FrameAnnotations(**annotation) for annotation in annotations]


def get_latest_annotations(video_session_id: int, limit: int = 10) -> List[FrameAnnotations]:
    annotations = frame_annotations_repository.get_latest_annotations(video_session_id, limit)
    return [FrameAnnotations(**annotation) for annotation in annotations]


def create_annotation(
    video_session_id: int, 
    frame_timestamp: Decimal, 
    annotation_data: Dict[str, Any]
) -> FrameAnnotations:
    annotation = FrameAnnotations(
        VideoSessionID=video_session_id,
        FrameTimestamp=frame_timestamp,
        AnnotationData=annotation_data
    )
    
    annotation_id = frame_annotations_repository.create_annotation(annotation)
    return get_annotation_by_id(annotation_id)


def create_annotation_batch(annotations: List[FrameAnnotations]) -> List[FrameAnnotations]:
    if not annotations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Annotations list cannot be empty"
        )
    
    annotation_ids = frame_annotations_repository.create_annotation_batch(annotations)
    return [get_annotation_by_id(annotation_id) for annotation_id in annotation_ids]


def update_annotation_fields(annotation_id: int, updates: Dict[str, Any]) -> Dict[str, str]:
    get_annotation_by_id(annotation_id)
    
    allowed_fields = {
        "frame_timestamp": Decimal,
        "annotation_data": dict
    }
    
    filtered_updates = {}
    for field, value in updates.items():
        if field in allowed_fields and value is not None:
            if field == "frame_timestamp" and isinstance(value, (int, float, str)):
                filtered_updates[field] = Decimal(str(value))
            else:
                filtered_updates[field] = value
    
    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )
    
    frame_annotations_repository.update_annotation(annotation_id, filtered_updates)
    return {"message": "Frame annotation updated successfully"}


def delete_annotation(annotation_id: int) -> Dict[str, str]:
    get_annotation_by_id(annotation_id)
    frame_annotations_repository.delete_annotation(annotation_id)
    return {"message": "Frame annotation deleted successfully"}


def delete_annotations_by_video_session(video_session_id: int) -> Dict[str, Any]:
    deleted_count = frame_annotations_repository.delete_annotations_by_video_session(video_session_id)
    return {
        "message": "Video session annotations deleted successfully",
        "deleted_count": deleted_count
    }


def get_video_session_statistics(video_session_id: int) -> Dict[str, Any]:
    total_count = frame_annotations_repository.get_annotation_count_by_video_session(video_session_id)
    annotations = get_annotations_by_video_session(video_session_id)
    
    if not annotations:
        return {
            "video_session_id": video_session_id,
            "total_annotations": 0,
            "duration": None,
            "average_interval": None
        }
    
    timestamps = [ann.FrameTimestamp for ann in annotations]
    duration = max(timestamps) - min(timestamps)
    average_interval = duration / (len(timestamps) - 1) if len(timestamps) > 1 else None
    
    return {
        "video_session_id": video_session_id,
        "total_annotations": total_count,
        "duration": float(duration),
        "average_interval": float(average_interval) if average_interval else None,
        "first_timestamp": float(min(timestamps)),
        "last_timestamp": float(max(timestamps))
    }


def get_annotations_by_date_range(start_date: datetime, end_date: datetime) -> List[FrameAnnotations]:
    annotations = frame_annotations_repository.get_annotations_by_date_range(start_date, end_date)
    return [FrameAnnotations(**annotation) for annotation in annotations]


def get_video_session_timeline(video_session_id: int) -> List[Dict[str, Any]]:
    timeline_data = frame_annotations_repository.get_video_session_timeline(video_session_id)
    return timeline_data


def search_annotations_by_content(video_session_id: int, search_term: str) -> List[FrameAnnotations]:
    if not search_term.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search term cannot be empty"
        )
    
    annotations = frame_annotations_repository.search_annotations_by_content(video_session_id, search_term)
    return [FrameAnnotations(**annotation) for annotation in annotations]


def get_annotation_density_analysis(video_session_id: int, interval_seconds: int = 10) -> List[Dict[str, Any]]:
    annotations = get_annotations_by_video_session(video_session_id)
    
    if not annotations:
        return []
    
    interval_decimal = Decimal(str(interval_seconds))
    min_timestamp = min(ann.FrameTimestamp for ann in annotations)
    max_timestamp = max(ann.FrameTimestamp for ann in annotations)
    
    intervals = []
    current_start = min_timestamp
    
    while current_start <= max_timestamp:
        current_end = current_start + interval_decimal
        count = sum(
            1 for ann in annotations 
            if current_start <= ann.FrameTimestamp < current_end
        )
        
        intervals.append({
            "start_time": float(current_start),
            "end_time": float(current_end),
            "annotation_count": count
        })
        
        current_start = current_end
    
    return intervals


def export_annotations_for_video_session(video_session_id: int) -> Dict[str, Any]:
    annotations = get_annotations_by_video_session(video_session_id)
    statistics = get_video_session_statistics(video_session_id)
    
    return {
        "video_session_id": video_session_id,
        "export_timestamp": datetime.utcnow(),
        "statistics": statistics,
        "annotations": [
            {
                "id": ann.ID,
                "timestamp": float(ann.FrameTimestamp),
                "data": ann.AnnotationData,
                "created_at": ann.CreatedAt
            }
            for ann in annotations
        ]
    }


def cleanup_old_annotations(days_old: int = 90) -> Dict[str, Any]:
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    old_annotations = get_annotations_by_date_range(
        datetime.min, 
        cutoff_date
    )
    
    deleted_count = 0
    for annotation in old_annotations:
        frame_annotations_repository.delete_annotation(annotation.ID)
        deleted_count += 1
    
    return {
        "message": f"Cleaned up {deleted_count} old annotations",
        "deleted_count": deleted_count
    }


def get_annotations_summary_by_content_type(video_session_id: int) -> Dict[str, int]:
    annotations = get_annotations_by_video_session(video_session_id)
    content_types = {}
    
    for annotation in annotations:
        annotation_data = annotation.AnnotationData
        if isinstance(annotation_data, dict):
            for key in annotation_data.keys():
                content_types[key] = content_types.get(key, 0) + 1
    
    return content_types