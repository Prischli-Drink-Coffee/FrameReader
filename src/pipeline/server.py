import os
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, status, Form, Query
from typing import Dict
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from src.utils.custom_logging import setup_logging
from src.utils.env import Env
from src import path_to_project
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import Query, Form
from decimal import Decimal

from typing import Dict
from fastapi import Request, Response, Depends
from src.services.cookie_services import session_manager

from src.database.models import (
    Users,
    UserSessions,
    VideoSessions,
    FrameAnnotations,
    ProcessingStatusEnum
)
from src.services import (
    cookie_services,
    frame_annotations_services,
    user_services,
    user_sessions_services,
    video_sessions_services
)

env = Env()
log = setup_logging()


app = FastAPI()

app_server = FastAPI(title="FrameReader Server API",
                     version="1.0.0",
                     description="This API server is intended for the FrameReader project. For rights, \
                                  contact the service owner (dfvolkhin@edu.hse.ru).")

app.mount("/server", app_server)
# app.mount("/public", app_public)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PublicMainTag = OpenApiTag(name="Main", description="CRUD operations main")
ServerMainTag = OpenApiTag(name="Main", description="CRUD operations main")
ServerCookieTag = OpenApiTag(name="Cookie", description="CRUD operations cookie")
ServerUserTag = OpenApiTag(name="User", description="CRUD operations user")
ServerFrameAnnotationsTag = OpenApiTag(name="FrameAnnotations", description="CRUD operations frame annotations")
ServerUserSessionsTag = OpenApiTag(name="UserSessions", description="CRUD operations user sessions")
ServerVideoSessionsTag = OpenApiTag(name="VideoSessions", description="CRUD operations video sessions")

app_server.openapi_tags = [
    ServerMainTag.model_dump(),
    ServerCookieTag.model_dump(),
    ServerUserTag.model_dump(),
    ServerFrameAnnotationsTag.model_dump(),
    ServerUserSessionsTag.model_dump(),
    ServerVideoSessionsTag.model_dump(),
]

# app_public.openapi_tags = [
#     # PublicMainTag.model_dump(),
# ]


@app_server.post("/auth/session/create", response_model=Dict[str, Any], tags=["Cookie"])
async def create_or_get_session(request: Request, response: Response):
    """
    Route for create or get existing user session with fingerprint authentication.

    :param request: FastAPI request object
    :param response: FastAPI response object

    :return: response model dict with session info
    """
    try:
        return session_manager.create_or_get_user_session(request, response)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/auth/session/current", response_model=Dict[str, Any], tags=["Cookie"])
async def get_current_session(request: Request):
    """
    Route for get current user session information.

    :param request: FastAPI request object

    :return: response model dict with current session info
    """
    try:
        return session_manager.get_current_user_from_request(request)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/auth/session/logout", response_model=Dict[str, str], tags=["Cookie"])
async def logout_session(request: Request, response: Response):
    """
    Route for logout current user session.

    :param request: FastAPI request object
    :param response: FastAPI response object

    :return: response model dict with logout message
    """
    try:
        return session_manager.logout_user(request, response)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/auth/session/validate", response_model=bool, tags=["Cookie"])
async def validate_session(request: Request):
    """
    Route for validate current user session.

    :param request: FastAPI request object

    :return: response model bool indicating if session is valid
    """
    try:
        session_manager.get_current_user_from_request(request)
        return True
    except HTTPException:
        return False


@app_server.post("/auth/session/refresh", response_model=Dict[str, Any], tags=["Cookie"])
async def refresh_session(request: Request, response: Response):
    """
    Route for refresh current user session token.

    :param request: FastAPI request object
    :param response: FastAPI response object

    :return: response model dict with refreshed session info
    """
    try:
        current_session = session_manager.get_current_user_from_request(request)
        return session_manager.create_or_get_user_session(request, response)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


def get_current_user_dependency(request: Request) -> Dict[str, Any]:
    """
    Dependency function for extracting current user from session.

    :param request: FastAPI request object

    :return: dict with current user session info
    """
    return session_manager.get_current_user_from_request(request)


@app_server.get("/auth/session/info", response_model=Dict[str, Any], tags=["Cookie"])
async def get_session_info(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user_dependency)
):
    """
    Route for get detailed session information for authenticated user.

    :param request: FastAPI request object
    :param current_user: Current user session info from dependency

    :return: response model dict with detailed session info
    """
    try:
        session_security_info = user_sessions_services.get_session_security_info(
            current_user["session_id"]
        )
        user_stats = user_services.get_user_statistics(current_user["user_id"])
        
        return {
            "session_info": session_security_info,
            "user_stats": user_stats,
            "fingerprint_hash": current_user["fingerprint_hash"]
        }
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/auth/session/clear", response_model=Dict[str, str], tags=["Cookie"])
async def clear_session_cookie(response: Response):
    """
    Route for clear session cookie without server-side validation.

    :param response: FastAPI response object

    :return: response model dict with clear message
    """
    try:
        session_manager.jwt_manager.clear_cookie(response)
        return {"message": "Session cookie cleared"}
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/", response_model=List[Users], tags=["User"])
async def get_all_users():
    """
    Route for get all users from database.

    :return: response model List[Users].
    """
    try:
        return user_services.get_all_users()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/user_id/{user_id}", response_model=Users, tags=["User"])
async def get_user_by_id(user_id: int):
    """
    Route for get user by UserID.

    :param user_id: ID by user. [int]

    :return: response model Users.
    """
    try:
        return user_services.get_user_by_id(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/fingerprint/{fingerprint_hash}", response_model=Users, tags=["User"])
async def get_user_by_fingerprint(fingerprint_hash: str):
    """
    Route for get user by fingerprint hash.

    :param fingerprint_hash: Fingerprint hash by user. [str]

    :return: response model Users.
    """
    try:
        user = user_services.get_user_by_fingerprint(fingerprint_hash)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/users/", response_model=Users, tags=["User"])
async def create_user(fingerprint_hash: str = Form(...)):
    """
    Route for create user in database.

    :param fingerprint_hash: Fingerprint hash for user. [str]

    :return: response model Users.
    """
    try:
        return user_services.create_user(fingerprint_hash)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/users/get-or-create/", response_model=Users, tags=["User"])
async def get_or_create_user(fingerprint_hash: str = Form(...)):
    """
    Route for get existing user or create new user.

    :param fingerprint_hash: Fingerprint hash for user. [str]

    :return: response model Users.
    """
    try:
        return user_services.get_or_create_user(fingerprint_hash)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/users/{user_id}/activity", response_model=Dict, tags=["User"])
async def update_user_activity(user_id: int):
    """
    Route for update user activity timestamp.

    :param user_id: ID by user. [int]

    :return: response model dict.
    """
    try:
        return user_services.update_user_activity(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/users/{user_id}/videos/increment", response_model=Dict, tags=["User"])
async def increment_user_videos(user_id: int):
    """
    Route for increment user videos processed count.

    :param user_id: ID by user. [int]

    :return: response model dict.
    """
    try:
        return user_services.increment_user_videos(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/users/{user_id}", response_model=Dict, tags=["User"])
async def update_user_fields(user_id: int, updates: Dict[str, Any]):
    """
    Route for update user fields in database.

    :param user_id: ID by user. [int]
    :param updates: Fields to update. [Dict[str, Any]]

    :return: response model dict.
    """
    try:
        return user_services.update_user_fields(user_id, updates)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/users/{user_id}", response_model=Dict, tags=["User"])
async def delete_user(user_id: int):
    """
    Route for delete user from database.

    :param user_id: ID by user. [int]

    :return: response model dict.
    """
    try:
        return user_services.delete_user(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/active/", response_model=List[Users], tags=["User"])
async def get_active_users():
    """
    Route for get active users from current month.

    :return: response model List[Users].
    """
    try:
        return user_services.get_active_users()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/{user_id}/statistics", response_model=Dict[str, Any], tags=["User"])
async def get_user_statistics(user_id: int):
    """
    Route for get user statistics.

    :param user_id: ID by user. [int]

    :return: response model dict with statistics.
    """
    try:
        return user_services.get_user_statistics(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/activity-range/", response_model=List[Users], tags=["User"])
async def get_users_by_activity_range(
    start_date: datetime = Query(..., description="Start date for activity range"),
    end_date: datetime = Query(..., description="End date for activity range")
):
    """
    Route for get users by activity date range.

    :param start_date: Start date for filtering. [datetime]
    :param end_date: End date for filtering. [datetime]

    :return: response model List[Users].
    """
    try:
        return user_services.get_users_by_activity_range(start_date, end_date)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/active/count", response_model=int, tags=["User"])
async def get_active_users_count():
    """
    Route for get count of active users.

    :return: response model int.
    """
    try:
        return user_services.get_active_users_count()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/", response_model=List[UserSessions], tags=["UserSessions"])
async def get_all_sessions():
    """
    Route for get all user sessions from database.

    :return: response model List[UserSessions].
    """
    try:
        return user_sessions_services.get_all_sessions()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/{session_id}", response_model=UserSessions, tags=["UserSessions"])
async def get_session_by_id(session_id: int):
    """
    Route for get session by ID.

    :param session_id: ID by session. [int]

    :return: response model UserSessions.
    """
    try:
        return user_sessions_services.get_session_by_id(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/token/{token_hash}", response_model=UserSessions, tags=["UserSessions"])
async def get_session_by_token_hash(token_hash: str):
    """
    Route for get session by JWT token hash.

    :param token_hash: JWT token hash. [str]

    :return: response model UserSessions.
    """
    try:
        session = user_sessions_services.get_session_by_token_hash(token_hash)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        return session
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/user/{user_id}/active", response_model=List[UserSessions], tags=["UserSessions"])
async def get_active_sessions_by_user(user_id: int):
    """
    Route for get active sessions by user ID.

    :param user_id: ID by user. [int]

    :return: response model List[UserSessions].
    """
    try:
        return user_sessions_services.get_active_sessions_by_user(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/user/{user_id}", response_model=List[UserSessions], tags=["UserSessions"])
async def get_sessions_by_user(user_id: int):
    """
    Route for get all sessions by user ID.

    :param user_id: ID by user. [int]

    :return: response model List[UserSessions].
    """
    try:
        return user_sessions_services.get_sessions_by_user(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/user-sessions/", response_model=UserSessions, tags=["UserSessions"])
async def create_session(session: UserSessions):
    """
    Route for create user session in database.

    :param session: Session model. [UserSessions]

    :return: response model UserSessions.
    """
    try:
        return user_sessions_services.create_session(session)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/user-sessions/{session_id}", response_model=Dict[str, str], tags=["UserSessions"])
async def update_session_fields(session_id: int, updates: Dict[str, Any]):
    """
    Route for update session fields in database.

    :param session_id: ID by session. [int]
    :param updates: Fields to update. [Dict[str, Any]]

    :return: response model dict.
    """
    try:
        return user_sessions_services.update_session_fields(session_id, updates)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/user-sessions/{session_id}/activity", response_model=Dict[str, str], tags=["UserSessions"])
async def update_session_activity(session_id: int, last_activity: Optional[datetime] = None):
    """
    Route for update session activity timestamp.

    :param session_id: ID by session. [int]
    :param last_activity: Last activity timestamp. [datetime]

    :return: response model dict.
    """
    try:
        activity_time = last_activity or datetime.utcnow()
        return user_sessions_services.update_session_activity(session_id, activity_time)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/user-sessions/{session_id}/deactivate", response_model=Dict[str, str], tags=["UserSessions"])
async def deactivate_session(session_id: int):
    """
    Route for deactivate session.

    :param session_id: ID by session. [int]

    :return: response model dict.
    """
    try:
        return user_sessions_services.deactivate_session(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/user-sessions/user/{user_id}/deactivate-all", response_model=Dict[str, str], tags=["UserSessions"])
async def deactivate_user_sessions(user_id: int):
    """
    Route for deactivate all user sessions.

    :param user_id: ID by user. [int]

    :return: response model dict.
    """
    try:
        return user_sessions_services.deactivate_user_sessions(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/user-sessions/{session_id}", response_model=Dict[str, str], tags=["UserSessions"])
async def delete_session(session_id: int):
    """
    Route for delete session from database.

    :param session_id: ID by session. [int]

    :return: response model dict.
    """
    try:
        return user_sessions_services.delete_session(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/user-sessions/cleanup/expired", response_model=Dict[str, Any], tags=["UserSessions"])
async def cleanup_expired_sessions():
    """
    Route for cleanup expired sessions from database.

    :return: response model dict with cleanup results.
    """
    try:
        return user_sessions_services.cleanup_expired_sessions()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/expired/", response_model=List[UserSessions], tags=["UserSessions"])
async def get_expired_sessions():
    """
    Route for get expired sessions.

    :return: response model List[UserSessions].
    """
    try:
        return user_sessions_services.get_expired_sessions()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/ip/{ip_address}", response_model=List[UserSessions], tags=["UserSessions"])
async def get_sessions_by_ip(ip_address: str):
    """
    Route for get sessions by IP address.

    :param ip_address: IP address. [str]

    :return: response model List[UserSessions].
    """
    try:
        return user_sessions_services.get_sessions_by_ip(ip_address)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/user/{user_id}/statistics", response_model=Dict[str, Any], tags=["UserSessions"])
async def get_user_session_statistics(user_id: int):
    """
    Route for get user session statistics.

    :param user_id: ID by user. [int]

    :return: response model dict with statistics.
    """
    try:
        return user_sessions_services.get_user_session_statistics(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/user-sessions/{session_id}/extend", response_model=Dict[str, str], tags=["UserSessions"])
async def extend_session(session_id: int, extend_days: int = Query(30, description="Number of days to extend")):
    """
    Route for extend session expiration time.

    :param session_id: ID by session. [int]
    :param extend_days: Number of days to extend. [int]

    :return: response model dict.
    """
    try:
        return user_sessions_services.extend_session(session_id, extend_days)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/{session_id}/validate", response_model=bool, tags=["UserSessions"])
async def validate_active_session(session_id: int):
    """
    Route for validate if session is active.

    :param session_id: ID by session. [int]

    :return: response model bool.
    """
    try:
        return user_sessions_services.validate_active_session(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/user-sessions/{session_id}/security-info", response_model=Dict[str, Any], tags=["UserSessions"])
async def get_session_security_info(session_id: int):
    """
    Route for get session security information.

    :param session_id: ID by session. [int]

    :return: response model dict with security info.
    """
    try:
        return user_sessions_services.get_session_security_info(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def get_all_annotations():
    """
    Route for get all frame annotations from database.

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.get_all_annotations()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/{annotation_id}", response_model=FrameAnnotations, tags=["FrameAnnotations"])
async def get_annotation_by_id(annotation_id: int):
    """
    Route for get frame annotation by ID.

    :param annotation_id: ID by annotation. [int]

    :return: response model FrameAnnotations.
    """
    try:
        return frame_annotations_services.get_annotation_by_id(annotation_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def get_annotations_by_video_session(video_session_id: int):
    """
    Route for get all annotations by video session ID.

    :param video_session_id: ID by video session. [int]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.get_annotations_by_video_session(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/range", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def get_annotations_by_timestamp_range(
    video_session_id: int,
    start_time: Decimal = Query(..., description="Start timestamp"),
    end_time: Decimal = Query(..., description="End timestamp")
):
    """
    Route for get annotations by timestamp range.

    :param video_session_id: ID by video session. [int]
    :param start_time: Start timestamp. [Decimal]
    :param end_time: End timestamp. [Decimal]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.get_annotations_by_timestamp_range(
            video_session_id, start_time, end_time
        )
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/latest", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def get_latest_annotations(
    video_session_id: int,
    limit: int = Query(10, description="Number of latest annotations to retrieve")
):
    """
    Route for get latest annotations by video session.

    :param video_session_id: ID by video session. [int]
    :param limit: Number of annotations to retrieve. [int]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.get_latest_annotations(video_session_id, limit)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/frame-annotations/", response_model=FrameAnnotations, tags=["FrameAnnotations"])
async def create_annotation(
    video_session_id: int,
    frame_timestamp: Decimal,
    annotation_data: Dict[str, Any]
):
    """
    Route for create frame annotation.

    :param video_session_id: ID by video session. [int]
    :param frame_timestamp: Frame timestamp. [Decimal]
    :param annotation_data: Annotation data. [Dict[str, Any]]

    :return: response model FrameAnnotations.
    """
    try:
        return frame_annotations_services.create_annotation(
            video_session_id, frame_timestamp, annotation_data
        )
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/frame-annotations/batch", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def create_annotation_batch(annotations: List[FrameAnnotations]):
    """
    Route for create multiple frame annotations.

    :param annotations: List of annotations to create. [List[FrameAnnotations]]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.create_annotation_batch(annotations)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/frame-annotations/{annotation_id}", response_model=Dict[str, str], tags=["FrameAnnotations"])
async def update_annotation_fields(annotation_id: int, updates: Dict[str, Any]):
    """
    Route for update annotation fields.

    :param annotation_id: ID by annotation. [int]
    :param updates: Fields to update. [Dict[str, Any]]

    :return: response model dict.
    """
    try:
        return frame_annotations_services.update_annotation_fields(annotation_id, updates)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/frame-annotations/{annotation_id}", response_model=Dict[str, str], tags=["FrameAnnotations"])
async def delete_annotation(annotation_id: int):
    """
    Route for delete frame annotation.

    :param annotation_id: ID by annotation. [int]

    :return: response model dict.
    """
    try:
        return frame_annotations_services.delete_annotation(annotation_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/frame-annotations/video-session/{video_session_id}", response_model=Dict[str, Any], tags=["FrameAnnotations"])
async def delete_annotations_by_video_session(video_session_id: int):
    """
    Route for delete all annotations by video session.

    :param video_session_id: ID by video session. [int]

    :return: response model dict with deletion results.
    """
    try:
        return frame_annotations_services.delete_annotations_by_video_session(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/statistics", response_model=Dict[str, Any], tags=["FrameAnnotations"])
async def get_video_session_statistics(video_session_id: int):
    """
    Route for get video session annotation statistics.

    :param video_session_id: ID by video session. [int]

    :return: response model dict with statistics.
    """
    try:
        return frame_annotations_services.get_video_session_statistics(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/date-range/", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def get_annotations_by_date_range(
    start_date: datetime = Query(..., description="Start date for range"),
    end_date: datetime = Query(..., description="End date for range")
):
    """
    Route for get annotations by date range.

    :param start_date: Start date for filtering. [datetime]
    :param end_date: End date for filtering. [datetime]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.get_annotations_by_date_range(start_date, end_date)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/timeline", response_model=List[Dict[str, Any]], tags=["FrameAnnotations"])
async def get_video_session_timeline(video_session_id: int):
    """
    Route for get video session timeline.

    :param video_session_id: ID by video session. [int]

    :return: response model list of timeline data.
    """
    try:
        return frame_annotations_services.get_video_session_timeline(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/search", response_model=List[FrameAnnotations], tags=["FrameAnnotations"])
async def search_annotations_by_content(
    video_session_id: int,
    search_term: str = Query(..., description="Search term for annotation content")
):
    """
    Route for search annotations by content.

    :param video_session_id: ID by video session. [int]
    :param search_term: Search term. [str]

    :return: response model List[FrameAnnotations].
    """
    try:
        return frame_annotations_services.search_annotations_by_content(video_session_id, search_term)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/density", response_model=List[Dict[str, Any]], tags=["FrameAnnotations"])
async def get_annotation_density_analysis(
    video_session_id: int,
    interval_seconds: int = Query(10, description="Interval in seconds for density analysis")
):
    """
    Route for get annotation density analysis.

    :param video_session_id: ID by video session. [int]
    :param interval_seconds: Interval in seconds. [int]

    :return: response model list of density intervals.
    """
    try:
        return frame_annotations_services.get_annotation_density_analysis(video_session_id, interval_seconds)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/export", response_model=Dict[str, Any], tags=["FrameAnnotations"])
async def export_annotations_for_video_session(video_session_id: int):
    """
    Route for export annotations for video session.

    :param video_session_id: ID by video session. [int]

    :return: response model dict with export data.
    """
    try:
        return frame_annotations_services.export_annotations_for_video_session(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/frame-annotations/cleanup/old", response_model=Dict[str, Any], tags=["FrameAnnotations"])
async def cleanup_old_annotations(
    days_old: int = Query(90, description="Number of days to consider annotations as old")
):
    """
    Route for cleanup old annotations.

    :param days_old: Number of days to consider old. [int]

    :return: response model dict with cleanup results.
    """
    try:
        return frame_annotations_services.cleanup_old_annotations(days_old)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/frame-annotations/video-session/{video_session_id}/content-summary", response_model=Dict[str, int], tags=["FrameAnnotations"])
async def get_annotations_summary_by_content_type(video_session_id: int):
    """
    Route for get annotations summary by content type.

    :param video_session_id: ID by video session. [int]

    :return: response model dict with content type counts.
    """
    try:
        return frame_annotations_services.get_annotations_summary_by_content_type(video_session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_all_video_sessions():
    """
    Route for get all video sessions from database.

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_all_video_sessions()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/{session_id}", response_model=VideoSessions, tags=["VideoSessions"])
async def get_video_session_by_id(session_id: int):
    """
    Route for get video session by ID.

    :param session_id: ID by video session. [int]

    :return: response model VideoSessions.
    """
    try:
        return video_sessions_services.get_video_session_by_id(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/user/{user_id}", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_video_sessions_by_user(user_id: int):
    """
    Route for get all video sessions by user ID.

    :param user_id: ID by user. [int]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_video_sessions_by_user(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/status/{status}", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_video_sessions_by_status(status: ProcessingStatusEnum):
    """
    Route for get video sessions by processing status.

    :param status: Processing status. [ProcessingStatusEnum]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_video_sessions_by_status(status)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/user/{user_id}/status/{status}", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_video_sessions_by_user_and_status(user_id: int, status: ProcessingStatusEnum):
    """
    Route for get video sessions by user ID and status.

    :param user_id: ID by user. [int]
    :param status: Processing status. [ProcessingStatusEnum]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_video_sessions_by_user_and_status(user_id, status)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/processing/", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_processing_sessions():
    """
    Route for get all processing video sessions.

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_processing_sessions()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/video-sessions/", response_model=VideoSessions, tags=["VideoSessions"])
async def create_video_session(
    user_id: int = Form(...),
    video_url: str = Form(...)
):
    """
    Route for create video session.

    :param user_id: ID by user. [int]
    :param video_url: URL of video to process. [str]

    :return: response model VideoSessions.
    """
    try:
        return video_sessions_services.create_video_session(user_id, video_url)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/video-sessions/{session_id}", response_model=Dict[str, str], tags=["VideoSessions"])
async def update_video_session_fields(session_id: int, updates: Dict[str, Any]):
    """
    Route for update video session fields.

    :param session_id: ID by video session. [int]
    :param updates: Fields to update. [Dict[str, Any]]

    :return: response model dict.
    """
    try:
        return video_sessions_services.update_video_session_fields(session_id, updates)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/video-sessions/{session_id}/status", response_model=Dict[str, str], tags=["VideoSessions"])
async def update_session_status(
    session_id: int,
    status: ProcessingStatusEnum,
    completed_at: Optional[datetime] = None
):
    """
    Route for update video session status.

    :param session_id: ID by video session. [int]
    :param status: New processing status. [ProcessingStatusEnum]
    :param completed_at: Completion timestamp. [datetime]

    :return: response model dict.
    """
    try:
        return video_sessions_services.update_session_status(session_id, status, completed_at)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/video-sessions/{session_id}/complete", response_model=Dict[str, str], tags=["VideoSessions"])
async def complete_video_session(session_id: int):
    """
    Route for complete video session.

    :param session_id: ID by video session. [int]

    :return: response model dict.
    """
    try:
        return video_sessions_services.complete_video_session(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.patch("/video-sessions/{session_id}/fail", response_model=Dict[str, str], tags=["VideoSessions"])
async def fail_video_session(session_id: int, error_message: Optional[str] = None):
    """
    Route for mark video session as failed.

    :param session_id: ID by video session. [int]
    :param error_message: Error message. [str]

    :return: response model dict.
    """
    try:
        return video_sessions_services.fail_video_session(session_id, error_message)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/video-sessions/{session_id}", response_model=Dict[str, str], tags=["VideoSessions"])
async def delete_video_session(session_id: int):
    """
    Route for delete video session.

    :param session_id: ID by video session. [int]

    :return: response model dict.
    """
    try:
        return video_sessions_services.delete_video_session(session_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/user/{user_id}/count", response_model=int, tags=["VideoSessions"])
async def get_user_session_count(user_id: int):
    """
    Route for get user video session count.

    :param user_id: ID by user. [int]

    :return: response model int.
    """
    try:
        return video_sessions_services.get_user_session_count(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/date-range/", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_sessions_by_date_range(
    start_date: datetime = Query(..., description="Start date for range"),
    end_date: datetime = Query(..., description="End date for range")
):
    """
    Route for get video sessions by date range.

    :param start_date: Start date for filtering. [datetime]
    :param end_date: End date for filtering. [datetime]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_sessions_by_date_range(start_date, end_date)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/user/{user_id}/completed", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_completed_sessions_by_user(
    user_id: int,
    limit: int = Query(10, description="Number of sessions to retrieve")
):
    """
    Route for get completed video sessions by user.

    :param user_id: ID by user. [int]
    :param limit: Number of sessions to retrieve. [int]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_user_video_history(user_id, limit)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/user/{user_id}/statistics", response_model=Dict[str, Any], tags=["VideoSessions"])
async def get_user_session_statistics(user_id: int):
    """
    Route for get user video session statistics.

    :param user_id: ID by user. [int]

    :return: response model dict with statistics.
    """
    try:
        return video_sessions_services.get_user_session_statistics(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/recent/", response_model=List[VideoSessions], tags=["VideoSessions"])
async def get_recent_sessions(
    days: int = Query(7, description="Number of days to look back")
):
    """
    Route for get recent video sessions.

    :param days: Number of days to look back. [int]

    :return: response model List[VideoSessions].
    """
    try:
        return video_sessions_services.get_recent_sessions(days)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/video-sessions/cleanup/failed", response_model=Dict[str, Any], tags=["VideoSessions"])
async def cleanup_old_failed_sessions(
    days_old: int = Query(30, description="Number of days to consider sessions as old")
):
    """
    Route for cleanup old failed video sessions.

    :param days_old: Number of days to consider old. [int]

    :return: response model dict with cleanup results.
    """
    try:
        return video_sessions_services.cleanup_old_failed_sessions(days_old)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/{session_id}/duration", response_model=Optional[float], tags=["VideoSessions"])
async def get_session_duration(session_id: int):
    """
    Route for get video session duration in seconds.

    :param session_id: ID by video session. [int]

    :return: response model float or None.
    """
    try:
        duration = video_sessions_services.get_session_duration(session_id)
        return duration.total_seconds() if duration else None
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video-sessions/queue/status", response_model=Dict[str, Any], tags=["VideoSessions"])
async def get_processing_queue_status():
    """
    Route for get processing queue status.

    :return: response model dict with queue information.
    """
    try:
        return video_sessions_services.get_processing_queue_status()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex



def run_server():
    import logging
    import uvicorn
    import yaml
    from src import path_to_logging
    uvicorn_log_config = path_to_logging()
    with open(uvicorn_log_config, 'r') as f:
        uvicorn_config = yaml.safe_load(f.read())
        logging.config.dictConfig(uvicorn_config)
    if env.__getattr__("DEBUG") == "TRUE":
        reload = True
    elif env.__getattr__("DEBUG") == "FALSE":
        reload = False
    else:
        raise Exception("Not init debug mode in env file")
    uvicorn.run("src.pipeline.server:app", host=env.__getattr__("HOST"), port=int(env.__getattr__("SERVER_PORT")),
                log_config=uvicorn_log_config, reload=reload)


if __name__ == "__main__":
    log.info("Start run server")
    run_server()
