from pydantic import BaseModel, Field, StrictStr, StrictInt, StrictBool
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal


class ProcessingStatusEnum(str, Enum):
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Users(BaseModel):
    """
    Model of anonymous users
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    FingerprintHash: StrictStr = Field(...,
                                       alias="fingerprint_hash",
                                       examples=["a1b2c3d4e5f6789012345678901234567890abcdef"])
    FirstVisit: datetime = Field(...,
                                 alias="first_visit",
                                 examples=[f"{datetime.now()}"])
    LastActivity: datetime = Field(...,
                                   alias="last_activity",
                                   examples=[f"{datetime.now()}"])
    TotalSessions: StrictInt = Field(default=0,
                                     alias="total_sessions",
                                     examples=[5])
    TotalVideosProcessed: StrictInt = Field(default=0,
                                            alias="total_videos_processed",
                                            examples=[12])
    CreatedAt: Optional[datetime] = Field(None,
                                          alias="created_at",
                                          examples=[f"{datetime.now()}"])


class UserSessions(BaseModel):
    """
    Model of user sessions
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    UserID: StrictInt = Field(...,
                              alias="user_id",
                              examples=[1])
    JwtTokenHash: StrictStr = Field(...,
                                    alias="jwt_token_hash",
                                    examples=["b2c3d4e5f6789012345678901234567890abcdef12"])
    ExpiresAt: datetime = Field(...,
                                alias="expires_at",
                                examples=[f"{datetime.now()}"])
    UserAgent: Optional[StrictStr] = Field(None,
                                           alias="user_agent",
                                           examples=["Mozilla/5.0 (Windows NT 10.0; Win64; x64)"])
    IPAddress: Optional[StrictStr] = Field(None,
                                           alias="ip_address",
                                           examples=["192.168.1.1"])
    IsActive: StrictInt = Field(default=1,
                                alias="is_active",
                                examples=[1])
    CreatedAt: Optional[datetime] = Field(None,
                                          alias="created_at",
                                          examples=[f"{datetime.now()}"])


class VideoSessions(BaseModel):
    """
    Model of video processing sessions
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    UserID: StrictInt = Field(...,
                              alias="user_id",
                              examples=[1])
    VideoURL: StrictStr = Field(...,
                                alias="video_url",
                                examples=["https://rutube.ru/video/98a85192e297ff4db1860f43ff7a2738/"])
    ProcessingStatus: ProcessingStatusEnum = Field(default=ProcessingStatusEnum.PROCESSING,
                                               alias="processing_status",
                                               examples=["processing"])
    StartedAt: Optional[datetime] = Field(None,
                                          alias="started_at",
                                          examples=[f"{datetime.now()}"])
    CompletedAt: Optional[datetime] = Field(None,
                                            alias="completed_at",
                                            examples=[f"{datetime.now()}"])


class FrameAnnotations(BaseModel):
    """
    Model of frame annotations
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    VideoSessionID: StrictInt = Field(...,
                                      alias="video_session_id",
                                      examples=[1])
    FrameTimestamp: Decimal = Field(...,
                                    alias="frame_timestamp",
                                    examples=[15.250])
    AnnotationData: Dict[str, Any] = Field(...,
                                           alias="annotation_data",
                                           examples=[{"results": [{"text_sequence": "Hello World"}]}])
    CreatedAt: Optional[datetime] = Field(None,
                                          alias="created_at",
                                          examples=[f"{datetime.now()}"])
