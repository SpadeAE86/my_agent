from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy.dialects.mysql import VARCHAR, BIGINT


class VideoSourceUploadCache(SQLModel, table=True):
    """
    Cache: local video file fingerprint -> OBS object path/url.

    Motivation:
    - `run_video_analysis_v2.py` used a local JSON cache to avoid re-uploading the same file.
    - Move that cache into MySQL so it is shared across runs / machines.

    Fingerprint choice:
    - We reuse `_video_sig(video_path)` (abs_path + mtime + size) => sha1[:12]
    - This is stable enough for "same file at same path"; if file changes, mtime/size changes -> new key.
    """

    __tablename__ = "video_source_upload_cache"

    # same as _video_sig(video_path) (12 hex chars) but allow a bit more room
    sig: str = Field(sa_column=Column(VARCHAR(32), primary_key=True, nullable=False))

    file_name: str = Field(sa_column=Column(VARCHAR(255), nullable=False))
    abs_path: str = Field(sa_column=Column(Text, nullable=False))
    file_size: int = Field(sa_column=Column(BIGINT, nullable=False))
    file_mtime: int = Field(sa_column=Column(BIGINT, nullable=False))

    obs_key: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    obs_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )

