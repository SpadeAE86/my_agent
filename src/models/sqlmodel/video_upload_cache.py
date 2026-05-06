from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy.dialects.mysql import VARCHAR, BIGINT


class VideoSourceUploadCache(SQLModel, table=True):
    """
    缓存：本地源视频文件名 -> OBS key/url（避免同一文件名重复上传）。

    键：``file_name`` 使用 ``os.path.basename(path)``（与 OBS 对象名一致），
    便于按 ``ai_picture/car_video_analysis/source_video/{CAR_MODEL}/{file_name}`` 推断路径。

    若曾使用旧版 ``sig`` 主键表，需迁移或重建本表。
    """

    __tablename__ = "video_source_upload_cache"

    id: Optional[int] = Field(default=None, primary_key=True)

    file_name: str = Field(
        sa_column=Column(VARCHAR(512), nullable=False, unique=True, index=True),
        description="视频文件名（basename），与 OBS 上对象名一致",
    )
    abs_path: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    file_size: Optional[int] = Field(default=None, sa_column=Column(BIGINT, nullable=True))
    file_mtime: Optional[int] = Field(default=None, sa_column=Column(BIGINT, nullable=True))

    obs_key: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    obs_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )
