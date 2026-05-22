from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, Float, Integer, Boolean, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR


class VideoMaterialMatchHistory(SQLModel, table=True):
    """一次素材检索履历（视频匹配分镜 / 视频分析搜索栏），联 http_request_traces.request_id。"""

    __tablename__ = "video_material_match_history"

    id: Optional[int] = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True, nullable=False))

    request_id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(36), nullable=True, index=True),
        description="http_request_traces.id",
    )

    source: str = Field(
        sa_column=Column(VARCHAR(32), nullable=False),
        description="video_match_shot | video_analysis_search",
    )
    workspace: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(64), nullable=True, index=True))

    status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
        description="pending | running | done | failed",
    )
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    video_match_job_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True, index=True))
    video_match_shot_row_id: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))

    va_context_history_id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(64), nullable=True),
        description="视频分析页上下文 history_id，不参与收窄索引",
    )

    hit_count: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    top1_obs_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    elapsed_ms: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    query_preview: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    search_mode: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(32), nullable=True))

    strategy_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
    )

    enable_road_run_fallback: Optional[bool] = Field(
        default=None,
        sa_column=Column(Boolean, nullable=True),
        description="前端传来的路跑兜底标识",
    )

    top_hits_json: Optional[List[Any]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
        description="Top 命中列表（最多20条），含 history_id / video_path / _score",
    )

    top5_obs_urls: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
        description="Top5 命中视频 URL 列表（最多5个）",
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )