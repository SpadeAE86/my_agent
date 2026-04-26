from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR
from sqlalchemy import String


class VideoAnalysisHistory(SQLModel, table=True):
    """
    One analysis run (one uploaded video) = one history row.
    `id` matches the project_id returned by /video-analysis.
    """

    __tablename__ = "video_analysis_history"

    id: str = Field(sa_column=Column(VARCHAR(64), primary_key=True, nullable=False))
    name: str = Field(sa_column=Column(Text, nullable=False))
    time: str = Field(sa_column=Column(Text, nullable=False))
    video_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoAnalysisShotCard(SQLModel, table=True):
    """
    One storyboard card (one scene) under a history item.
    We store list-like fields as MySQL JSON.
    """

    __tablename__ = "video_analysis_shot_cards"

    id: Optional[int] = Field(default=None, primary_key=True)

    history_id: str = Field(sa_column=Column(VARCHAR(64), nullable=False, index=True))
    scene_id: int = Field(nullable=False)

    start_time: float = Field(nullable=False)
    end_time: float = Field(nullable=False)
    duration_seconds: float = Field(nullable=False)

    thumbnail: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    frame_urls: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    subject: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    object: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    movement: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    adjective: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    search_tags: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    marketing_tags: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    appealing_audience: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    visual_quality: Optional[List[float]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    # --- OpenSearch indexing status (derived index, DB is source of truth) ---
    # Values: PENDING / OK / FAILED
    os_index_status: str = Field(
        default="PENDING",
        sa_column=Column(String(16), nullable=False, server_default="PENDING", index=True),
    )
    os_index_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )

