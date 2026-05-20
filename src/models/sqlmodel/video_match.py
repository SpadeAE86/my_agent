from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, Float, Integer, ForeignKey, func, Index
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR

class VideoMatchJob(SQLModel, table=True):
    """
    One「视频匹配」会话：口播转写 +（后续）分镜检索与混剪。
    """

    __tablename__ = "video_match_job"

    id: str = Field(sa_column=Column(VARCHAR(36), primary_key=True, nullable=False))

    workspace: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(64), nullable=True))

    serial_no: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, autoincrement=True, unique=True, index=True)
    )

    script: str = Field(sa_column=Column(Text, nullable=False))
    topic: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    title: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    car_model: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    #: 与索引 `frame_size` 对齐：横版16:9 / 竖版9:16；转写后写入每镜 tags_json 并参与检索过滤
    frame_size: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(32), nullable=True))
    #: 与索引 `frame_orientation` 对齐：横屏 / 竖屏；可不填具体比例只做横竖约束
    frame_orientation: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(32), nullable=True))

    parse_status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
        description="pending | running | done | failed",
    )
    parse_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    extract_status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
        description="pending | running | done | failed",
    )
    extract_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    search_status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
        description="pending | running | done | failed",
    )
    search_total_ms: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))
    search_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    search_strategy_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(MySQLJSON, nullable=True),
    )

    # 联表 http_request_traces.id：整 job「转写 / 解析」阶段请求记录，便于任务看板
    request_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), nullable=True, index=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoMatchShotRow(SQLModel, table=True):
    """单条分镜：Stage1 展示字段 + Stage2 标签 JSON（供后续 OpenSearch 匹配）。"""

    __tablename__ = "video_match_shot_row"
    #: 与按 job 拉分镜并 ORDER BY shot_order 的查询对齐，避免宽行 JSON/filesort 触发 sort_buffer 1038
    __table_args__ = (Index("ix_video_match_shot_row_job_shot_order", "job_id", "shot_order"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(
        sa_column=Column(
            VARCHAR(36),
            ForeignKey("video_match_job.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )

    shot_order: int = Field(sa_column=Column(Integer, nullable=False))
    storyboard_id: int = Field(sa_column=Column(Integer, nullable=False))

    segment_text: str = Field(sa_column=Column(Text, nullable=False))
    duration_sec: float = Field(sa_column=Column(Float, nullable=False))
    description: str = Field(sa_column=Column(Text, nullable=False))

    tags_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    extract_status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
    )
    extract_request_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), nullable=True, index=True))

    search_status: str = Field(
        default="pending",
        sa_column=Column(VARCHAR(32), nullable=False),
    )
    top1_obs_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    obs_audio_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    match_top_hits_json: Optional[List[Any]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    match_elapsed_ms: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    # 联表 http_request_traces.id：本分镜 OpenSearch 匹配阶段请求记录
    search_request_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), nullable=True, index=True))

    # 联表 video_material_match_history.id：当前/最近一次素材检索履历（任务看板素材匹配）
    match_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            VARCHAR(36),
            ForeignKey("video_material_match_history.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )
