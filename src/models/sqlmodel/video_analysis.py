from __future__ import annotations

from datetime import datetime
from typing import List, Optional, TypedDict

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, func, BigInteger, Integer
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR
from sqlalchemy import String


class VideoAnalysisSearchStrategy(SQLModel, table=True):
    """
    搜索策略配置表：保存用户自定义的 BM25 和 Vector 权重，以及名称等。
    """
    __tablename__ = "video_analysis_search_strategy"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(VARCHAR(128), nullable=False, unique=True))
    bm25_weight: float = Field(default=0.3, nullable=False)
    vector_weight: float = Field(default=0.7, nullable=False)
    
    # 细粒度的字段权重配置
    text_weights: Optional[dict] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    vector_weights: Optional[dict] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    
    is_default: bool = Field(default=False, nullable=False)
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


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
    workspace: str = Field(default="v1", sa_column=Column(VARCHAR(32), nullable=False, server_default="v1"))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoAnalysisVideoV2(SQLModel, table=True):
    """
    v2 分析用「视频」行：自增 ``id`` 为分镜表 / 参考帧表外键；
    ``video_key`` 与 ``video_analysis_history.id``、OpenSearch 文档 id 前缀一致。
    新数据约定 ``video_key == str(id)``；历史数据可能仍为 ``v2_<hash>`` 等旧键。

    ``source_file_name``：源视频 basename（与本地抽帧目录名一致），同一文件多次 analysis / overwrite
    复用同一行；帧序列缓存按 ``video_id`` 全量替换，不保留历史版本。
    """

    __tablename__ = "video_analysis_video_v2"

    id: Optional[int] = Field(default=None, primary_key=True)
    video_key: str = Field(sa_column=Column(VARCHAR(64), nullable=False, unique=True, index=True))
    source_file_name: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(512), nullable=True, unique=True, index=True),
        description="源视频文件名（basename）；用于同素材复用 video 行，不依赖上传缓存表",
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoAnalysisSceneFrames(SQLModel, table=True):
    """
    分镜参考帧缓存表（workspace 共用）：``(video_id, scene_id)`` 唯一定位一组帧 URL（OBS 公网 URL）。

    对象路径约定（与理解 schema 解耦）：``ai_picture/video_analysis_frames/{video_key}/{scene_id}/``；
    ``video_id`` 为本表外键，仅用于 MySQL 关联。

    本表跨 workspace 共用，避免同一素材因切换分析版本而重复上传抽帧。

    DB 迁移（首次部署时执行）：
        ALTER TABLE video_analysis_scene_frames_v2
            RENAME TO video_analysis_scene_frames;
    """

    __tablename__ = "video_analysis_scene_frames"

    video_id: int = Field(foreign_key="video_analysis_video_v2.id", primary_key=True)
    scene_id: int = Field(primary_key=True)
    frame_paths: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoAnalysisSceneSplitFramesCache(SQLModel, table=True):
    """
    分镜切分后各镜 OBS 参考帧 URL 列表缓存（逻辑外键 ``video_id`` -> ``video_analysis_video_v2.id``）。

    与 ``video_analysis_scene_frames_v2`` 并存：便于「先本地 → 再本表 → 缺失则 HTTP 拉回本地」；
    切分结果变化时应 ``replace``：更新仍存在的 ``scene_id`` 的 URL 列表，并删除本次切分中已不存在的镜行。
    """

    __tablename__ = "video_analysis_scene_split_frames_cache"

    video_id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    scene_id: int = Field(sa_column=Column(Integer, primary_key=True))
    obs_frame_url_list: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class VideoAnalysisShotCardSharedFields(TypedDict, total=False):
    """
    分镜卡片「两表共有」字段说明（非 ORM）：SQLModel 多表继承同一组 Field 时，
    会复用同一套 Column 对象导致启动报错；因此 v1 / v2 在 ORM 层各自完整声明列，
    此处仅作文档与类型辅助。
    """

    id: int
    history_id: str
    video_id: int  # v2：FK -> video_analysis_video_v2.id
    scene_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    thumbnail: Optional[str]  # v1：首帧 URL
    obs_video_url: Optional[str]  # API 组装字段：源视频 URL（来自 history.video_url 连表）
    frame_urls: Optional[List[str]]  # API 组装字段：来自 video_analysis_scene_frames_v2
    description: Optional[str]
    subject: Optional[str]
    object: Optional[List[str]]
    movement: Optional[str]


class VideoAnalysisShotCard(SQLModel, table=True):
    """
    历史分镜卡片 v1（表 `video_analysis_shot_cards`）。
    与 v2 共有的列在本类中完整声明（勿再抽成带 Field 的 SQLModel 父类，见模块说明）。
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


class VideoAnalysisShotCardV2(SQLModel, table=True):
    """
    分镜卡片 v2（表 `video_analysis_shot_cards_v2`），与 SceneAnalysisResultV2 对齐。
    主键为 (video_id, scene_id)；源视频 URL 与参考帧列表见 ``video_analysis_history``、
    ``video_analysis_scene_frames_v2``，由服务层连表组装到 API 输出。
    """

    __tablename__ = "video_analysis_shot_cards_v2"

    video_id: int = Field(foreign_key="video_analysis_video_v2.id", primary_key=True)
    scene_id: int = Field(primary_key=True)

    start_time: float = Field(nullable=False)
    end_time: float = Field(nullable=False)
    duration_seconds: float = Field(nullable=False)

    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    subject: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    object: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    movement: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    analysis_doc_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), nullable=True))
    car_model: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    frame_size: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    resolution: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    video_duration: Optional[float] = Field(default=None, nullable=True)

    footage_type: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    shot_style: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    shot_type: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    camera_movement: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    scene_location: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    car_color: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    product_status_scene: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    has_presenter: Optional[bool] = Field(default=None, nullable=True)
    generic_hq_road_run: bool = Field(default=False, nullable=False)

    person_detail: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    key_words: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    text: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    video_usage: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    design_adjectives: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    function_adjectives: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    design_selling_points: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    function_selling_points: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    scenario_a: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    scenario_b: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    marketing_phrases: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    marketing_tags: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    appealing_audience: Optional[List[str]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    topic: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    weather: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    time: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

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
