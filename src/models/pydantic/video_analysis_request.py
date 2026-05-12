# models/pydantic/video_analysis_request.py — 视频分析相关 API 请求体定义
from pydantic import BaseModel, Field
from typing import Optional, List


class VideoAnalysisRequest(BaseModel):
    """POST /video-analysis 的请求体
    上传的视频文件通过 multipart/form-data 传递，以下是可选的业务参数
    """
    frame_interval: float = Field(default=2.0, description="抽帧间隔秒数")
    threshold: float = Field(default=30.0, description="场景切换灵敏度阈值")
    custom_prompt: Optional[str] = Field(default=None, description="自定义分析提示词, 不传则使用默认")


class ShotCard(BaseModel):
    """单个分镜卡片结果（对应前端渲染的一张卡片）"""
    scene_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    thumbnail: Optional[str] = Field(default=None, description="分镜首帧 OBS 公网 URL")
    frame_urls: List[str] = Field(default_factory=list, description="该分镜所有抽帧的 OBS 公网 URL 列表")
    # 豆包视觉模型返回的结构化字段 (v1 & v2 混合)
    description: Optional[str] = None
    subject: Optional[str] = None
    object: Optional[List[str]] = None
    movement: Optional[str] = None
    adjective: Optional[List[str]] = None
    search_tags: Optional[List[str]] = None
    marketing_tags: Optional[List[str]] = None
    appealing_audience: Optional[List[str]] = None
    visual_quality: Optional[List[float]] = None
    
    # v2 新增字段
    car_model: Optional[str] = None
    frame_size: Optional[str] = None
    resolution: Optional[str] = None
    video_duration: Optional[float] = None
    footage_type: Optional[str] = None
    shot_style: Optional[str] = None
    shot_type: Optional[str] = None
    camera_movement: Optional[str] = None
    scene_location: Optional[List[str]] = None
    car_color: Optional[str] = None
    product_status_scene: Optional[str] = None
    has_presenter: Optional[bool] = None
    generic_hq_road_run: Optional[bool] = None
    person_detail: Optional[List[str]] = None
    key_words: Optional[List[str]] = None
    text: Optional[List[str]] = None
    video_usage: Optional[List[str]] = None
    design_adjectives: Optional[List[str]] = None
    function_adjectives: Optional[List[str]] = None
    design_selling_points: Optional[List[str]] = None
    function_selling_points: Optional[List[str]] = None
    scenario_a: Optional[List[str]] = None
    scenario_b: Optional[List[str]] = None
    marketing_phrases: Optional[List[str]] = None
    topic: Optional[str] = None
    weather: Optional[str] = None
    time: Optional[str] = None

    analysis_doc_id: Optional[str] = Field(default=None, description="OpenSearch 文档 ID")
    error: Optional[str] = Field(default=None, description="若该分镜分析失败, 这里记录错误信息")


class VideoAnalysisHistoryItem(BaseModel):
    """一条历史分析记录"""
    id: str = Field(..., description="历史记录唯一 ID")
    name: str = Field(..., description="展示用名称, 通常为上传视频文件名+时间戳")
    time: str = Field(..., description="生成时间 ISO 字符串")
    video_url: Optional[str] = Field(default=None, description="原视频的 OBS 公网 URL（可选）")
    workspace: str = Field(default="v1", description="工作区标识，如 v1 / v2")
    cards: List[ShotCard] = Field(default_factory=list, description="该次分析的分镜卡片列表")
    request_id: Optional[str] = Field(default=None, description="联表 http_request_traces.id")


class HistorySaveRequest(BaseModel):
    """保存/覆盖全部历史记录"""
    history: List[VideoAnalysisHistoryItem]


class HistoryUpdateRequest(BaseModel):
    """局部更新: 追加或替换某一条历史记录"""
    item: VideoAnalysisHistoryItem
