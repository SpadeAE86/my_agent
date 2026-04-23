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
    # 豆包视觉模型返回的结构化字段
    description: Optional[str] = None
    subject: Optional[str] = None
    object: Optional[List[str]] = None
    movement: Optional[str] = None
    adjective: Optional[List[str]] = None
    search_tags: Optional[List[str]] = None
    marketing_tags: Optional[List[str]] = None
    appealing_audience: Optional[List[str]] = None
    visual_quality: Optional[List[float]] = None
    error: Optional[str] = Field(default=None, description="若该分镜分析失败, 这里记录错误信息")


class VideoAnalysisHistoryItem(BaseModel):
    """一条历史分析记录"""
    id: str = Field(..., description="历史记录唯一 ID")
    name: str = Field(..., description="展示用名称, 通常为上传视频文件名+时间戳")
    time: str = Field(..., description="生成时间 ISO 字符串")
    video_url: Optional[str] = Field(default=None, description="原视频的 OBS 公网 URL（可选）")
    cards: List[ShotCard] = Field(default_factory=list, description="该次分析的分镜卡片列表")


class HistorySaveRequest(BaseModel):
    """保存/覆盖全部历史记录"""
    history: List[VideoAnalysisHistoryItem]


class HistoryUpdateRequest(BaseModel):
    """局部更新: 追加或替换某一条历史记录"""
    item: VideoAnalysisHistoryItem
