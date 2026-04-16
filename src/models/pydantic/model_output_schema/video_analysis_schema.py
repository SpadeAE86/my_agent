from typing import List, Optional
from pydantic import BaseModel, Field


class SceneAnalysisResult(BaseModel):
    description: str = Field(..., description="对当前视频片段画面的整体客观描述，包含场景、人物、光线等")
    subject: str = Field(..., description="画面核心主体")
    object: Optional[List[str]] = Field(default=None, description="画面中的重要客体，请使用简短通俗的词语")
    movement: str = Field(..., description="主体正在进行的动作，使用动宾短语描述")
    adjective: List[str] = Field(..., description="形容画面氛围的词语")
    search_tags: List[str] = Field(...,
                                   description="站在用户角度，提取5-8个用户可能会用来搜索这段画面的通俗关键词或短语，例如：做饮料、甜品、田园风、奶茶、治愈系")
    # 营销标签（新增）
    marketing_tags: List[str] = Field(..., description="用于商业混剪的营销场景标签")

    # 目标受众
    appealing_audience: List[str] = Field(..., description="适合观看这段视频的目标受众标签，例如：年轻人、学生、职场人士、家庭主妇等")

    # 视觉质量评估
    visual_quality: List[float] = Field(..., description="视觉质量评分")