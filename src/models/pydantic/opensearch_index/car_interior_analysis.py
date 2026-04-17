from typing import List, Optional
from pydantic import Field
from .base_index import BaseIndex


class CarInteriorAnalysis(BaseIndex):
    class Meta:
        index_name = "car_interior_analysis"
    
    id: Optional[str] = Field(
        None,
        description="文档唯一标识符"
    )
    
    description: str = Field(
        "",
        description="视频片段的详细描述"
    )
    
    subject: str = Field(
        "",
        description="内容的主题/核心对象"
    )
    
    object: List[str] = Field(
        default_factory=list,
        description="识别到的对象列表"
    )
    
    movement: str = Field(
        "",
        description="识别到的动作/运动描述"
    )
    
    adjective: List[str] = Field(
        default_factory=list,
        description="描述性形容词列表"
    )
    
    search_tags: List[str] = Field(
        default_factory=list,
        description="搜索标签列表"
    )
    
    marketing_tags: List[str] = Field(
        default_factory=list,
        description="营销标签列表"
    )
    
    appealing_audience: List[str] = Field(
        default_factory=list,
        description="目标受众列表"
    )
    
    visual_quality: float = Field(
        0.0,
        description="视觉质量评分 (0-10)"
    )
    
    description_vector: Optional[List[float]] = Field(
        None,
        description="description 字段的向量表示"
    )
    
    subject_vector: Optional[List[float]] = Field(
        None,
        description="subject 字段的向量表示"
    )
    
    combined_vector: Optional[List[float]] = Field(
        None,
        description="组合字段的向量表示"
    )
    
    @classmethod
    def from_analysis_result(cls, analysis_result: dict, embedding_model) -> 'CarInteriorAnalysis':
        description = analysis_result.get('description', '')
        subject = analysis_result.get('subject', '')
        objects = analysis_result.get('object', [])
        adjectives = analysis_result.get('adjective', [])
        search_tags = analysis_result.get('search_tags', [])
        
        combined_text = f"{description} {subject} {' '.join(objects)} {' '.join(adjectives)} {' '.join(search_tags)}"
        
        return cls(
            id=analysis_result.get('id'),
            description=description,
            subject=subject,
            object=objects,
            movement=analysis_result.get('movement', ''),
            adjective=adjectives,
            search_tags=search_tags,
            marketing_tags=analysis_result.get('marketing_tags', []),
            appealing_audience=analysis_result.get('appealing_audience', []),
            visual_quality=analysis_result.get('visual_quality', 0.0),
            description_vector=cls._generate_embedding(description, embedding_model),
            subject_vector=cls._generate_embedding(subject, embedding_model),
            combined_vector=cls._generate_embedding(combined_text, embedding_model)
        )
    
    @staticmethod
    def _generate_embedding(text: str, model) -> List[float]:
        if not text:
            return [0.0] * 384
        embedding = model.encode(text)
        return embedding.tolist()
