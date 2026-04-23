from typing import Annotated, List, Optional
from pydantic import Field
from .base_index import BaseIndex
from .markers import Keyword, Text, Vector, Float


prompt_text = """
\t你是一个专业的视频分镜分析师，同时你也了解用户在搜索视频时的习惯。
    请分析这些视频片段里的画面。
    【重要规则】
    1. 提取 object 时，请使用最通用的词汇，贴合日常口语表达。
    2. search_tags 字段极其重要，请发挥联想，写出用户搜什么词时应该看到这个视频。

    ### 1. 营销场景标签
    - **场景类型**：判断属于哪种营销场景
      可选：产品展示、使用场景、情感共鸣、品牌故事、教程演示、对比评测、生活方式展示


    - **目标受众**：这个画面最能打动哪类人群？
      示例：Z世代、精致妈妈、职场精英、银发族、健身达人、美食爱好者

    ### 2. 商业价值评估 (0-10分)
    - 产品展示清晰度：画面是否适合展示产品细节
    - 情感共鸣度：是否能引起观众情感共鸣
    - 画面美感度：构图、光线、色彩的专业程度
    - 通用适配性：是否容易与其他素材混剪
    """


class CarInteriorAnalysis(BaseIndex):
    class Meta:
        index_name = "car_interior_analysis"
    
    id: Optional[str] = Field(
        None,
        description="文档唯一标识符"
    )
    
    description: Annotated[str, Text(2.0)] = Field(
        "",
        description="视频片段的详细描述",
    )
    
    subject: Annotated[str, Text(1.5)] = Field(
        "",
        description="内容的主题/核心对象",
    )
    
    object: Annotated[List[str], Keyword(1.2)] = Field(
        default_factory=list,
        description="识别到的对象列表",
    )
    
    movement: Annotated[str, Text(1.0)] = Field(
        "",
        description="识别到的动作/运动描述",
    )
    
    adjective: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list,
        description="描述性形容词列表",
    )
    
    search_tags: Annotated[List[str], Keyword(2.5)] = Field(
        default_factory=list,
        description="搜索标签列表",
    )
    
    marketing_tags: Annotated[List[str], Keyword(1.5)] = Field(
        default_factory=list,
        description="营销标签列表",
    )
    
    appealing_audience: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list,
        description="目标受众列表",
    )
    
    clarity_score: Annotated[float, Float()] = Field(0.0, description="清晰度评分 (0-10)")
    composition_score: Annotated[float, Float()] = Field(0.0, description="构图评分 (0-10)")
    lighting_score: Annotated[float, Float()] = Field(0.0, description="光影/曝光评分 (0-10)")
    color_score: Annotated[float, Float()] = Field(0.0, description="色彩/调色评分 (0-10)")
    
    description_vector: Annotated[Optional[List[float]], Vector(384, 1.0)] = Field(
        None,
        description="description 字段的向量表示",
    )
    
    subject_vector: Annotated[Optional[List[float]], Vector(384, 1.5)] = Field(
        None,
        description="subject 字段的向量表示",
    )
    
    combined_vector: Annotated[Optional[List[float]], Vector(384, 2.0)] = Field(
        None,
        description="组合字段的向量表示",
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
            clarity_score=float((analysis_result.get('visual_quality') or [0, 0, 0, 0])[0]),
            composition_score=float((analysis_result.get('visual_quality') or [0, 0, 0, 0])[1]),
            lighting_score=float((analysis_result.get('visual_quality') or [0, 0, 0, 0])[2]),
            color_score=float((analysis_result.get('visual_quality') or [0, 0, 0, 0])[3]),
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
