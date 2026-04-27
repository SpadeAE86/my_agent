from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import Field

from .base_index import BaseIndex
from .markers import Boolean, Keyword, Text, Vector, Float
from . import index_v2_enums


CN_ANALYZER = "standard"  # change to "ik_max_word" after installing IK plugin


class CarInteriorAnalysisV2(BaseIndex):
    """
    Index V2:
    - More structured filter fields (keyword/boolean)
    - Fewer, more targeted vector fields (方案A：按字段分别向量)
    - Avoids OpenSearch multi-field by using explicit *_text companions when needed.
    """

    class Meta:
        index_name = "car_interior_analysis_v2"
        # If your OpenSearch cluster has IK plugin installed, set CN_ANALYZER="ik_max_word".
        settings = None

    # --- Identity / linkage ---
    id: Optional[str] = Field(None, description="文档唯一标识符（建议 history_id + scene_id 组合）")

    # --- Deterministic / metadata fields (mostly filterable) ---
    car_model: Annotated[str, Keyword(1.0)] = Field("未知", description="车型（外部标签，默认未知）")
    frame_size: Annotated[str, Keyword(1.0)] = Field("未知", description="尺寸/比例（由帧宽高计算）")
    resolution: Annotated[str, Keyword(0.7)] = Field(
        "未知", description="分辨率（如 1920x1080；由帧宽高计算，用于区分素材质量）"
    )
    video_duration: Annotated[float, Float()] = Field(
        0.0, description="视频/分镜时长（秒）。用于过滤不满足时长的片段。"
    )

    footage_type: Annotated[str, Keyword(1.2)] = Field("未知", description="画面类型（固定枚举，如 CG/实拍/直播切片等）")
    shot_style: Annotated[str, Keyword(1.0)] = Field("未知", description="镜头风格/拍摄方式（固定枚举）")
    shot_type: Annotated[str, Keyword(1.0)] = Field(
        "未知",
        description=f"镜头景幅/景别（枚举）：{index_v2_enums.SHOT_TYPE_CHOICES}。",
    )

    scene_location: Annotated[List[str], Text(1.0, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="画面场景（可多值，偏地点/路况/空间类型）"
    )

    car_color: Annotated[str, Keyword(1.0)] = Field("未知", description="车色（固定枚举）")
    car_color_detail: Annotated[str, Text(0.7, analyzer=CN_ANALYZER)] = Field(
        "", description="车色细节补充（如 哑光黑/珠光白/涂装/贴膜 等）"
    )

    product_status_scene: Annotated[str, Keyword(1.0)] = Field(
        "未知", description="产品状态场景（标准化：静态/路跑 + 内饰/外观/空间/发布会 等）"
    )
    product_status_scene_text: Annotated[str, Text(0.6, analyzer=CN_ANALYZER)] = Field(
        "", description="产品状态场景（文本兜底，便于检索 内饰/外观 等关键词）"
    )

    has_presenter: Annotated[Optional[bool], Boolean()] = Field(
        None, description="是否包含出镜讲解员/达人/主持人"
    )

    person_detail: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list,
        description=f"人物细分标签（枚举，可多值）：{index_v2_enums.PERSON_DETAIL_CHOICES}。多人时可同时包含多个（如 男性+女性 或 成人+小孩）。",
    )

    key_traits: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list,
        description=f"素材关键特点（枚举，可多值）：{index_v2_enums.KEY_TRAITS_CHOICES}。",
    )

    weather: Annotated[str, Keyword(1.0)] = Field("未知", description="天气（固定枚举）")
    time: Annotated[str, Keyword(1.0)] = Field("未知", description="时间（固定枚举，如 白天/夜晚/黄昏/室内）")
    video_usage: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list, description="素材用途（枚举，可多值；尽量 1 个，必要时可多个）"
    )

    # --- Core understanding fields ---
    description: Annotated[str, Text(2.0, analyzer=CN_ANALYZER)] = Field("", description="画面客观描述（可全文检索）")
    movement: Annotated[str, Keyword(1.2)] = Field("未知", description="核心动作（固定枚举，单值）")

    subject: Annotated[str, Text(1.3, analyzer=CN_ANALYZER)] = Field("", description="主体（文本，允许多表述）")
    object: Annotated[List[str], Text(1.0, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="客体/部件/乘客相关对象（多值文本）"
    )

    # --- Two-list design to force semantic cohesion ---
    # Selling points: design/entity-focused vs performance/function-focused
    design_selling_points: Annotated[List[str], Text(1.2, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="外观/设计卖点（偏实体/部件/可见结构）"
    )
    function_selling_points: Annotated[List[str], Text(1.5, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="性能/功能卖点（偏能力模块/抽象功能）"
    )

    # Adjectives (used as "product detail" proxy): split into two cohesive lists
    design_adjectives: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list, description="外观/质感类形容词（keyword，多值）"
    )
    function_adjectives: Annotated[List[str], Keyword(1.0)] = Field(
        default_factory=list, description="性能/体验类形容词（keyword，多值）"
    )

    scenario_a: Annotated[List[str], Text(1.0, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="生活/用车场景列表 A（语义尽量靠拢）"
    )
    scenario_b: Annotated[List[str], Text(1.0, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="生活/用车场景列表 B（与 A 语义尽量不同）"
    )

    marketing_phrases: Annotated[List[str], Text(1.4, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="营销短句/口播式检索短语（用户会怎么搜/怎么说）"
    )

    appealing_audience: Annotated[List[str], Text(0.8, analyzer=CN_ANALYZER)] = Field(
        default_factory=list, description="目标受众（文本，多值）"
    )

    # --- Vectors (方案A：按字段分别向量) ---
    description_vector: Annotated[Optional[List[float]], Vector(384, 1.0)] = Field(
        None, description="description 的向量"
    )
    function_selling_points_vector: Annotated[Optional[List[float]], Vector(384, 1.2)] = Field(
        None, description="function_selling_points 的向量"
    )
    design_selling_points_vector: Annotated[Optional[List[float]], Vector(384, 0.9)] = Field(
        None, description="design_selling_points 的向量"
    )
    scenario_a_vector: Annotated[Optional[List[float]], Vector(384, 0.7)] = Field(
        None, description="scenario_a 的向量"
    )
    scenario_b_vector: Annotated[Optional[List[float]], Vector(384, 0.7)] = Field(
        None, description="scenario_b 的向量"
    )
    marketing_phrases_vector: Annotated[Optional[List[float]], Vector(384, 1.3)] = Field(
        None, description="marketing_phrases 的向量"
    )

    @classmethod
    def from_analysis_result(cls, analysis_result: dict, embedding_model) -> "CarInteriorAnalysisV2":
        """
        Build an index document from Doubao output dict (SceneAnalysisResultV2).
        embedding_model: SentenceTransformer-like object with encode(text)->np.ndarray
        """

        def join_list(xs: List[str]) -> str:
            return " ".join([x for x in (xs or []) if x])

        description = analysis_result.get("description", "") or ""
        function_sp = analysis_result.get("function_selling_points", []) or []
        design_sp = analysis_result.get("design_selling_points", []) or []
        scenario_a = analysis_result.get("scenario_a", []) or []
        scenario_b = analysis_result.get("scenario_b", []) or []
        marketing_phrases = analysis_result.get("marketing_phrases", []) or []

        return cls(
            id=analysis_result.get("id"),
            car_model=analysis_result.get("car_model", "未知"),
            frame_size=analysis_result.get("frame_size", "未知"),
            resolution=analysis_result.get("resolution", "未知"),
            video_duration=float(analysis_result.get("video_duration", 0.0) or 0.0),
            footage_type=analysis_result.get("footage_type", "未知"),
            shot_style=analysis_result.get("shot_style", "未知"),
            shot_type=analysis_result.get("shot_type", "未知"),
            scene_location=analysis_result.get("scene_location", []) or [],
            car_color=analysis_result.get("car_color", "未知"),
            car_color_detail=analysis_result.get("car_color_detail", "") or "",
            product_status_scene=analysis_result.get("product_status_scene", "未知"),
            product_status_scene_text=analysis_result.get("product_status_scene_text", "") or "",
            has_presenter=analysis_result.get("has_presenter", None),
            person_detail=analysis_result.get("person_detail", []) or [],
            key_traits=analysis_result.get("key_traits", []) or [],
            weather=analysis_result.get("weather", "未知"),
            time=analysis_result.get("time", "未知"),
            video_usage=(
                [analysis_result.get("video_usage")]
                if isinstance(analysis_result.get("video_usage"), str)
                else (analysis_result.get("video_usage", []) or [])
            ),
            description=description,
            movement=analysis_result.get("movement", "未知"),
            subject=analysis_result.get("subject", "") or "",
            object=analysis_result.get("object", []) or [],
            design_selling_points=design_sp,
            function_selling_points=function_sp,
            design_adjectives=analysis_result.get("design_adjectives", []) or [],
            function_adjectives=analysis_result.get("function_adjectives", []) or [],
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            marketing_phrases=marketing_phrases,
            appealing_audience=analysis_result.get("appealing_audience", []) or [],
            description_vector=cls._generate_embedding(description, embedding_model),
            function_selling_points_vector=cls._generate_embedding(join_list(function_sp), embedding_model),
            design_selling_points_vector=cls._generate_embedding(join_list(design_sp), embedding_model),
            scenario_a_vector=cls._generate_embedding(join_list(scenario_a), embedding_model),
            scenario_b_vector=cls._generate_embedding(join_list(scenario_b), embedding_model),
            marketing_phrases_vector=cls._generate_embedding(join_list(marketing_phrases), embedding_model),
        )

    @staticmethod
    def _generate_embedding(text: str, model) -> Optional[List[float]]:
        """
        Return None for empty text to avoid indexing a zero-vector.
        Some OpenSearch knn_vector configs (e.g. cosinesimil) reject zero vectors.
        """
        if not text:
            return None
        emb = model.encode(text)
        v = emb.tolist()
        # Defensive: if an upstream model ever returns a zero vector, skip it.
        try:
            if all(float(x) == 0.0 for x in v):
                return None
        except Exception:
            pass
        return v

