from typing import List, Optional
from pydantic import BaseModel, Field

from models.pydantic.opensearch_index import index_v2_enums


def _enum_hint(choices: List[str]) -> str:
    """
    Build a compact enum hint string for prompt/schema descriptions.
    Example: "固定枚举，可选：A/B/C"
    """
    return "固定枚举，可选：" + "/".join([c for c in choices if c])


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


class SceneAnalysisResultV2(BaseModel):
    description: str = Field(
        ...,
        description=(
            "对当前视频片段画面的整体客观描述。要求尽量“可检索、可复用”，"
            "主体是谁/是什么、在做什么、画面有哪些关键元素、镜头/光线/构图的客观特征。"
            "避免广告语、夸张形容、主观评价；"
        ),
        examples=[
            "一辆SUV在冰雪覆盖的坡道上向上行驶，背景为晴朗蓝天，路面有积雪与冰面，尾部扬起细微雾气，画面明亮清晰。"
        ],
    )

    subject: str = Field(
        ...,
        description=(
            "画面核心主体（1 个），尽量使用“具体可检索”的名称："
            "例如“智己LS6汽车”“驾驶员”“中控大屏”“后排座椅”“激光雷达”“电池”。"
            "避免使用泛化词（如“车辆”“东西”）。"
        ),
        examples=["智己LS6汽车"],
    )

    object: Optional[List[str]] = Field(
        default=None,
        description="重要客体（只允许车或乘客相关：部件/座舱/屏幕/方向盘/驾驶员等；不要写环境）。1-4 个短词名词。",
        examples=[["方向盘", "中控大屏", "老人", "儿童"]],
        min_length=1,
        max_length=4,
    )

    movement: str = Field(
        ...,
        description=(
            "核心动作（单值）。从标准动作候选集中选择，不包含环境信息、不包含结果或评价。"
            f"{_enum_hint(index_v2_enums.MOVEMENT_CHOICES)}"
        ),
        examples=["掉头"],
    )

    # === Fields that Doubao needs to understand for IndexV2 ===
    footage_type: str = Field(
        ...,
        description=f"画面类型（{_enum_hint(index_v2_enums.FOOTAGE_TYPE_CHOICES)}）。",
        examples=["CG"],
    )
    shot_style: str = Field(
        ...,
        description=f"镜头风格/拍摄方式（{_enum_hint(index_v2_enums.SHOT_STYLE_CHOICES)}）。",
        examples=["车内POV"],
    )
    shot_type: str = Field(
        ...,
        description=(
            f"镜头景幅/景别（{_enum_hint(index_v2_enums.SHOT_TYPE_CHOICES)}）。"
            "决定主体在画面中的占比：大远景/远景/中景/特写/大特写。"
        ),
        examples=["特写"],
    )
    scene_location: List[str] = Field(
        ...,
        description="画面场景/路况/空间类型（1-4 个），短词名词化，如：山路/沙漠/地库/公路/冰雪/城市道路/狭窄街道/赛道/展厅 等。",
        examples=[["地库", "城市道路", "冰雪"]],
        min_length=1,
        max_length=4,
    )
    car_color: str = Field(
        ...,
        description=f"车色（{_enum_hint(index_v2_enums.CAR_COLOR_CHOICES)}）。",
        examples=["黑"],
    )
    car_color_detail: str = Field(
        "",
        description="车色细节补充（可选），如：哑光黑/珠光白/渐变涂装/贴膜等；没有就留空。",
        examples=["哑光黑"],
    )
    product_status_scene: str = Field(
        ...,
        description=f"产品状态场景（{_enum_hint(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}）。",
        examples=["静态内饰"],
    )
    has_presenter: bool = Field(
        ...,
        description="是否包含出镜讲解员/达人/主持人（只根据画面可见判断）。",
        examples=[False],
    )

    person_detail: List[str] = Field(
        ...,
        description=(
            f"人物细分标签（{_enum_hint(index_v2_enums.PERSON_DETAIL_CHOICES)}）。"
            "无人物时只填 [无人物]；多人时可以填多个，例如 [男性, 女性] 或 [小孩, 女性]。"
        ),
        examples=[["无人物"], ["男性"], ["男性", "女性"]],
        min_length=1,
        max_length=3,
    )

    key_traits: List[str] = Field(
        ...,
        description=(
            f"素材关键特点标签（{_enum_hint(index_v2_enums.KEY_TRAITS_CHOICES)}）。"
            "这是素材里最重要的可过滤特点。可多选，但不要误选，必须从枚举里选。"
        ),
        examples=[["续航", "大电池", "充电快", "路跑"], ["地库掉头", "狭窄街道", "新手"], ["安静", "降噪", "带人的内饰"]],
        min_length=1,
        max_length=12,
    )
    weather: str = Field(
        ...,
        description=f"天气（{_enum_hint(index_v2_enums.WEATHER_CHOICES)}）。",
        examples=["雪天"],
    )
    time: str = Field(
        ...,
        description=f"时间（{_enum_hint(index_v2_enums.TIME_CHOICES)}）。",
        examples=["白天"],
    )
    video_usage: List[str] = Field(
        ...,
        description=(
            f"素材用途（{_enum_hint(index_v2_enums.VIDEO_USAGE_CHOICES)}）。"
            "尽量只给 1 个；如确实同时满足两个方向且都有用，才给多个。"
        ),
        examples=[["功能讲解"], ["建立空间感", "烘托氛围"]],
        min_length=1,
        max_length=3,
    )

    # Split adjectives + selling points into two cohesive lists
    design_adjectives: List[str] = Field(
        ...,
        description="外观/设计/质感类形容词（2-4 个），偏可见外观与材质观感。",
        examples=[["精致", "高级", "现代"]],
        min_length=2,
        max_length=4,
    )
    function_adjectives: List[str] = Field(
        ...,
        description="性能/体验类形容词（2-4 个），偏驾驶/舒适/安静/稳定等体验特性。",
        examples=[["安静", "稳定", "舒适"]],
        min_length=2,
        max_length=4,
    )
    design_selling_points: List[str] = Field(
        ...,
        description="外观/设计卖点（2-4 个），偏实体/部件/可见结构（不要写环境/动作）。",
        examples=[["真皮座椅", "大屏中控", "座舱材质"]],
        min_length=2,
        max_length=4,
    )
    function_selling_points: List[str] = Field(
        ...,
        description="性能/功能卖点（2-4 个），偏能力模块/特殊功能（不要写环境/动作）。",
        examples=[["一键AI泊车", "雨夜模式", "爆胎稳定控制","快充"]],
        min_length=2,
        max_length=4,
    )

    scenario_a: List[str] = Field(
        ...,
        description="生活/用车场景列表 A（1-4 个），同列表内语义尽量靠拢。",
        examples=[["通勤", "地库停车"]],
        min_length=1,
        max_length=4,
    )
    scenario_b: List[str] = Field(
        ...,
        description="生活/用车场景列表 B（1-4 个），与 A 尽量不同。",
        examples=[["周末自驾", "长途出行"]],
        min_length=1,
        max_length=4,
    )

    marketing_phrases: List[str] = Field(
        ...,
        description=(
            "营销短句/口播式检索短语（1-6 个）。"
            "用用户会搜索/会说的口吻，不要用“演示/展示”。"
            "例：雨夜看得清、堵车跟车不累、地库一把掉头、停车一把进"
        ),
        examples=[["地库一把掉头", "停车一把进"]],
        min_length=1,
        max_length=6,
    )

    marketing_tags: List[str] = Field(..., description="用于商业混剪的营销场景标签（如：产品展示、使用场景、对比评测、教程演示）")
    appealing_audience: List[str] = Field(..., description="目标受众标签（如：汽车爱好者、家庭用户、新手司机、北方车主等）")
    # visual_quality removed in V2 test schema (can be reintroduced later if needed)