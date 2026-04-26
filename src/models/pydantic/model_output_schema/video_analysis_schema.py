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
            "例如“智己LS6汽车”“驾驶员”“中控大屏”“后排座椅”。"
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
            "示例：转弯/掉头/泊车/充电/静态展示"
        ),
        examples=["掉头"],
    )

    # === Fields that Doubao needs to understand for IndexV2 ===
    footage_type: str = Field(
        ...,
        description="画面类型（固定枚举），如：CG/原创实拍/KOL拍摄/TVC切片/直播切片/海报/未知。",
        examples=["CG"],
    )
    shot_style: str = Field(
        ...,
        description="镜头风格/拍摄方式（固定枚举），如：车内POV/车外跟拍/固定机位/手持/航拍/屏幕录制/展台转盘/未知。",
        examples=["车内POV"],
    )
    scene_location: List[str] = Field(
        ...,
        description="画面场景/路况/空间类型（1-4 个），短词名词化，如：地库/公路/冰雪/现代城区/赛道/展厅 等。",
        examples=[["地库", "城市道路"]],
        min_length=1,
        max_length=4,
    )
    car_color: str = Field(
        ...,
        description="车色（固定枚举），如：黑/白/灰/红/蓝/黄/紫/粉/茶/多种/其他涂装/未知。",
        examples=["黑"],
    )
    car_color_detail: str = Field(
        "",
        description="车色细节补充（可选），如：哑光黑/珠光白/渐变涂装/贴膜等；没有就留空。",
        examples=["哑光黑"],
    )
    product_status_scene: str = Field(
        ...,
        description="产品状态场景（标准化短词），如：静态内饰/静态外观/静态空间/路跑内饰/路跑外观/发布会现场/未知。",
        examples=["静态内饰"],
    )
    has_presenter: bool = Field(
        ...,
        description="是否包含出镜讲解员/达人/主持人（只根据画面可见判断）。",
        examples=[False],
    )
    weather: str = Field(
        ...,
        description="天气（固定枚举），如：晴天/阴天/雨天/雪天/雾天/夜雨/室内/未知。",
        examples=["雪天"],
    )
    time: str = Field(
        ...,
        description="时间（固定枚举），如：白天/夜晚/黄昏/清晨/室内/未知。",
        examples=["白天"],
    )
    video_usage: str = Field(
        ...,
        description="素材用途（固定枚举），如：产品展示/使用场景/功能讲解/对比评测/直播切片/海报静帧/未知。",
        examples=["功能讲解"],
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
        examples=[["一键AI泊车", "雨夜模式", "爆胎稳定控制"]],
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