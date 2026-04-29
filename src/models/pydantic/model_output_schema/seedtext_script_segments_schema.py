from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, RootModel
# 假设你的枚举定义在这里
from models.pydantic.opensearch_index import index_v2_enums


def _enum_hint(choices: List[str]) -> str:
    return "固定枚举，可选：" + "/".join([c for c in choices if c])

#
# IMPORTANT DESIGN NOTE
# ---------------------
# Do NOT mix "storyboard planning" and "strict tagging" into a single schema.
# - Stage 1: storyboard plan (human-readable; easier for the model to do well)
# - Stage 2: strict tags (IndexV2-aligned; easier to validate and index)
#

class SeedtextStoryboardSegment(BaseModel):
    """Stage 1 (分镜规划): 口播短句 -> 可拍的镜头规划。

    特点：
    - 字段以“自然语言可读”为主，但仍要求结构稳定
    - shot_type / shot_style / video_usage 用枚举以方便后续对齐检索
    - selling_point/adjective 等允许更自然的短语，不强制拆成 IndexV2 的两套列表
    """

    index: int = Field(0, description="原始脚本序号")
    id: int = Field(..., description="分镜序号，从 1 开始递增（Stage1 与 Stage2 需保持一致）")

    segment_text: str = Field(
        ...,
        description="该分段的口播文本（建议约 2-4 秒的短句），通常是一个信息点。",
        examples=["同级唯一，全系标配大厂底盘。"]
    )

    duration: float = Field(
        ...,
        description="建议时长（秒）。用于指导剪辑节奏，通常 1.5-4.0 秒。",
        ge=0.5,
        le=10.0,
        examples=[3.0],
    )

    description: str = Field(
        ...,
        description="建议画面描述（自然语言即可）：主体、动作、场景、镜头意图。不要写口播原文复读。",
        examples=["车内POV，中控大屏展示一键泊车界面，地库场景，画面干净清晰。"],
    )

    video_usage: str = Field(
        ...,
        description=f"该分镜的用途（{_enum_hint(index_v2_enums.VIDEO_USAGE_CHOICES)}）。",
        examples=["功能讲解"],
    )
    shot_style: str = Field(
        ...,
        description=f"建议镜头风格（{_enum_hint(index_v2_enums.SHOT_STYLE_CHOICES)}）。",
    )
    shot_type: str = Field(
        ...,
        description=f"建议镜头景别（{_enum_hint(index_v2_enums.SHOT_TYPE_CHOICES)}）。",
    )

    subject: str = Field(
        ...,
        description="画面核心主体（自然语言短词）。如：智己LS6 / 中控大屏 / 后排座椅。",
        examples=["智己LS6"],
    )
    object: List[str] = Field(
        ...,
        description="画面关键客体（短词列表）。如：方向盘/仪表盘/地库/坡道/雨夜。",
        min_length=0,
        max_length=8,
        examples=[["中控大屏", "方向盘"]],
    )

    scene_location: List[str] = Field(
        default_factory=list,
        description="画面场景关键词（短词列表）。如：地库/冰雪/雨夜/高速/城市道路。",
        max_length=4,
        examples=[["地库"]],
    )

    selling_point: List[str] = Field(
        ...,
        description="该句口播想表达的核心卖点（短词/短语列表，1-4 个）。",
        min_length=1,
        max_length=4,
        examples=[["一键AI泊车", "自动识别车位"]],
    )

    adjective: List[str] = Field(
        default_factory=list,
        description="氛围/体验形容词（短词列表，0-6 个），如：安静/稳/高级/舒适/清晰。",
        max_length=6,
        examples=[["省心", "顺滑"]],
    )


class SeedtextStoryboardEnvelope(BaseModel):
    """Stage 1 输出封装"""

    storyboard: List[SeedtextStoryboardSegment] = Field(..., description="按顺序排列的分镜规划列表")


class SeedtextIndexTagsSegment(BaseModel):
    """Stage 2 (标准标签抽取): 把 Stage 1 的每个分镜规划映射成 IndexV2 对齐字段。

    特点：
    - 字段尽量对齐 `car_interior_analysis_v2` 的 schema
    - 需要枚举/keyword 时必须从 choices 里选（允许“未知”兜底）
    - 输出用于检索/入库，不追求文学表达

    与 OpenSearch `car_interior_analysis_v2` 向量字段（见 `script_match_service.match_script_tags_segments`）：
    - `design_adjectives` / `function_adjectives` 各对应一路 knn：`design_adjectives_vector`、`function_adjectives_vector`。
    - `scenario_a` / `scenario_b` 仍是两组文本列表（BM25 可分开命中）；入库向量将两组拼接后 embed 为单一 `scenario_vector`。
    """

    index: int = Field(0, description="原始脚本序号（与 Stage1 对齐）")
    id: int = Field(..., description="分镜序号（与 Stage1 对齐）")

    segment_text: str = Field(..., description="对应的口播短句（来自 Stage1）")
    duration: float = Field(
        ...,
        description="该分镜的目标时长（秒）。必须与 Stage1 对应分镜的 duration 完全一致（直接复制，不要改写）。",
        ge=0.5,
        le=10.0,
        examples=[3.0],
    )

    description: str = Field(..., description="可检索的画面描述（短句，客观）")
    movement: str = Field(..., description=f"核心动作（{_enum_hint(index_v2_enums.MOVEMENT_CHOICES)}）。")
    subject: str = Field(..., description="画面核心主体（短词）")
    object: Optional[List[str]] = Field(
        default=None,
        description=(
            "客体（车或乘客相关短词）。允许做同义扩展以提高命中率，例如："
            "后排/后排座椅/乘客/人坐后排；方向盘/中控屏/车机/仪表盘；轮胎/轮毂/刹车等。"
        ),
        max_length=10,
    )

    footage_type: str = Field(
        index_v2_enums.UNKNOWN,
        description=(
            f"画面类型（{_enum_hint(index_v2_enums.FOOTAGE_TYPE_CHOICES)}）。"
            "尽量不要填“未知”：如果描述明显是实拍（户外/车内POV/跟拍等），优先填“生活实拍 ”；如果是从非常精致的角度，非常漂亮的色彩光泽运镜创意，优先填写“TVC切片”；如果是拍摄角度画面构图挑选的很好，但质量达不到成品，优先填写“专业摄影”，优先填写“专业拍摄”。只有完全无法判断才用“未知”。"
        ),
    )

    topic: str = Field(
        index_v2_enums.UNKNOWN,
        description=f"主题 topic（{_enum_hint(index_v2_enums.TOPIC_CHOICES)}）。单值，用于脚本 topic 匹配。",
    )

    text: List[str] = Field(
        default_factory=list,
        description="画面关键文字与数值（keyword 列表）。尽量收集屏幕/UI/字幕出现的文字与数值：NOA/Auto Park/800V/15分钟/310公里/1500km/4.79米/27.1英寸/5K 等。",
        max_length=16,
    )
    shot_style: str = Field(index_v2_enums.UNKNOWN, description=f"镜头风格（{_enum_hint(index_v2_enums.SHOT_STYLE_CHOICES)}）。")
    shot_type: str = Field(index_v2_enums.UNKNOWN, description=f"镜头景别（{_enum_hint(index_v2_enums.SHOT_TYPE_CHOICES)}）。")
    video_usage: List[str] = Field(
        ...,
        description=f"素材用途（{_enum_hint(index_v2_enums.VIDEO_USAGE_CHOICES)}）。建议 1-2 个。",
        min_length=1,
        max_length=2,
    )

    scene_location: List[str] = Field(
        default_factory=list,
        description="场景（尽量填 2-4 个短词以提高命中率，例如：地库/高速/山路/街区/露营/充电站/展厅）。",
        max_length=6,
    )

    # Loose -> strict splits (still keep them small)
    design_selling_points: List[str] = Field(
        default_factory=list, description="设计/实体卖点（尽量填 2-4 个短词/短语）", max_length=6
    )
    function_selling_points: List[str] = Field(
        default_factory=list, description="功能/性能卖点（尽量填 2-4 个短词/短语）", max_length=6
    )
    design_adjectives: List[str] = Field(
        default_factory=list,
        description="设计/质感形容词（尽量填 2-4 个）。索引向量字段：design_adjectives_vector。",
        max_length=6,
    )
    function_adjectives: List[str] = Field(
        default_factory=list,
        description="体验/性能形容词（尽量填 2-4 个）。索引向量字段：function_adjectives_vector。",
        max_length=6,
    )

    scenario_a: List[str] = Field(
        default_factory=list,
        description="用车/生活场景 A（尽量填 2-4 个）。与 B 分列便于多样化产出；向量入库与 B 合并为 scenario_vector。",
        max_length=6,
    )
    scenario_b: List[str] = Field(
        default_factory=list,
        description="用车/生活场景 B（尽量填 2-4 个），语义尽量与 A 区分；向量入库与 A 合并为 scenario_vector。",
        max_length=6,
    )

    marketing_phrases: List[str] = Field(
        default_factory=list,
        description="口播式检索短语（尽量填 3-8 个），贴近用户会搜的说法（短、口语化）。",
        max_length=10,
    )
    marketing_tags: List[str] = Field(default_factory=list, description="营销场景标签（尽量填 2-6 个）", max_length=8)
    appealing_audience: List[str] = Field(default_factory=list, description="目标受众（尽量填 2-6 个）", max_length=8)

    # Fields that Stage 2 may not be able to infer reliably from script; still try best-effort.
    car_color: str = Field(
        index_v2_enums.UNKNOWN,
        description=(
            f"车色（{_enum_hint(index_v2_enums.CAR_COLOR_CHOICES)}）。"
            "尽量不要填 未知：如果 storyboard/画面描述没有明确颜色，可按常见拍摄素材做保守推断（如 黑/白/灰），或结合产品/场景描述推断。"
        ),
    )
    product_status_scene: str = Field(
        index_v2_enums.UNKNOWN,
        description=f"产品状态场景（{_enum_hint(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}）。",
    )
    has_presenter: Optional[bool] = Field(default=None, description="是否含达人/讲解员（脚本无法判断可为 null）")
    weather: str = Field(
        index_v2_enums.UNKNOWN,
        description=(
            f"天气（{_enum_hint(index_v2_enums.WEATHER_CHOICES)}）。"
            "尽量不要填 未知：可根据 storyboard 里的场景词推断（如 雨夜->夜雨/雨天；冰雪->雪天/极寒；夏天/高温->酷暑；室内->阴天或晴天均可选其一）。"
        ),
    )
    time: str = Field(
        index_v2_enums.UNKNOWN,
        description=(
            f"时间（{_enum_hint(index_v2_enums.TIME_CHOICES)}）。"
            "尽量不要填 未知：室内->室内；出现夜/雨夜->夜晚；出现清晨/日出->清晨；否则默认白天。"
        ),
    )

    extra_tags: Optional[List[str]] = Field(
        default=None,
        description="无法归类但可能有助检索的额外短标签（鼓励有想象力，尽量 3-10 个；短词）。",
    )


class SeedtextIndexTagsEnvelope(BaseModel):
    """Stage 2 输出封装"""

    segment_result: List[SeedtextIndexTagsSegment] = Field(..., description="按顺序排列的分镜标签列表")


# Backward-compat aliases (keep old names to avoid breaking callers abruptly)
SeedtextSegmentTagItem = SeedtextIndexTagsSegment
SeedtextSegmentResultEnvelope = SeedtextIndexTagsEnvelope

# 使用示例：
# SeedtextSegmentResultEnvelope.model_validate(ai_json_response)