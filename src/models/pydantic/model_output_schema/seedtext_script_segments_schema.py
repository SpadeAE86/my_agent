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
    """

    index: int = Field(0, description="原始脚本序号（与 Stage1 对齐）")
    id: int = Field(..., description="分镜序号（与 Stage1 对齐）")

    segment_text: str = Field(..., description="对应的口播短句（来自 Stage1）")

    description: str = Field(..., description="可检索的画面描述（短句，客观）")
    movement: str = Field(..., description=f"核心动作（{_enum_hint(index_v2_enums.MOVEMENT_CHOICES)}）。")
    subject: str = Field(..., description="画面核心主体（短词）")
    object: Optional[List[str]] = Field(default=None, description="客体（车或乘客相关，1-6 个短词）", max_length=6)

    footage_type: str = Field(index_v2_enums.UNKNOWN, description=f"画面类型（{_enum_hint(index_v2_enums.FOOTAGE_TYPE_CHOICES)}）。")
    shot_style: str = Field(index_v2_enums.UNKNOWN, description=f"镜头风格（{_enum_hint(index_v2_enums.SHOT_STYLE_CHOICES)}）。")
    shot_type: str = Field(index_v2_enums.UNKNOWN, description=f"镜头景别（{_enum_hint(index_v2_enums.SHOT_TYPE_CHOICES)}）。")
    video_usage: List[str] = Field(
        ...,
        description=f"素材用途（{_enum_hint(index_v2_enums.VIDEO_USAGE_CHOICES)}）。建议 1-2 个。",
        min_length=1,
        max_length=2,
    )

    scene_location: List[str] = Field(default_factory=list, description="场景（1-4 个短词）", max_length=4)

    # Loose -> strict splits (still keep them small)
    design_selling_points: List[str] = Field(default_factory=list, description="设计/实体卖点（0-4 个）", max_length=4)
    function_selling_points: List[str] = Field(default_factory=list, description="功能/性能卖点（0-4 个）", max_length=4)
    design_adjectives: List[str] = Field(default_factory=list, description="设计/质感形容词（0-4 个）", max_length=4)
    function_adjectives: List[str] = Field(default_factory=list, description="体验/性能形容词（0-4 个）", max_length=4)

    scenario_a: List[str] = Field(default_factory=list, description="场景 A（0-4 个）", max_length=4)
    scenario_b: List[str] = Field(default_factory=list, description="场景 B（0-4 个）", max_length=4)

    marketing_phrases: List[str] = Field(
        default_factory=list,
        description="口播式检索短语（1-6 个），贴近用户会搜的说法。",
        max_length=6,
    )
    marketing_tags: List[str] = Field(default_factory=list, description="营销场景标签（0-4 个）", max_length=4)
    appealing_audience: List[str] = Field(default_factory=list, description="目标受众（0-6 个）", max_length=6)

    # Fields that Stage 2 may not be able to infer reliably from script; keep as optional/unknown.
    car_color: str = Field(index_v2_enums.UNKNOWN, description=f"车色（{_enum_hint(index_v2_enums.CAR_COLOR_CHOICES)}）。")
    product_status_scene: str = Field(
        index_v2_enums.UNKNOWN,
        description=f"产品状态场景（{_enum_hint(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}）。",
    )
    has_presenter: Optional[bool] = Field(default=None, description="是否含达人/讲解员（脚本无法判断可为 null）")
    weather: str = Field(index_v2_enums.UNKNOWN, description=f"天气（{_enum_hint(index_v2_enums.WEATHER_CHOICES)}）。")
    time: str = Field(index_v2_enums.UNKNOWN, description=f"时间（{_enum_hint(index_v2_enums.TIME_CHOICES)}）。")

    extra_tags: Optional[List[str]] = Field(default=None, description="无法归类但可能有助检索的额外短标签")


class SeedtextIndexTagsEnvelope(BaseModel):
    """Stage 2 输出封装"""

    segment_result: List[SeedtextIndexTagsSegment] = Field(..., description="按顺序排列的分镜标签列表")


# Backward-compat aliases (keep old names to avoid breaking callers abruptly)
SeedtextSegmentTagItem = SeedtextIndexTagsSegment
SeedtextSegmentResultEnvelope = SeedtextIndexTagsEnvelope

# 使用示例：
# SeedtextSegmentResultEnvelope.model_validate(ai_json_response)