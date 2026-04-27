from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, RootModel
# 假设你的枚举定义在这里
from models.pydantic.opensearch_index import index_v2_enums


def _enum_hint(choices: List[str]) -> str:
    return "固定枚举，可选：" + "/".join([c for c in choices if c])


class SeedtextSegmentTagItem(BaseModel):
    """
    基于口播文案切分的分镜打标模型。
    每个 Item 代表文案中的一个“短句/断句”，并为其预设匹配的视觉标签。
    """

    index: int = Field(0, description="原始脚本序号")
    id: int = Field(..., description="分镜序号，从 1 开始递增")

    # --- 核心文案字段 ---
    segment_text: str = Field(
        ...,
        description="该分段的口播文本（建议 3 秒口播长度），通常是一个短句。",
        examples=["同级唯一，全系标配大厂底盘。"]
    )

    # --- 视觉参数对齐 (与 SceneAnalysisResultV2 一致) ---
    video_usage: List[str] = Field(
        ...,
        description=f"素材用途（{_enum_hint(index_v2_enums.VIDEO_USAGE_CHOICES)}）。建议 1-2 个。",
        min_length=1, max_length=2
    )
    shot_style: str = Field(
        ...,
        description=f"建议镜头风格（{_enum_hint(index_v2_enums.SHOT_STYLE_CHOICES)}）。",
    )
    shot_type: str = Field(
        ...,
        description=f"建议镜头景别（{_enum_hint(index_v2_enums.SHOT_TYPE_CHOICES)}）。",
    )
    movement: str = Field(
        ...,
        description=f"画面核心动作（{_enum_hint(index_v2_enums.MOVEMENT_CHOICES)}）。",
    )

    # --- 描述与卖点对齐 ---
    # 将原 tag_list 拆解为更精准的对齐字段
    design_selling_points: List[str] = Field(
        ...,
        description="该片段涉及的设计/实体卖点（2-4 个），如：真皮座椅、激光雷达。",
        min_length=1, max_length=4
    )
    function_selling_points: List[str] = Field(
        ...,
        description="该片段涉及的功能/性能卖点（2-4 个），如：四轮转向、无感减震。",
        min_length=1, max_length=4
    )

    # --- 营销与受众对齐 ---
    marketing_phrases: List[str] = Field(
        ...,
        description="口播式检索短句（1-3 个），贴近用户口语，如：过弯不甩尾、颠簸不晕车。",
        min_length=1, max_length=3
    )
    marketing_tags: List[str] = Field(
        ...,
        description="营销场景标签（如：产品展示、功能讲解、氛围烘托）。",
        min_length=1
    )
    appealing_audience: List[str] = Field(
        ...,
        description="目标受众标签（如：新手司机、家庭用户）。",
        min_length=1
    )

    # --- 场景预设 (可选，帮助 AI 构思画面) ---
    scene_location: List[str] = Field(
        default=["城市道路"],
        description="建议画面场景（如：冰雪、地库、公路）。",
        max_length=2
    )

    extra_tags: Optional[List[str]] = Field(
        default=None,
        description="额外标签, 可以附上没有适合字段的标签。"
    )

class SeedtextSegmentResultEnvelope(BaseModel):
    """
    最终返回的封装格式，确保 AI 输出稳定的 JSON 结构
    """
    segment_result: List[SeedtextSegmentTagItem] = Field(
        ...,
        description="按顺序排列的分镜标签列表"
    )

# 使用示例：
# SeedtextSegmentResultEnvelope.model_validate(ai_json_response)