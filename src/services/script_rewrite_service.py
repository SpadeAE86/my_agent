from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from models.pydantic.opensearch_index import index_v2_enums
from models.pydantic.model_output_schema.seedtext_script_segments_schema import (
    SeedtextStoryboardEnvelope,
    SeedtextIndexTagsEnvelope,
)
from utils.call_model_utils import call_doubao_seedtext


def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])


MODEL_CANDIDATES = [
    "Seed 2.0 Pro",
    "Seed 2.0 Lite",
    "Seed 2.0 Mini",
]


SYSTEM_PROMPT_STAGE1 = f"""你是一个“汽车短视频口播脚本 → 分镜规划(StoryBoard)”的结构化生成器。

目标：把一段口播脚本，拆成多个“3秒口播分段”，并为每个分段给出可拍的镜头规划。
注意：此阶段不要强行把所有字段拆成严格标签；更像“分镜脚本”，但要结构稳定。

硬性规则（非常重要）：
1) 只输出严格 JSON（不要 Markdown，不要解释，不要多余文本）。
2) 分段规则：
   - 素材平均时长只有 3 秒：请把长句再切碎成“3秒口播短句”，符合广告分镜的分段节奏
   - 优先按中文标点切分：逗号/顿号/分号/冒号/感叹号/句号；必要时可额外断句
   - 【非常重要：一句话内也要拆】同一句话只要出现“逗号/顿号/分号”等，且各分句表达的是不同信息点（功能参数 vs 使用场景 vs 情绪代入），必须拆成多个段
   - 每段只聚焦 1 个核心卖点或 1 个具体场景
3) 你必须严格按 schema 输出（由调用方提供 json_schema）
4) 禁止输出大段原文；列表都要“短、可检索、去重、无空字符串”。

枚举可选值（必须从中选）：
- shot_style: {_join_choices(index_v2_enums.SHOT_STYLE_CHOICES)}
- shot_type: {_join_choices(index_v2_enums.SHOT_TYPE_CHOICES)}
- video_usage: {_join_choices(index_v2_enums.VIDEO_USAGE_CHOICES)}
""".strip()


SYSTEM_PROMPT_STAGE2 = f"""你是一个“分镜规划(StoryBoard) → 严格检索标签(IndexV2)”的结构化信息抽取器。

目标：把 Stage1 的 storyboard 中每个分镜，转换为可入库/可检索的严格标签字段。
注意：本阶段输出将用于 OpenSearch 索引（字段要短、稳定、可命中）。

硬性规则（非常重要）：
1) 只输出严格 JSON（不要 Markdown，不要解释，不要多余文本）。
2) 你必须严格按 schema 输出（由调用方提供 json_schema）
3) 枚举/keyword 字段必须从 choices 中选择；不确定就用“未知”或 null（按 schema）。
4) 同一概念不要堆叠同义词；短词化；去重；不要输出空字符串。

规范化与纠错（必须遵守）：
A) shot_type vs shot_style 不可混用：
   - 景别词（{_join_choices(index_v2_enums.SHOT_TYPE_CHOICES)}）只能写入 shot_type
   - shot_style 只能从（{_join_choices(index_v2_enums.SHOT_STYLE_CHOICES)}）选择
B) weather vs time 不可混用：
   - time 只能从：{_join_choices(index_v2_enums.TIME_CHOICES)}
   - weather 只能从：{_join_choices(index_v2_enums.WEATHER_CHOICES)}
C) car_color 归一化：
   - car_color 必须严格从：{_join_choices(index_v2_enums.CAR_COLOR_CHOICES)}
   - 禁止输出“黑色/白色/蓝色/银色/绿色”等变体；细节写入 car_color_detail
D) video_usage 归一化：
   - video_usage(list) 必须从：{_join_choices(index_v2_enums.VIDEO_USAGE_CHOICES)}
   - 同义归并：品牌传达/品牌形象传达 -> 品牌/形象传达；权益说明 -> 权益/价格说明；路跑场景展示 -> 使用场景展示
E) product_status_scene 不允许带括号备注：
   - product_status_scene 必须从：{_join_choices(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}
""".strip()


async def _call_seedtext_with_fallback(
    *,
    prompt: str,
    system_prompt: str,
    output_schema: Any,
) -> str:
    last_err: Optional[Exception] = None
    for m in MODEL_CANDIDATES:
        try:
            out = await call_doubao_seedtext(
                model=m,
                system_prompt=system_prompt,
                prompt=prompt,
                output_schema=output_schema,
            )
            if out:
                return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"SeedText call failed for all candidates. last_err={last_err}")


def _extract_json_text(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```json"):
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    return t


async def rewrite_script_to_storyboard_and_tags(
    script: str,
    *,
    index: int = 0,
) -> Tuple[SeedtextStoryboardEnvelope, SeedtextIndexTagsEnvelope]:
    """
    Two-stage rewrite:
    Stage1: script -> storyboard
    Stage2: storyboard -> index tags (strict)
    """
    script = (script or "").strip()
    if not script:
        raise ValueError("script is empty")

    stage1_raw = await _call_seedtext_with_fallback(
        prompt=script,
        system_prompt=SYSTEM_PROMPT_STAGE1,
        output_schema=SeedtextStoryboardEnvelope,
    )
    storyboard = SeedtextStoryboardEnvelope.model_validate(json.loads(_extract_json_text(stage1_raw)))

    # ensure index field is consistent
    for seg in storyboard.storyboard:
        seg.index = index

    stage2_prompt = (
        "下面是 Stage1 生成的 storyboard JSON，请基于它输出 Stage2 的严格标签。\n\n"
        + json.dumps(storyboard.model_dump(), ensure_ascii=False)
    )
    stage2_raw = await _call_seedtext_with_fallback(
        prompt=stage2_prompt,
        system_prompt=SYSTEM_PROMPT_STAGE2,
        output_schema=SeedtextIndexTagsEnvelope,
    )
    tags = SeedtextIndexTagsEnvelope.model_validate(json.loads(_extract_json_text(stage2_raw)))
    for seg in tags.segment_result:
        seg.index = index

    return storyboard, tags

