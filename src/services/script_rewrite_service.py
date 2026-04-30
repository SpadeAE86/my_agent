from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Tuple

from models.pydantic.opensearch_index import index_v2_enums
from models.pydantic.model_output_schema.seedtext_script_segments_schema import (
    SeedtextStoryboardEnvelope,
    SeedtextIndexTagsEnvelope,
)
from utils.call_model_utils import call_doubao_seedtext
from utils.alivoice_utils import AliTTS

try:
    from pymediainfo import MediaInfo
except Exception:  # pragma: no cover
    MediaInfo = None  # type: ignore


def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])


MODEL_CANDIDATES = ["Seed 2.0 Lite"]

# --- Optional: use AliTTS + pymediainfo to compute accurate durations ---
# Flip to False (or comment out the block in rewrite_script_to_storyboard_and_tags) to skip.
ENABLE_TTS_DURATION = True
TTS_VOICE = "zhimao"
TTS_SPEED = 0
TTS_VOLUME = 80
TTS_DURATION_PAD_SECONDS = 0.4
TTS_MAX_CONCURRENCY = 4


SYSTEM_PROMPT_STAGE1 = f"""你是一个“汽车短视频口播脚本 → 分镜规划(StoryBoard)”的结构化生成器。

目标：把一段口播脚本，改写成“可用于汽车宣传片混剪”的分镜规划（StoryBoard）。
要求：节奏要像真实混剪，而不是死跟广告词逐句复述。
注意：此阶段不要强行把所有字段拆成严格标签；更像“分镜脚本”，但要结构稳定。

硬性规则（非常重要）：
1) 只输出严格 JSON（不要 Markdown，不要解释，不要多余文本）。
2) 分段规则：
   - 【时长与节奏最重要】每条分镜都必须填写 duration（秒），用“广告口播常见语速：1 秒≈5 个汉字”估算。
     - 计算方法：duration ≈ 字数/5（允许小数，保留 1 位小数）。
     - 正常混剪的节奏分布建议：多数 2.0s，少量 3.0s，偶尔 1.0s（用于转场/情绪点/极致特写）。
   - 分段长度不要求固定 3 秒：要长短错落，像剪辑节奏一样自然。
   - 优先按中文标点切分：逗号/顿号/分号/冒号/感叹号/句号；必要时可额外断句
   - 【非常重要：一句话内也要拆】同一句话只要出现“逗号/顿号/分号”等，且各分句表达的是不同信息点（功能参数 vs 使用场景 vs 情绪代入），必须拆成多个段
   - 每段只聚焦 1 个核心卖点或 1 个具体场景
3) 你必须严格按 schema 输出（由调用方提供 json_schema）
4) 禁止输出大段原文；列表都要“短、可检索、去重、无空字符串”。

混剪经验（必须遵守，决定检索可用性）：
- 允许不严格按原文顺序：你可以把“路跑/氛围/空间感”的镜头穿插在硬卖点之间，让整条片更像宣传片。
- 路跑镜头可以连续 2~4 条（不同镜头语言）：例如 远景→航拍→车外跟拍→车内POV→特写。
- 镜头语言要有节奏点：每 6~10 条分镜里，至少出现 2 条远景/航拍（建立空间感），至少 2 条特写/大特写（材质/屏幕/UI/轮毂/灯语/按键）。
- “路跑”可以作为万能过渡：参数/功能讲解后，插一条路跑（风噪/静谧/稳定/速度感/安全感）提高混剪观感。
- 不要每段都“说功能”：适当加入 1 秒情绪点（如：安静/高级/推背感/稳/爽/安心）和画面点（如：轮毂/灯语/材质/空间/操作动作）。

关于“参数/文字上屏”的重要约束（必须遵守，避免检索偏航）：
- 口播里出现“数值/参数/术语”（如 4.79 米、得房率、转向、四轮转向）≠ 画面一定出现“屏幕文字/参数界面/字幕条”。
- 除非脚本明确说“屏幕显示/仪表盘显示/字幕写着/镜头给到参数/特写UI”，否则不要把分镜写成“车机屏幕展示参数/字幕展示参数”。
- 当口播是参数/术语时，优先用更具象、更容易命中的画面来承载：例如
  - 转向/四轮转向/转弯半径 -> movement=转弯/掉头；主体=车辆或方向盘；场景=地库/窄路/公路/城市路口
  - 空间/得房率 -> 后排坐人、腿部空间、后备箱装载、静态内饰
  - 续航/补能 -> 路跑/通勤、充电桩插枪、仪表盘“续航条”(仅在明确展示时)
  - 可以在extra_tags和objects里面出现屏幕尝试召回可能存在的片段，但不要让“屏幕/字幕/参数上屏”的镜头占主导，否则检索时会过度偏向“参数界面/字幕条”而忽略其他重要场景。
- “屏幕/字幕/参数上屏”的镜头只占少量点缀：默认不超过全部分镜的 20%（除非脚本明确大量讲UI/参数展示）。

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
   - 禁止输出“黑色/白色/蓝色/银色/绿色”等变体；如脚本强调“哑光/珠光/贴膜”等细节，请放入 extra_tags 或 marketing_phrases
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
            last_err = RuntimeError(f"SeedText returned empty output. model={m}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"SeedText call failed for all candidates. last_err={last_err}")


def _extract_json_text(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```json"):
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    return t


def _mediainfo_duration_seconds(path: str) -> Optional[float]:
    """
    Return duration in seconds from pymediainfo, if available.
    """
    if not path or not os.path.exists(path):
        return None
    if MediaInfo is None:
        return None
    try:
        mi = MediaInfo.parse(path)
        # Prefer "General" track duration (ms)
        for tr in (mi.tracks or []):
            if getattr(tr, "track_type", None) == "General":
                d = getattr(tr, "duration", None)
                if d is not None:
                    return float(d) / 1000.0
        # Fallback: any track duration
        for tr in (mi.tracks or []):
            d = getattr(tr, "duration", None)
            if d is not None:
                return float(d) / 1000.0
    except Exception:
        return None
    return None


async def _apply_tts_durations_inplace(storyboard: SeedtextStoryboardEnvelope) -> None:
    """
    Mutate storyboard.storyboard[*].duration using AliTTS audio length + padding.
    Any errors fall back to existing duration.
    """
    if not storyboard or not getattr(storyboard, "storyboard", None):
        return

    # Create a temp workspace folder for wavs.
    out_dir = Path(mkdtemp(prefix="tts_dur_"))

    sem = asyncio.Semaphore(int(TTS_MAX_CONCURRENCY))

    async def one(i: int) -> None:
        seg = storyboard.storyboard[i]
        text = (seg.segment_text or "").strip()
        if not text:
            return
        wav_path = str(out_dir / f"seg_{i:04d}.wav")
        try:
            async with sem:
                tts = AliTTS(
                    tid=f"tts_dur_{i}",
                    test_file=wav_path,
                    voice=TTS_VOICE,
                    speed=TTS_SPEED,
                    volume=TTS_VOLUME,
                )
                # Generate wav
                await tts.async_start(text)
            dur = _mediainfo_duration_seconds(wav_path)
            if dur is None:
                return
            dur2 = float(dur) + float(TTS_DURATION_PAD_SECONDS)
            # Keep a sane floor; avoid 0 duration.
            seg.duration = max(0.5, round(dur2, 2))
        except Exception:
            return

    await asyncio.gather(*[one(i) for i in range(len(storyboard.storyboard))])


async def rewrite_script_to_storyboard_and_tags(
    script: str,
    *,
    topic: str | None = None,
    title: str | None = None,
    car_model: str | None = None,
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

    # Provide additional context (topic/title/car_model) to guide a more coherent storyboard.
    # Keep it plain text to avoid confusing the model with nested JSON in the user prompt.
    ctx_lines: List[str] = []
    if isinstance(topic, str) and topic.strip():
        ctx_lines.append(f"【主题】{topic.strip()}")
    if isinstance(title, str) and title.strip():
        ctx_lines.append(f"【标题】{title.strip()}")
    if isinstance(car_model, str) and car_model.strip():
        ctx_lines.append(f"【车型】{car_model.strip()}")
    ctx_lines.append("【口播脚本】")
    ctx_lines.append(script)
    stage1_prompt = "\n".join(ctx_lines).strip()

    stage1_raw = await _call_seedtext_with_fallback(
        prompt=stage1_prompt,
        system_prompt=SYSTEM_PROMPT_STAGE1,
        output_schema=SeedtextStoryboardEnvelope,
    )
    storyboard = SeedtextStoryboardEnvelope.model_validate(json.loads(_extract_json_text(stage1_raw)))

    # ensure index field is consistent
    for seg in storyboard.storyboard:
        seg.index = index

    # --- Optional: TTS precise duration step (can be toggled off) ---
    # If you want to skip, set ENABLE_TTS_DURATION = False.
    if ENABLE_TTS_DURATION:
        await _apply_tts_durations_inplace(storyboard)

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

