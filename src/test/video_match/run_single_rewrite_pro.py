"""
Run a single 2-stage rewrite using *Pro* SeedText model only.

Goal:
- Validate whether the Pro model supports response_format.type=json_schema
- Verify Stage1 (storyboard) + Stage2 (index tags) end-to-end

Usage (from DIYProject root):
  python -m src.test.run_single_rewrite_pro
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.pydantic.model_output_schema.seedtext_script_segments_schema import (  # noqa: E402
    SeedtextIndexTagsEnvelope,
    SeedtextStoryboardEnvelope,
)
from models.pydantic.opensearch_index import index_v2_enums  # noqa: E402
from utils.call_model_utils import call_doubao_seedtext  # noqa: E402


def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])


MODEL = "Seed 2.0 Pro"

TOPIC = "油车车主的真香时刻"
TITLE = "开油车的都后悔了！这车比油车还省心"
CAR_MODEL = "新一代LS6"

SCRIPT = (
    "智己LS6超级增程，亏电油耗不到5.32升，还能加92号油，一年两箱油就够了，用车成本直接砍半！\n"
    "关键是它后排空间超大，近90%得房率，我一米八的大个子坐后排还能翘二郎腿，孩子都能在后排跑！\n"
)


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
- 口播里出现“数值/参数/术语”≠ 画面一定出现“屏幕文字/参数界面/字幕条”。
- 除非脚本明确说“屏幕显示/仪表盘显示/字幕写着/镜头给到参数/特写UI”，否则不要把分镜写成“车机屏幕展示参数/字幕展示参数”。
- 当口播是参数/术语时，优先用更具象、更容易命中的画面来承载（转弯/掉头/绕桩/窄路会车、空间展示、装载等）。
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

关于“文字上屏/屏幕录制/参数镜头”的抽取约束（必须遵守）：
- 只有当 Stage1 的该分镜 description/segment_text 明确出现“屏幕/仪表/字幕/文字上屏/参数界面/UI特写/倒车影像”等视觉线索时：
  - 才允许 footage_type 选择“屏幕录制”
  - 才允许在 text(list) 里写具体字符串（如 65.9kWh、1500公里、4.79米）
- 如果 Stage1 只是口播提到数值/参数，但没有明确“上屏”，则：
  - text 必须为空列表或 null
  - footage_type 不要填“屏幕录制”
  - 把“数值/参数”放进 function_selling_points 或 marketing_phrases（短词化）
  - 用 movement/scene_location/subject/object 体现具象画面（例如 转弯/掉头/地库/窄路）
""".strip()


def _extract_json_text(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```json"):
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    return t


async def main():
    if not os.getenv("ARK_API_KEY"):
        raise RuntimeError("Missing ARK_API_KEY in environment/.env")

    ctx = f"【主题】{TOPIC}\n【标题】{TITLE}\n【车型】{CAR_MODEL}\n【口播脚本】\n{SCRIPT}".strip()

    print("=== Stage1: storyboard (Pro + json_schema) ===")
    stage1_raw = await call_doubao_seedtext(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_STAGE1,
        prompt=ctx,
        output_schema=SeedtextStoryboardEnvelope,
    )
    if not stage1_raw:
        raise RuntimeError("Stage1 returned empty")
    storyboard = SeedtextStoryboardEnvelope.model_validate(json.loads(_extract_json_text(stage1_raw)))

    print("=== Stage2: tags (Pro + json_schema) ===")
    stage2_prompt = (
        "下面是 Stage1 生成的 storyboard JSON，请基于它输出 Stage2 的严格标签。\n\n"
        + json.dumps(storyboard.model_dump(), ensure_ascii=False)
    )
    stage2_raw = await call_doubao_seedtext(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_STAGE2,
        prompt=stage2_prompt,
        output_schema=SeedtextIndexTagsEnvelope,
    )
    if not stage2_raw:
        raise RuntimeError("Stage2 returned empty")
    tags = SeedtextIndexTagsEnvelope.model_validate(json.loads(_extract_json_text(stage2_raw)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "workspace" / f"single_rewrite_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "storyboard.json").write_text(
        json.dumps(storyboard.model_dump(exclude_none=True), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tags.json").write_text(
        json.dumps(tags.model_dump(exclude_none=True), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote:", str(out_dir))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

