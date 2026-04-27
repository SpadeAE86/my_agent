"""
seedtext_rewrite_scripts_to_labels.py

Use Doubao Seed text model to rewrite long ad scripts into flat label lists
that can be indexed / matched against tag-based retrieval.

Run:
  python -m src.test.seedtext_rewrite_scripts_to_labels
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Any, Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils.call_model_utils import call_doubao_seedtext  # noqa: E402
from models.pydantic.opensearch_index import index_v2_enums  # noqa: E402
from models.pydantic.model_output_schema.seedtext_script_segments_schema import (  # noqa: E402
    SeedtextStoryboardEnvelope,
    SeedtextIndexTagsEnvelope,
)

# Best-effort: make Windows console output UTF-8 to avoid mojibake.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


INDEX_FIELD_HINTS = """
你要输出的字段将用于 OpenSearch 索引 car_interior_analysis_v2 的以下字段（请尽量让值可命中）：
- frame_size / resolution / footage_type / shot_style
- scene_location[] / car_color / car_color_detail
- product_status_scene / product_status_scene_text
- has_presenter / weather / time / video_usage
- description / movement / subject / object[]
- design_selling_points[] / function_selling_points[]
- design_adjectives[] / function_adjectives[]
- scenario_a[] / scenario_b[]
- marketing_phrases[] / appealing_audience[]
"""

STAGE1_SCHEMA_JSON = SeedtextStoryboardEnvelope.model_json_schema()
STAGE2_SCHEMA_JSON = SeedtextIndexTagsEnvelope.model_json_schema()


def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])

# Choose an available model in your Ark account.
# The script will try them in order until one works.
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
   - 素材平均时长只有 3 秒：请把长句再切碎成“3秒口播短句”，符合广告分镜的分段节奏；每段口播最好控制在 6~60字/词，过长会导致匹配到的素材候选不足。
   - 优先按中文标点切分：逗号/顿号/分号/冒号/感叹号/句号；必要时可额外断句
   - 【非常重要：一句话内也要拆】同一句话只要出现“逗号/顿号/分号”等，且各分句表达的是不同信息点（功能参数 vs 使用场景 vs 情绪代入），必须拆成多个段；不要把多个信息点塞进一个 segment。
   - 【片段的功能,场景,景幅出现变化就拆】如果用于搭配同一句话的所需片段里同时包含“功能讲解”和“场景演示/代入”，必须拆开，并给不同的 video_usage / movement（如需要）。
   - 每段只聚焦 1 个核心卖点或 1 个具体场景，不要把所有卖点塞进一段
   - segment_text 要像“口播分段”一样自然，但不要长篇复读原文
3) 你必须严格按 schema 输出：{STAGE1_SCHEMA_JSON}
4) 禁止输出大段原文；列表都要“短、可检索、去重、无空字符串”。
6) 字段约束：
   - 每条分镜都必须包含 index（脚本序号）与 id（分镜序号）
   - index 用来标记这条分镜属于哪一个输入脚本；如果一次只给一个脚本，就统一填 0

枚举可选值（必须从中选）：
- shot_style: {index_v2_enums.SHOT_STYLE_CHOICES}
- shot_type: {index_v2_enums.SHOT_TYPE_CHOICES}
- video_usage: {index_v2_enums.VIDEO_USAGE_CHOICES}


拆分示例（只示意分段，不示意全部字段）：
输入句：
“转弯半径4.79米，狭窄地库一把掉头，新手也不慌。”

应正确拆成 2 条分镜：
1) “转弯半径4.79米。”
2) “狭窄地库一把掉头，新手也不慌。”

"""


SYSTEM_PROMPT_STAGE2 = f"""你是一个“分镜规划(StoryBoard) → 严格检索标签(IndexV2)”的结构化信息抽取器。

目标：把 Stage1 的 storyboard 中每个分镜，转换为可入库/可检索的严格标签字段。
注意：本阶段输出将用于 OpenSearch 索引（字段要短、稳定、可命中）。

硬性规则（非常重要）：
1) 只输出严格 JSON（不要 Markdown，不要解释，不要多余文本）。
2) 你必须严格按 schema 输出：{STAGE2_SCHEMA_JSON}
3) 枚举/keyword 字段必须从 choices 中选择；不确定就用“未知”或 null（按 schema）。
4) 同一概念不要堆叠同义词；短词化；去重；不要输出空字符串。

规范化与纠错（必须遵守）：
A) shot_type vs shot_style 不可混用：
   - 如果你要输出的值属于景别（例如：{_join_choices(index_v2_enums.SHOT_TYPE_CHOICES)}），只能写入 shot_type。
   - shot_style 必须输出拍摄方式（例如：{_join_choices(index_v2_enums.SHOT_STYLE_CHOICES)}），禁止输出“特写/中景/远景”等景别词。
B) weather vs time 不可混用：
   - time 只能从：{_join_choices(index_v2_enums.TIME_CHOICES)}
   - weather 只能从：{_join_choices(index_v2_enums.WEATHER_CHOICES)}
   - 禁止把“白天/夜晚/黄昏/室内”写进 weather；禁止把“雨天/雪天/阴天/晴天/极寒”等写进 time。
C) car_color 归一化（禁止输出同义变体）：
   - car_color 必须严格从：{_join_choices(index_v2_enums.CAR_COLOR_CHOICES)}
   - 禁止输出“黑色/白色/蓝色/银色/绿色”等带“色”或不在枚举里的值；颜色细节写入 car_color_detail。
D) video_usage 归一化（只允许标准枚举）：
   - video_usage(list) 必须从：{_join_choices(index_v2_enums.VIDEO_USAGE_CHOICES)}
   - 同义归并：品牌传达/品牌形象传达 -> 品牌/形象传达；权益说明 -> 权益/价格说明；路跑场景展示 -> 使用场景展示。
E) product_status_scene 不允许带括号备注：
   - product_status_scene 必须从：{_join_choices(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}
   - 像“含动态灯语/充电状态/节日装饰”等细节，请写入 product_status_scene_text（若 schema 没有该字段则放入 extra_tags）。

枚举可选值（必须从中选）：
- movement: {index_v2_enums.MOVEMENT_CHOICES}
- shot_style: {index_v2_enums.SHOT_STYLE_CHOICES}
- shot_type: {index_v2_enums.SHOT_TYPE_CHOICES}
- video_usage(list): {index_v2_enums.VIDEO_USAGE_CHOICES}
- weather: {index_v2_enums.WEATHER_CHOICES}
- time: {index_v2_enums.TIME_CHOICES}
- car_color: {index_v2_enums.CAR_COLOR_CHOICES}
- product_status_scene: {index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES}
"""




async def main(prompt: str ):
    print("🚀 Stage1: 调用豆包 Seedtext 生成分镜规划 (storyboard)...")

    try:
        stage1_text = None
        last_err: Exception | None = None
        for m in MODEL_CANDIDATES:
            try:
                stage1_text = await call_doubao_seedtext(
                    model=m,
                    system_prompt=SYSTEM_PROMPT_STAGE1,
                    prompt=prompt,
                    output_schema=SeedtextStoryboardEnvelope,
                )
                if stage1_text:
                    break
            except Exception as e:
                last_err = e
                continue

        if not stage1_text:
            raise RuntimeError(f"Stage1 failed for all MODEL_CANDIDATES. last_err={last_err}")

        clean_json = stage1_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()

        raw_data = json.loads(clean_json)

        storyboard = SeedtextStoryboardEnvelope.model_validate(raw_data)

        print("✅ Stage1 完成，分镜条数:", len(storyboard.storyboard))

        # Stage2
        print("🚀 Stage2: 调用豆包 Seedtext 抽取严格标签 (IndexV2)...")
        stage2_prompt = (
            "下面是 Stage1 生成的 storyboard JSON，请基于它输出 Stage2 的严格标签。\n\n"
            + json.dumps(storyboard.model_dump(), ensure_ascii=False)
        )
        stage2_text = None
        last_err2: Exception | None = None
        for m in MODEL_CANDIDATES:
            try:
                stage2_text = await call_doubao_seedtext(
                    model=m,
                    system_prompt=SYSTEM_PROMPT_STAGE2,
                    prompt=stage2_prompt,
                    output_schema=SeedtextIndexTagsEnvelope,
                )
                if stage2_text:
                    break
            except Exception as e:
                last_err2 = e
                continue

        if not stage2_text:
            raise RuntimeError(f"Stage2 failed for all MODEL_CANDIDATES. last_err={last_err2}")

        clean_json2 = stage2_text.strip()
        if clean_json2.startswith("```json"):
            clean_json2 = clean_json2.split("```json")[1].split("```")[0].strip()
        raw_data2 = json.loads(clean_json2)
        result_envelope = SeedtextIndexTagsEnvelope.model_validate(raw_data2)

        # 5. 打印美化后的结果
        print("\n✅ 脚本拆解完成！结果如下：\n")
        print(f"{'ID':<4} | {'Index':<6} | {'Usage':<15} | {'Shot':<10} | {'Text'}")
        print("-" * 80)

        for item in result_envelope.segment_result:
            usage_str = ",".join(item.video_usage or [])
            print(f"{item.id:<4} | {item.index:<6} | {usage_str:<15} | {item.shot_type:<10} | {item.segment_text}")

            # 如果你想看更详细的卖点标签
            # print(f"   └─ 卖点: {item.function_selling_points}")
            # print(f"   └─ 营销: {item.marketing_phrases}")

        # 6. 可选：保存到本地
        with open("script_storyboard_output.json", "w", encoding="utf-8") as f:
            json.dump(storyboard.model_dump(), f, ensure_ascii=False, indent=2)
            print(f"\n💾 Stage1 已保存至: {os.path.abspath('script_storyboard_output.json')}")

        with open("script_tags_output.json", "w", encoding="utf-8") as f:
            json.dump(result_envelope.model_dump(), f, ensure_ascii=False, indent=2)
            print(f"\n💾 详细数据已保存至: {os.path.abspath('script_tags_output.json')}")

    except json.JSONDecodeError:
        print("❌ 模型返回的不是有效的 JSON 格式：")
        print("Stage1/Stage2 raw text may contain invalid JSON.")
    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    # 使用 asyncio 运行
    input_prompt: str = "这台智己LS6，搭载65.9kWh行业超大增程电池，纯电能跑450公里，综合续航超过1500公里！我通勤一周充一次电就够了！而且它转弯半径只有4.79米，比紧凑级轿车还小，我这种新手女司机在狭窄地库掉头，一把就过了！"

    asyncio.run(main(input_prompt))
