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
    SeedtextSegmentResultEnvelope, SeedtextSegmentTagItem,
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

SUB_SCHEMA = SeedtextSegmentTagItem.model_json_schema()
SCHEMA_JSON = SeedtextSegmentResultEnvelope.model_json_schema()


SYSTEM_PROMPT = f"""你是一个“汽车短视频口播脚本 → 检索标签”的结构化信息抽取器。

目标：把一段口播脚本，拆成多个“口播分段”，并为每个分段生成可检索标签集合。
用途：每个口播分段将用于从素材库中检索匹配的视频片段（所以每段要尽量聚焦单一卖点/单一场景）。

硬性规则（非常重要）：
1) 只输出严格 JSON（不要 Markdown，不要解释，不要多余文本）。
2) 分段规则：
   - 素材平均时长只有 3 秒：请把长句再切碎成“3秒口播短句”，符合广告分镜的分段节奏；每段口播最好控制在 6~60字/词，过长会导致匹配到的素材候选不足。
   - 优先按中文标点切分：逗号/顿号/分号/冒号/感叹号/句号；必要时可额外断句
   - 【非常重要：一句话内也要拆】同一句话只要出现“逗号/顿号/分号”等，且各分句表达的是不同信息点（功能参数 vs 使用场景 vs 情绪代入），必须拆成多个段；不要把多个信息点塞进一个 segment。
   - 【片段的功能,场景,景幅出现变化就拆】如果用于搭配同一句话的所需片段里同时包含“功能讲解”和“场景演示/代入”，必须拆开，并给不同的 video_usage / movement（如需要）。
   - 每段只聚焦 1 个核心卖点或 1 个具体场景，不要把所有卖点塞进一段
   - segment_text 要像“口播分段”一样自然，但不要长篇复读原文
3) 每段的理解需要以{SUB_SCHEMA}里定义的字段为指导，尽量把每段的核心卖点/场景/动作等信息都体现在这些字段里（尤其是 video_usage, movement, shot_style, shot_type 这些对片段效果和检索很重要的字段）。
    每段的 extra_tags 字段可以带上没有合适字段但这个分镜应该有的一些特点。
4) 标签约束（很重要）：
   - tag_list 里的每个标签尽量是“短词/短语”，避免整句与比较句（例如“比XX更小”这种不要）
   - 同一概念不要重复同义词堆叠（长续航/超长续航/续航很长 → 选 1 个）
5) 禁止输出大段原文；所有列表都要“短、可检索、去重、无空字符串”。
6) 字段约束：
   - 每条分镜都必须包含 index（脚本序号）与 id（分镜序号）
   - index 用来标记这条分镜属于哪一个输入脚本；如果一次只给一个脚本，就统一填 0

枚举可选值（必须从中选）：
- movement: {index_v2_enums.MOVEMENT_CHOICES}
- shot_style: {index_v2_enums.SHOT_STYLE_CHOICES}
- shot_type: {index_v2_enums.SHOT_TYPE_CHOICES}
- video_usage(list): {index_v2_enums.VIDEO_USAGE_CHOICES}（尽量 1 个，必要才多加，最多3个）


拆分示例：
输入句：
“转弯半径4.79米，狭窄地库一把掉头，新手也不慌。”

由于包含功能讲解为目的片段和使用场景演示为目的的片段，应正确拆成 2 条分镜：
1) 
   segment_text: “转弯半径4.79米。”
   video_usage: ["功能讲解"]
   movement: 转弯

2)
   segment_text: “狭窄地库一把掉头，新手也不慌。”
   video_usage: ["使用场景展示"]
   movement: 掉头

"""




async def main(prompt: str ):
    # 1. 构造用户输入：将多个脚本带上序号传给 AI

    print("🚀 正在调用豆包 Seedtext 模型进行脚本拆解与打标...")

    try:
        # 2. 调用模型
        # 注意：call_doubao_seedtext 内部应处理 JSON 模式和 Schema 注入
        # 如果 call_doubao_seedtext 支持 response_format，建议传入 SCHEMA_JSON
        response_text = await call_doubao_seedtext(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            response_format="json_object" # 如果你的工具函数支持
        )
        print(response_text)
        # 3. 解析结果
        # 模型可能返回带有 ```json 块的内容，需要清洗
        clean_json = response_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()

        raw_data = json.loads(clean_json)

        # 4. 使用 Pydantic 校验结果，确保符合我们定义的 Schema
        result_envelope = SeedtextSegmentResultEnvelope.model_validate(raw_data)

        # 5. 打印美化后的结果
        print("\n✅ 脚本拆解完成！结果如下：\n")
        print(f"{'ID':<4} | {'Index':<6} | {'Usage':<15} | {'Shot':<10} | {'Text'}")
        print("-" * 80)

        for item in result_envelope.segment_result:
            usage_str = ",".join(item.video_usage)
            print(f"{item.id:<4} | {item.index:<6} | {usage_str:<15} | {item.shot_type:<10} | {item.segment_text}")

            # 如果你想看更详细的卖点标签
            # print(f"   └─ 卖点: {item.function_selling_points}")
            # print(f"   └─ 营销: {item.marketing_phrases}")

        # 6. 可选：保存到本地
        with open("script_tags_output.json", "w", encoding="utf-8") as f:
            json.dump(result_envelope.model_dump(), f, ensure_ascii=False, indent=2)
            print(f"\n💾 详细数据已保存至: {os.path.abspath('script_tags_output.json')}")

    except json.JSONDecodeError:
        print("❌ 模型返回的不是有效的 JSON 格式：")
        print(response_text)
    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    # 使用 asyncio 运行
    input_prompt: str = "这台智己LS6，搭载65.9kWh行业超大增程电池，纯电能跑450公里，综合续航超过1500公里！我通勤一周充一次电就够了！而且它转弯半径只有4.79米，比紧凑级轿车还小，我这种新手女司机在狭窄地库掉头，一把就过了！"

    asyncio.run(main(input_prompt))
