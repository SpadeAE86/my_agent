# -*- coding: utf-8 -*-
"""
seed_car_interior_docs.py

生成 5 条模拟的 ShotCard 分析结果并入库到 OpenSearch 的 car_interior_analysis 索引。

运行:
  python -m src.test.seed_car_interior_docs
或在 my_agent/src 下:
  python -m test.seed_car_interior_docs
"""

import os
import sys
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.pydantic.video_analysis_request import ShotCard
from services.analysis_video import index_shotcards_to_opensearch


def mock_cards() -> list[ShotCard]:
    base = {
        "description": "视频片段展示汽车内部场景，以白色皮质座椅为核心，光线明亮柔和，整体设计简约豪华。",
        "subject": "汽车座椅（汽车内饰）",
        "object": ["座椅", "天窗", "杯架", "车门", "中控台"],
        "movement": "副驾驶座椅靠背调节（从直立向倾斜调整）",
        "adjective": ["豪华", "舒适", "明亮", "简约", "高档", "整洁", "现代", "精致"],
        "search_tags": ["汽车内饰", "白色皮质座椅", "座椅调节", "豪华汽车", "车载天窗", "舒适驾乘", "汽车内部设计", "中高端汽车"],
        "marketing_tags": ["产品展示", "使用场景"],
        "appealing_audience": ["汽车爱好者", "购车人群", "中高端消费者", "追求舒适出行者", "有车族"],
        "visual_quality": [8, 7, 8, 7],
        "thumbnail": "https://via.placeholder.com/640x360?text=Shot",
        "frame_urls": ["https://via.placeholder.com/640x360?text=Frame1"],
    }
    cards = []
    for i in range(1, 6):
        cards.append(
            ShotCard(
                scene_id=i,
                start_time=(i - 1) * 5,
                end_time=i * 5,
                duration_seconds=5,
                **base,
            )
        )
    return cards


async def main():
    cards = mock_cards()
    resp = await index_shotcards_to_opensearch(cards, id_prefix="seed_demo", refresh=True)
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())

