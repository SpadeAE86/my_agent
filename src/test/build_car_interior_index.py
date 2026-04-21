# -*- coding: utf-8 -*-
"""
build_car_interior_index.py

在 OpenSearch 中创建/覆盖 `car_interior_analysis` 索引。
复用:
- models.pydantic.opensearch_index.car_interior_analysis.CarInteriorAnalysis
- infra.storage.opensearch.create_index.index_manager

运行方式(示例):
  python -m src.test.build_car_interior_index
或在 my_agent/src 目录下:
  python -m test.build_car_interior_index
"""

import os
import sys
import asyncio

# 确保能 import 到 src 下模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch.create_index import index_manager
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis


async def main(overwrite: bool = True):
    # 1) 索引 settings：开启 knn
    settings = {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    }

    # 2) 字段 mapping
    field_types = {
        "id": {"type": "keyword"},
        "description": {"type": "text", "analyzer": "standard"},
        "subject": {"type": "text", "analyzer": "standard"},
        "object": {"type": "keyword"},
        "movement": {"type": "text", "analyzer": "standard"},
        "adjective": {"type": "keyword"},
        "search_tags": {"type": "keyword"},
        "marketing_tags": {"type": "keyword"},
        "appealing_audience": {"type": "keyword"},
        "visual_quality": {"type": "float"},
        # 下面 3 个向量字段维度为 384（见 CarInteriorAnalysis._generate_embedding）
        "description_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
        },
        "subject_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
        },
        "combined_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
        },
    }

    print("开始创建索引 car_interior_analysis ...")
    ok = await index_manager.create_index(
        model_class=CarInteriorAnalysis,
        field_types=field_types,
        settings=settings,
        overwrite=overwrite,
    )
    print(f"索引创建完成: {ok}")


if __name__ == "__main__":
    asyncio.run(main())

