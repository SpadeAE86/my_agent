# -*- coding: utf-8 -*-
"""
build_car_interior_index_v2.py

Create/overwrite OpenSearch index: `car_interior_analysis_v2`.

Run:
  python -m src.test.build_car_interior_index_v2
"""

import os
import sys
import asyncio

# Ensure we can import from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch.create_index import index_manager
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from infra.storage.opensearch_connector import opensearch_connector


async def main(overwrite: bool = True):
    # If your OpenSearch cluster requires index.knn to be enabled, keep this.
    # Otherwise you can comment out `settings=` and rely on defaults.
    settings = {
        "index": {"knn": True},
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }

    print("开始创建索引 car_interior_analysis_v2 ...")
    ok = await index_manager.create_index(
        model_class=CarInteriorAnalysisV2,
        field_types=None,  # derive from Annotated markers
        settings=settings,
        overwrite=overwrite,
    )
    print(f"索引创建完成: {ok}")
    # Ensure aiohttp session/connector is closed (avoid warnings on exit).
    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

