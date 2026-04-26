# -*- coding: utf-8 -*-
"""
check_car_interior_index_v2.py

Quick sanity checks for OpenSearch index `car_interior_analysis_v2`:
- count documents
- fetch one document id

Run:
  python -m src.test.check_car_interior_index_v2
"""

import os
import sys
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import get_index_name


async def main():
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    idx = get_index_name(CarInteriorAnalysisV2)

    cnt = await c.count(index=idx, body={"query": {"match_all": {}}})
    print("index:", idx)
    print("count:", cnt)

    one = await c.search(index=idx, body={"size": 1, "query": {"match_all": {}}})
    hits = ((one.get("hits") or {}).get("hits") or [])
    if hits:
        print("first_id:", hits[0].get("_id"))
    else:
        print("first_id: <none>")

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

