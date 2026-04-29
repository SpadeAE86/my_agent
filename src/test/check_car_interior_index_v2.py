# -*- coding: utf-8 -*-
"""
check_car_interior_index_v2.py

Quick sanity checks for OpenSearch index `car_interior_analysis_v2`:
- count documents
- fetch one document id
- report missing vector fields (likely caused by empty-text bug)

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


VECTOR_FIELDS = [
    "description_vector",
    "function_selling_points_vector",
    "design_selling_points_vector",
    "scenario_a_vector",
    "scenario_b_vector",
    "marketing_phrases_vector",
    "adjectives_vector",
]


async def main():
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    idx = get_index_name(CarInteriorAnalysisV2)

    cnt = await c.count(index=idx, body={"query": {"match_all": {}}})
    print("index:", idx)
    print("count:", cnt.get("count"))

    one = await c.search(index=idx, body={"size": 1, "query": {"match_all": {}}})
    hits = ((one.get("hits") or {}).get("hits") or [])
    if hits:
        print("first_id:", hits[0].get("_id"))
    else:
        print("first_id: <none>")

    # Missing vectors: count docs where each vector field is absent.
    for vf in VECTOR_FIELDS:
        missing = await c.count(
            index=idx,
            body={"query": {"bool": {"must_not": [{"exists": {"field": vf}}]}}},
        )
        print(f"missing.{vf}:", missing.get("count"))

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

