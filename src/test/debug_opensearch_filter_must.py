"""
debug_opensearch_filter_must.py

Minimal repro for: why a doc is returned even when filters should exclude it.

It runs an OpenSearch search with:
- ids filter including only DOC_ID
- term filters on topic/product_status_scene (expected to exclude the doc)

If the doc still returns, either:
- the doc's stored field values actually match filters, OR
- the cluster/query parsing ignores those filters in this query shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402


INDEX = "car_interior_analysis_v2"
DOC_ID = "v2_17a6bc414c8a_scene_001"

# Filters that should exclude DOC_ID according to your inspection
FILTERS = [
    {"term": {"topic": {"value": "舒适"}}},
    {"term": {"product_status_scene": {"value": "静态内饰"}}},
]


async def main() -> None:
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    body = {
        "size": 5,
        "query": {
            "bool": {
                "filter": [{"ids": {"values": [DOC_ID]}}] + FILTERS,
                "must": [{"match_all": {}}],
            }
        },
        "_source": {"includes": ["topic", "product_status_scene", "movement", "video_usage"]},
    }

    resp = await c.search(index=INDEX, body=body)
    hits = (((resp or {}).get("hits") or {}).get("hits") or [])
    print("query:", json.dumps(body, ensure_ascii=False, indent=2))
    print("hits_count:", len(hits))
    for h in hits:
        print("_id:", h.get("_id"), "_source:", h.get("_source"))

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

