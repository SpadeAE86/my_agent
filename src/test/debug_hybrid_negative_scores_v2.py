# -*- coding: utf-8 -*-
"""
debug_hybrid_negative_scores_v2.py

Purpose:
- Reproduce hybrid query raw hits + scores for car_interior_analysis_v2
- Compare with BM25-only and KNN-only routes
- Optionally enable OpenSearch profile/explain to locate weird scoring (e.g. huge negative scores)

Run:
  python -m src.test.debug_hybrid_negative_scores_v2
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Any, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import QueryBuilder
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import get_index_name, get_vector_fields


QUERY = "地库 一键泊车"
SIZE = 10
PROFILE = True
EXPLAIN = False

# Keep hybrid small to avoid subquery limit.
HYBRID_VECTOR_FIELDS = [
    "marketing_phrases_vector",
    "function_selling_points_vector",
    "description_vector",
]


def _min_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    return {"_id": h.get("_id"), "_score": h.get("_score")}


async def main():
    idx = get_index_name(CarInteriorAnalysisV2)
    qb = QueryBuilder()

    bm25 = qb.build_bm25_only_search(CarInteriorAnalysisV2, QUERY, size=SIZE)
    # pick one vector field for sanity
    vfs = get_vector_fields(CarInteriorAnalysisV2)
    one_vf = "marketing_phrases_vector" if "marketing_phrases_vector" in vfs else vfs[0]
    knn = qb.build_knn_only_search(CarInteriorAnalysisV2, QUERY, size=SIZE, vector_field=one_vf)

    hybrid = qb.build_dynamic_hybrid_search(
        CarInteriorAnalysisV2,
        QUERY,
        size=SIZE,
        vector_fields=HYBRID_VECTOR_FIELDS,
    )

    if PROFILE:
        bm25["profile"] = True
        knn["profile"] = True
        hybrid["profile"] = True
    if EXPLAIN:
        bm25["explain"] = True
        knn["explain"] = True
        hybrid["explain"] = True

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    rb = await c.search(index=idx, body=bm25)
    rk = await c.search(index=idx, body=knn)
    rh = await c.search(index=idx, body=hybrid)

    out = {
        "index": idx,
        "query": QUERY,
        "bm25_top": [_min_hit(h) for h in ((rb.get("hits") or {}).get("hits") or [])],
        "knn_field": one_vf,
        "knn_top": [_min_hit(h) for h in ((rk.get("hits") or {}).get("hits") or [])],
        "hybrid_vector_fields": HYBRID_VECTOR_FIELDS,
        "hybrid_top": [_min_hit(h) for h in ((rh.get("hits") or {}).get("hits") or [])],
    }

    await opensearch_connector.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

