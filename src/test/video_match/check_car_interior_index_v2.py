# -*- coding: utf-8 -*-
"""
check_car_interior_index_v2.py

Quick sanity checks for OpenSearch index `car_interior_analysis_v2`:
- count documents
- fetch one document id
- report missing vector fields (likely caused by empty-text bug)

Run:
  python -m src.test.check_car_interior_index_v2
  python -m src.test.check_car_interior_index_v2 --samples 5
"""

import argparse
import os
import sys
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import get_index_name, get_vector_fields


VECTOR_FIELDS = [
    "description_vector",
    "function_selling_points_vector",
    "design_selling_points_vector",
    "scenario_vector",
    "marketing_phrases_vector",
    "design_adjectives_vector",
    "function_adjectives_vector",
]


def _len_or_none(v):
    if v is None:
        return None
    if isinstance(v, list):
        return len(v)
    return f"<{type(v).__name__}>"


async def main():
    ap = argparse.ArgumentParser(description="Sanity check car_interior_analysis_v2 index.")
    ap.add_argument("--samples", type=int, default=0, help="打印若干条文档的向量字段长度摘要")
    args = ap.parse_args()

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

    n = max(0, int(args.samples))
    if n > 0:
        m = await c.indices.get_mapping(index=idx)
        root = m.get(idx) or next(iter(m.values()), {})
        props = (root.get("mappings") or {}).get("properties") or {}
        print("\n--- mapping: knn / *_vector fields (top-level) ---")
        for name, spec in sorted(props.items()):
            if not isinstance(spec, dict):
                continue
            t = spec.get("type")
            if t == "knn_vector" or name.endswith("_vector"):
                dim = spec.get("dimension")
                extra = f" dim={dim}" if dim is not None else ""
                print(f"  {name}: {t}{extra}")

        vf_names = get_vector_fields(CarInteriorAnalysisV2)
        sr = await c.search(index=idx, body={"size": n, "query": {"match_all": {}}, "_source": True})
        th = ((sr.get("hits") or {}).get("hits") or [])
        print(f"\n--- sample docs (n={len(th)}), vector lengths (expect 384 or null) ---")
        for h in th:
            sid = h.get("_id")
            src = h.get("_source") or {}
            print(f"  _id={sid}")
            for f in vf_names:
                print(f"    {f}: {_len_or_none(src.get(f))}")

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

