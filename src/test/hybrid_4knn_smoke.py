# -*- coding: utf-8 -*-
"""
hybrid_4knn_smoke.py

Smoke test: one OpenSearch ``hybrid`` request with
  1 × multi_match (BM25) + N × knn  (default N=4 → 5 sub-queries total).

Use this to see whether your local OpenSearch version / machine accepts the
sub-query count and normalization pipeline.

Default query text: 空调

Run (from repo, with env that has opensearch-py + sentence-transformers):

  cd my_agent/src
  python -m test.hybrid_4knn_smoke

Options:

  python -m test.hybrid_4knn_smoke --query "冰箱" --knn 4
  python -m test.hybrid_4knn_smoke --no-pipeline   # may error on some OS builds

Connection uses ``config.yml`` → opensearch section (same as the app).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from infra.storage.opensearch.query_builder import QueryBuilder  # noqa: E402
from models.pydantic.opensearch_index.base_index import (  # noqa: E402
    get_index_name,
    get_vector_fields,
    get_vector_weights,
)
from models.pydantic.opensearch_index.car_interior_analysis_v2 import (  # noqa: E402
    CarInteriorAnalysisV2,
)
from services.script_match_recall import ensure_hybrid_pipeline  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IndexModel = CarInteriorAnalysisV2
BASE_PIPELINE = "nlp-search-pipeline"
DEFAULT_QUERY = "空调"
DEFAULT_SIZE = 10
DEFAULT_ROUTE_K = 120


def _pick_top_knn_fields(knn_count: int) -> List[str]:
    all_v = get_vector_fields(IndexModel)
    wmap = get_vector_weights(IndexModel)
    ranked = sorted(all_v, key=lambda f: wmap.get(f, 1.0), reverse=True)
    return ranked[: max(0, knn_count)]


def _count_hybrid_subqueries(body: Dict[str, Any]) -> int:
    q = body.get("query")
    if not isinstance(q, dict):
        return 0
    if "bool" in q:
        must = (q.get("bool") or {}).get("must") or []
        if must and isinstance(must[0], dict) and "hybrid" in must[0]:
            sub = (must[0]["hybrid"] or {}).get("queries") or []
            return len(sub)
    if "hybrid" in q:
        sub = (q["hybrid"] or {}).get("queries") or []
        return len(sub)
    return 0


async def _run() -> int:
    ap = argparse.ArgumentParser(description="Hybrid 1×BM25 + N×KNN smoke test")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="search text (embedded as one string)")
    ap.add_argument("--knn", type=int, default=4, help="number of knn sub-queries (default 4)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="final hit size")
    ap.add_argument("--route-k", type=int, default=DEFAULT_ROUTE_K, help="k inside each knn clause")
    ap.add_argument("--no-pipeline", action="store_true", help="omit search_pipeline (often breaks hybrid)")
    ap.add_argument("--list-vectors", action="store_true", help="print vector fields and exit")
    args = ap.parse_args()

    try:
        idx = get_index_name(IndexModel)
        all_vectors = get_vector_fields(IndexModel)

        if args.list_vectors:
            print(json.dumps({"index": idx, "vector_fields": all_vectors}, ensure_ascii=False, indent=2))
            return 0

        picked = _pick_top_knn_fields(args.knn)
        if not picked:
            print("ERROR: no vector fields on model", file=sys.stderr)
            return 2

        query_text = (args.query or "").strip() or DEFAULT_QUERY

        qb = QueryBuilder()
        # build_dynamic_hybrid_search uses body["size"] for knn "k"; override after build
        body = qb.build_dynamic_hybrid_search(
            IndexModel,
            query_text,
            size=int(args.route_k),
            bm25_factor=0.3,
            vector_factor=0.7,
            vector_fields=picked,
        )
        body["size"] = int(args.size)
        n_sub = _count_hybrid_subqueries(body)

        print("=== hybrid_4knn_smoke ===")
        print(f"index:          {idx}")
        print(f"query:          {query_text!r}")
        print(f"knn fields:     {picked}")
        print(f"sub-queries:    {n_sub} (expect {1 + len(picked)})")
        print(f"size / knn k:   {args.size} / {args.route_k}")

        await opensearch_connector.ensure_init()
        client = await opensearch_connector.get_client()

        pipeline: str | None = None
        if not args.no_pipeline:
            n_expected = 1 + len(picked)
            pipeline = await ensure_hybrid_pipeline(
                client, pipeline_name=BASE_PIPELINE, num_queries=n_expected
            )
            print(f"search_pipeline: {pipeline!r} (base {BASE_PIPELINE!r})")
        else:
            print("search_pipeline: (none)")

        params = {"search_pipeline": pipeline} if pipeline else None
        try:
            resp = await client.search(index=idx, body=body, params=params)
        except Exception as e:
            print("\nSEARCH FAILED:")
            print(repr(e))
            print(
                "\nTips: OpenSearch hybrid limits vary; try smaller --knn. "
                "Ensure pipeline exists unless your version does not need it.",
            )
            return 1

        hits = (resp.get("hits") or {}).get("hits") or []
        total = (resp.get("hits") or {}).get("total")
        print(f"\nOK: hits.len={len(hits)} total={total}")
        for i, h in enumerate(hits[:5]):
            _id = h.get("_id")
            sc = h.get("_score")
            mq = h.get("matched_queries")
            print(f"  [{i}] {_id}  score={sc}  matched_queries={mq}")

        return 0
    finally:
        await opensearch_connector.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
