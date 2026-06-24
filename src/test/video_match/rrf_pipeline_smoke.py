# -*- coding: utf-8 -*-
"""
rrf_pipeline_smoke.py

Probe whether the local OpenSearch cluster accepts the **score-ranker-processor**
(RRF / reciprocal rank fusion) search pipeline introduced in OpenSearch 2.19+.

Flow:
  1) PUT ``/_search/pipeline/<name>`` with ``phase_results_processors`` → score-ranker + rrf
  2) RUN a small **hybrid** search (1×BM25 + 1×KNN by default) on ``car_interior_analysis_v2``
     using ``search_pipeline=<name>``
  3) Print OK + top hit ids/scores, or the raw error

This does **not** use normalization-processor; RRF is a different combination path.

Run::

  cd my_agent/src
  python -m test.rrf_pipeline_smoke

  python -m test.rrf_pipeline_smoke --knn-routes 2   # 1 BM25 + 2 KNN = 3 hybrid sub-queries
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IndexModel = CarInteriorAnalysisV2
DEFAULT_QUERY = "空调"
PIPELINE_NAME = "video-analysis-rrf-probe"


def _build_rrf_pipeline_body(
    *,
    num_queries: int,
    rank_constant: int,
    equal_weights: bool,
) -> Dict[str, Any]:
    comb: Dict[str, Any] = {
        "technique": "rrf",
        "rank_constant": int(rank_constant),
    }
    if equal_weights:
        if num_queries < 2:
            raise ValueError("Need at least 2 sub-queries for RRF probe.")
        w = round(1.0 / float(num_queries), 6)
        weights = [w] * num_queries
        weights[-1] = round(1.0 - sum(weights[:-1]), 6)
        comb["parameters"] = {"weights": weights}

    return {
        "description": "Probe: score-ranker RRF for hybrid sub-queries",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": comb,
                },
            },
        ],
    }


def _count_hybrid_subqueries(body: Dict[str, Any]) -> int:
    q = body.get("query")
    if not isinstance(q, dict):
        return 0
    if "hybrid" in q:
        sub = (q["hybrid"] or {}).get("queries") or []
        return len(sub)
    if "bool" in q:
        must = (q.get("bool") or {}).get("must") or []
        if must and isinstance(must[0], dict) and "hybrid" in must[0]:
            sub = (must[0]["hybrid"] or {}).get("queries") or []
            return len(sub)
    return 0


async def _put_pipeline(client: Any, name: str, body: Dict[str, Any]) -> tuple[bool, str]:
    try:
        resp = await client.http.put(f"/_search/pipeline/{name}", body=body)
        # opensearch-py returns aiohttp-like response or dict depending on version
        sc: Any = getattr(resp, "status", None)
        if sc is None:
            sc = getattr(resp, "status_code", None)
        if sc is not None and int(sc) not in (200, 201):
            try:
                txt = await resp.text() if hasattr(resp, "text") else str(resp)
            except Exception:
                txt = str(resp)
            return False, f"HTTP {sc}: {txt}"
        return True, "ok"
    except Exception as e:
        return False, repr(e)


async def _cluster_version(client: Any) -> Optional[str]:
    try:
        r = await client.info()
        if isinstance(r, dict):
            v = (r.get("version") or {}).get("number")
            return str(v) if v else None
    except Exception:
        pass
    return None


async def _run() -> int:
    ap = argparse.ArgumentParser(description="Probe OpenSearch RRF score-ranker pipeline")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--knn-routes", type=int, default=1, help="number of KNN sub-queries (BM25 always 1)")
    ap.add_argument("--rank-constant", type=int, default=60)
    ap.add_argument(
        "--equal-weights",
        action="store_true",
        help="set combination.parameters.weights (length = sub-query count, sum 1.0)",
    )
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--route-k", type=int, default=80)
    ap.add_argument("--pipeline", default=PIPELINE_NAME)
    args = ap.parse_args()

    knn_n = max(1, int(args.knn_routes))
    idx = get_index_name(IndexModel)
    all_vec = get_vector_fields(IndexModel)
    wmap = get_vector_weights(IndexModel)
    picked = sorted(all_vec, key=lambda f: wmap.get(f, 1.0), reverse=True)[:knn_n]
    query_text = (args.query or "").strip() or DEFAULT_QUERY

    print("=== rrf_pipeline_smoke ===")
    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    try:
        ver = await _cluster_version(client)
        if ver:
            print(f"cluster version: {ver}")

        qb = QueryBuilder()
        body = qb.build_dynamic_hybrid_search(
            IndexModel,
            query_text,
            size=int(args.route_k),
            bm25_factor=0.3,
            vector_factor=0.7,
            vector_fields=picked,
        )
        body["size"] = int(args.size)
        nq = _count_hybrid_subqueries(body)
        if nq < 2:
            print(f"ERROR: expected >=2 hybrid sub-queries, got {nq}", file=sys.stderr)
            return 2

        pipe_body = _build_rrf_pipeline_body(
            num_queries=nq,
            rank_constant=args.rank_constant,
            equal_weights=bool(args.equal_weights),
        )
        print(f"index: {idx}")
        print(f"query: {query_text!r}")
        print(f"hybrid sub-queries: {nq} (1 BM25 + {len(picked)} KNN)")
        print(f"pipeline: {args.pipeline!r}")
        print(
            "pipeline: score-ranker-processor / combination.technique=rrf"
            + (" + explicit equal weights" if args.equal_weights else " (no parameters.weights)")
        )

        ok, msg = await _put_pipeline(client, args.pipeline, pipe_body)
        if not ok:
            print("\nPUT pipeline FAILED (RRF / score-ranker likely unsupported or wrong JSON):")
            print(msg)
            print(
                "\nCheck: OpenSearch >= 2.19, search plugin / experimental flags per distribution. "
                "See: https://docs.opensearch.org/3.3/search-plugins/search-pipelines/score-ranker-processor/"
            )
            return 1

        print("\nPUT pipeline: OK")

        try:
            resp = await client.search(
                index=idx,
                body=body,
                params={"search_pipeline": args.pipeline},
            )
        except Exception as e:
            print("\nSEARCH with RRF pipeline FAILED:")
            print(repr(e))
            return 1

        hits = (resp.get("hits") or {}).get("hits") or []
        total = (resp.get("hits") or {}).get("total")
        print(f"\nSEARCH: OK  hits={len(hits)}  total={total}")
        for i, h in enumerate(hits[:5]):
            print(f"  [{i}] {h.get('_id')}  score={h.get('_score')}  mq={h.get('matched_queries')}")
        return 0
    finally:
        await opensearch_connector.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
