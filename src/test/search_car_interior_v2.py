# -*- coding: utf-8 -*-
"""
search_car_interior_v2.py

Common search playground for `car_interior_analysis_v2`, reusing QueryBuilder.

It runs multiple "routes" and prints:
- each route's top hits (doc_id + _score)
- the hybrid route's top hits (for rerank sanity)

Why multi-route?
- OpenSearch "hybrid" query does not reliably expose per-subquery contribution.
- Running routes separately gives you clear diagnostics: BM25 vs each vector field.

Run:
  python -m src.test.search_car_interior_v2
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import QueryBuilder
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import get_index_name, get_vector_fields


# === tweak here for quick runs ===
QUERY = "地库 一键泊车"
SIZE = 10

# Candidate pool per route (larger than SIZE to avoid truncation loss before fusion).
ROUTE_K = 200

# Where to persist outputs for iterative inspection.
OUTPUT_JSON = os.path.join(CURRENT_DIR, "query_output.json")

# Optional structured filters (term/terms/range).
# Example:
# FILTERS = {"movement": "泊车", "shot_style": "车内POV", "has_presenter": False}
FILTERS: Dict[str, Any] = {}

# Set True if you want OpenSearch to return `_explanation` for each hit.
# (May be heavy / not always helpful for hybrid.)
ENABLE_EXPLAIN = False

# For debugging pipeline details on the OpenSearch side.
ENABLE_PROFILE = False

# Limit vector sub-queries in hybrid to avoid OpenSearch hybrid max-subquery errors.
# Hybrid will use: 1x BM25 multi_match + len(HYBRID_VECTOR_FIELDS) knn queries.
HYBRID_VECTOR_FIELDS = [
    "marketing_phrases_vector",
    "function_selling_points_vector",
    "description_vector",
]

FUSION_METHOD = "rrf"  # "rrf" | "weighted"
RRF_K = 60
# Used when FUSION_METHOD="weighted": per-route weights
ROUTE_WEIGHTS = {
    "bm25": 1.0,
    "description_vector": 0.6,
    "function_selling_points_vector": 1.0,
    "marketing_phrases_vector": 1.2,
}


def _top_hits(resp: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    hits = ((resp.get("hits") or {}).get("hits") or [])[:limit]
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "_source_keys": sorted(list((h.get("_source") or {}).keys()))[:8],
                "_explanation": h.get("_explanation") if ENABLE_EXPLAIN else None,
            }
        )
    return out


async def _run_search(index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    return await c.search(index=index, body=body)


def _apply_common_flags(body: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(body)
    if ENABLE_EXPLAIN:
        body["explain"] = True
    if ENABLE_PROFILE:
        body["profile"] = True
    return body


def _apply_filters_to_body(body: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap the query in a bool/filter if filters provided.
    Works for both normal and hybrid queries by putting the existing query under bool.must.
    """
    if not filters:
        return body
    q = body.get("query") or {"match_all": {}}
    new_body = dict(body)
    new_body["query"] = {"bool": {"must": [q], "filter": []}}
    for field, val in filters.items():
        if isinstance(val, list):
            new_body["query"]["bool"]["filter"].append({"terms": {field: val}})
        elif isinstance(val, dict):
            new_body["query"]["bool"]["filter"].append({"range": {field: val}})
        else:
            new_body["query"]["bool"]["filter"].append({"term": {field: val}})
    return new_body


def _rrf_score(rank: int, *, k: int) -> float:
    # rank is 1-based
    return 1.0 / (k + rank)


def _fuse_routes(
    routes: Dict[str, List[Dict[str, Any]]],
    *,
    top_n: int,
    method: str,
    rrf_k: int,
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    routes: {route_name: [{"_id":..., "_score":...}, ...]} already ordered per-route.
    returns: final ranked list with per-route debug ranks/scores.
    """
    # Build per-route rank maps
    rank_map: Dict[str, Dict[str, int]] = {}
    score_map: Dict[str, Dict[str, float]] = {}
    all_ids = set()
    for route, hits in routes.items():
        rm: Dict[str, int] = {}
        sm: Dict[str, float] = {}
        for i, h in enumerate(hits):
            doc_id = h.get("_id")
            if not doc_id:
                continue
            if doc_id in rm:
                # keep best (highest) score if duplicated within route
                prev = sm.get(doc_id, float("-inf"))
                cur = float(h.get("_score") or 0.0)
                if cur > prev:
                    sm[doc_id] = cur
                    rm[doc_id] = i + 1
                continue
            rm[doc_id] = i + 1
            sm[doc_id] = float(h.get("_score") or 0.0)
            all_ids.add(doc_id)
        rank_map[route] = rm
        score_map[route] = sm

    # Compute fused score
    fused: List[Dict[str, Any]] = []
    for doc_id in all_ids:
        total = 0.0
        contrib: Dict[str, Any] = {}
        for route in routes.keys():
            r = rank_map[route].get(doc_id)
            s = score_map[route].get(doc_id)
            if r is None:
                continue
            if method == "rrf":
                part = _rrf_score(r, k=rrf_k)
            else:
                # weighted: use normalized rank contribution (coarse but stable)
                part = float(weights.get(route, 1.0)) * _rrf_score(r, k=rrf_k)
            total += part
            contrib[route] = {"rank": r, "score": s, "part": part}
        fused.append({"_id": doc_id, "final_score": total, "routes": contrib})

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused[:top_n]


async def main():
    idx = get_index_name(CarInteriorAnalysisV2)
    qb = QueryBuilder()

    # Route 1: pure keyword/BM25
    bm25_body = qb.build_bm25_only_search(CarInteriorAnalysisV2, QUERY, size=ROUTE_K)
    bm25_body = _apply_filters_to_body(_apply_common_flags(bm25_body), FILTERS)

    # Route 2: hybrid (bm25 + limited vector fields)
    # NOTE: this requires sentence-transformers (or a custom embedding model).
    hybrid_body = None
    try:
        hybrid_body = qb.build_dynamic_hybrid_search(
            CarInteriorAnalysisV2,
            QUERY,
            size=ROUTE_K,
            vector_fields=HYBRID_VECTOR_FIELDS,
        )
        hybrid_body = _apply_filters_to_body(_apply_common_flags(hybrid_body), FILTERS)
    except RuntimeError as e:
        hybrid_body = None
        print("WARN: hybrid disabled:", str(e))

    # Route 3+: each vector field alone (diagnostic)
    vector_fields = get_vector_fields(CarInteriorAnalysisV2)
    vector_bodies: List[Tuple[str, Dict[str, Any]]] = []
    for vf in vector_fields:
        try:
            body = qb.build_knn_only_search(CarInteriorAnalysisV2, QUERY, size=ROUTE_K, vector_field=vf)
        except RuntimeError as e:
            print("WARN: vector routes disabled:", str(e))
            vector_bodies = []
            break
        body = _apply_filters_to_body(_apply_common_flags(body), FILTERS)
        vector_bodies.append((vf, body))

    # Execute
    out: Dict[str, Any] = {"index": idx, "query": QUERY, "size": SIZE, "filters": FILTERS}
    try:
        bm25_resp = await _run_search(idx, bm25_body)
        out["bm25"] = {"top": _top_hits(bm25_resp, limit=min(ROUTE_K, 50))}

        if hybrid_body is not None:
            hybrid_resp = await _run_search(idx, hybrid_body)
            out["hybrid"] = {"top": _top_hits(hybrid_resp, limit=min(ROUTE_K, 50))}
        else:
            out["hybrid"] = {"top": [], "disabled": True}

        vec_out = {}
        for vf, body in vector_bodies:
            r = await _run_search(idx, body)
            vec_out[vf] = {"top": _top_hits(r, limit=min(ROUTE_K, 50))}
        out["vectors"] = vec_out

        # Build final fusion ranking from bm25 + selected vector routes (hybrid excluded due to instability).
        fusion_routes: Dict[str, List[Dict[str, Any]]] = {"bm25": out["bm25"]["top"]}
        for vf in ROUTE_WEIGHTS.keys():
            if vf == "bm25":
                continue
            if vf in vec_out:
                fusion_routes[vf] = vec_out[vf]["top"]

        out["final"] = {
            "method": FUSION_METHOD,
            "rrf_k": RRF_K,
            "route_k": ROUTE_K,
            "route_weights": ROUTE_WEIGHTS,
            "top": _fuse_routes(
                fusion_routes,
                top_n=SIZE,
                method=FUSION_METHOD,
                rrf_k=RRF_K,
                weights=ROUTE_WEIGHTS,
            ),
        }

    finally:
        await opensearch_connector.close()

    # Persist for iterative inspection
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Wrote:", OUTPUT_JSON)
    except Exception:
        pass
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

