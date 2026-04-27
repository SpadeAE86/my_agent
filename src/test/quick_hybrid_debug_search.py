"""
quick_hybrid_debug_search.py

Fast interactive-ish hybrid search debugger for OpenSearch.

Key feature: choose which vector fields participate in hybrid search
(OpenSearch hybrid often limits sub-queries; commonly <=4 knn routes).

PyCharm usage:
- Edit the CONFIG section below, then Run this file.
"""

from __future__ import annotations

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
from models.pydantic.opensearch_index.base_index import get_index_name, get_vector_fields  # noqa: E402
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2  # noqa: E402
from services.video_analysis_db_service import video_analysis_db_service  # noqa: E402

# Best-effort: make Windows console output UTF-8 to avoid mojibake.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =========================
# Search pipeline helper (same idea as tiny_hybrid_repro.py)
# =========================
async def ensure_hybrid_pipeline(client, *, pipeline_name: str, num_queries: int) -> None:
    """
    Ensure the hybrid search pipeline exists, otherwise OpenSearch returns:
    RequestError(400, 'illegal_argument_exception', 'Pipeline xxx is not defined')

    We use min_max normalization + arithmetic_mean combination.
    `num_queries` should equal: 1 (bm25) + N (knn vector routes).
    """
    if not pipeline_name:
        return
    if num_queries <= 0:
        return

    if num_queries == 1:
        weights = [1.0]
    else:
        bm25_w = 0.3
        vec_w = (1.0 - bm25_w) / float(num_queries - 1)
        weights = [bm25_w] + [vec_w] * (num_queries - 1)

    pipeline_body = {
        "description": "Post-processing pipeline for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": weights},
                    },
                }
            }
        ],
    }

    await client.http.put(f"/_search/pipeline/{pipeline_name}", body=pipeline_body)


# =========================
# CONFIG (edit and Run)
# =========================
QUERY = "地库 一键泊车"
PIPELINE = "hybrid-default"  # set "" / None to disable pipeline param

# Returned hits in the console
SIZE = 10

# Candidate pool size inside hybrid (larger improves fusion stability)
ROUTE_K = 200

# Hybrid often caps vector sub-queries. Keep <=4 unless your cluster allows more.
MAX_VECTORS = 4

# Choose which vector fields participate (order matters; will be truncated to MAX_VECTORS).
# Leave empty [] to auto-pick the first MAX_VECTORS vector fields.
PICKED_VECTOR_FIELDS: List[str] = [
    # "function_selling_points_vector",
    "marketing_phrases_vector",
    # "description_vector",
]

# Duration filter (seconds). Set to None to disable.
MIN_DURATION = None  # e.g. 2.5
MAX_DURATION = None  # e.g. 3.5

# Persist raw response for debugging (optional)
OUT_PATH = ""  # e.g. r"C:\temp\hybrid_debug.json"

# Print available vector fields and exit.
LIST_VECTORS_ONLY = False


def _apply_filters(body: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    if not filters:
        return body
    q = body.get("query") or {"match_all": {}}
    new_body = dict(body)
    new_body["query"] = {"bool": {"must": [q], "filter": []}}
    for field, val in filters.items():
        if isinstance(val, dict):
            new_body["query"]["bool"]["filter"].append({"range": {field: val}})
        elif isinstance(val, list):
            new_body["query"]["bool"]["filter"].append({"terms": {field: val}})
        else:
            new_body["query"]["bool"]["filter"].append({"term": {field: val}})
    return new_body


def _pick_vectors(requested: List[str], all_vectors: List[str], max_vectors: int) -> List[str]:
    if not requested:
        return all_vectors[:max_vectors]
    picked = [v for v in requested if v in all_vectors]
    return picked[:max_vectors]


async def _search(index: str, body: Dict[str, Any], *, pipeline: Optional[str]) -> Dict[str, Any]:
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    params = {"search_pipeline": pipeline} if pipeline else None
    return await c.search(index=index, body=body, params=params)


def _top_hits(resp: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    hits = ((resp.get("hits") or {}).get("hits") or [])[:limit]
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "movement": src.get("movement"),
                "shot_style": src.get("shot_style"),
                "shot_type": src.get("shot_type"),
                "video_duration": src.get("video_duration"),
                "description": (src.get("description") or "")[:120],
            }
        )
    return out


def _extract_history_id(doc_id: str) -> Optional[str]:
    """
    Expect doc id shaped like: "{history_id}_scene_001" (recommended).
    If not matched, return None.
    """
    if not doc_id:
        return None
    token = "_scene_"
    if token not in doc_id:
        return None
    return doc_id.split(token, 1)[0] or None


async def main() -> None:
    idx = get_index_name(CarInteriorAnalysisV2)
    all_vectors = get_vector_fields(CarInteriorAnalysisV2)

    if LIST_VECTORS_ONLY:
        print(json.dumps({"index": idx, "vector_fields": all_vectors}, ensure_ascii=False, indent=2))
        return

    picked = _pick_vectors(PICKED_VECTOR_FIELDS, all_vectors, max_vectors=int(MAX_VECTORS))
    if not picked:
        raise RuntimeError("No vector fields available/picked for hybrid search.")

    filters: Dict[str, Any] = {}
    if MIN_DURATION is not None or MAX_DURATION is not None:
        r: Dict[str, Any] = {}
        if MIN_DURATION is not None:
            r["gte"] = float(MIN_DURATION)
        if MAX_DURATION is not None:
            r["lte"] = float(MAX_DURATION)
        filters["video_duration"] = r

    qb = QueryBuilder()
    body = qb.build_dynamic_hybrid_search(
        CarInteriorAnalysisV2,
        QUERY,
        size=int(ROUTE_K),
        vector_fields=picked,
    )
    body["size"] = int(SIZE)
    body = _apply_filters(body, filters)

    try:
        # Ensure pipeline exists (consistent with tiny_hybrid_repro.py).
        await opensearch_connector.ensure_init()
        c = await opensearch_connector.get_client()
        await ensure_hybrid_pipeline(c, pipeline_name=PIPELINE, num_queries=1 + len(picked))

        resp = await _search(idx, body, pipeline=PIPELINE or None)
        summary = {
            "index": idx,
            "query": QUERY,
            "pipeline": PIPELINE,
            "picked_vector_fields": picked,
            "filters": filters,
            "top_hits": _top_hits(resp, limit=int(SIZE)),
        }

        # Best-effort: resolve video paths from DB (requires doc_id = history_id_scene_xxx).
        history_ids: List[str] = []
        for h in summary["top_hits"]:
            hid = _extract_history_id(str(h.get("_id") or ""))
            if hid and hid not in history_ids:
                history_ids.append(hid)

        video_paths: List[str] = []
        for hid in history_ids:
            try:
                item = await video_analysis_db_service.get_history_item(hid)
                if item and item.get("video_url"):
                    video_paths.append(str(item.get("video_url")))
            except Exception:
                continue

        summary["video_paths"] = video_paths
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if OUT_PATH:
            out_path = os.path.abspath(OUT_PATH)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"summary": summary, "raw": resp}, ensure_ascii=False, indent=2))
            print(f"\nWrote: {out_path}")
    finally:
        await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

