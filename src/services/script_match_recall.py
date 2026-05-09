"""
Script-match domain: async OpenSearch recall helpers.

These functions all require an active OpenSearch client and/or QueryBuilder.
They perform network I/O and should only be called from async contexts.

Public surface expected by script_match_service:
  - ensure_hybrid_pipeline
  - mget_sources_by_ids
  - global_bm25_chunked_merge_top_k
  - global_knn_top_k
  - fill_timeline_after_top1
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, List, Optional

from infra.logging.logger import logger as log
from infra.storage.opensearch.query_builder import QueryBuilder
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2

from services.script_match_query_builder import (
    INDEX_NAME,
    GLOBAL_KNN_QUERY_MAX_CHARS,
    _FILL_MAX_FOLLOW_SCENES,
    build_road_run_fallback_query_body,
    build_should_boosts,
    history_id_from_doc_id,
    parse_history_scene_from_doc_id,
    scene_doc_id,
)
from utils.search_utils import truncate_chars


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------


async def ensure_hybrid_pipeline(client: Any, *, pipeline_name: str, num_queries: int) -> str:
    """
    Upsert a normalization pipeline whose ``weights`` length matches ``hybrid.queries`` count.

    * ``num_queries == 1`` → ``{pipeline_name}-q1``
    * ``num_queries == 2`` → ``{pipeline_name}`` (base id, always PUT so cloud/local need no manual bootstrap)
    * ``num_queries >= 3`` → ``{pipeline_name}-q{num_queries}``
    """
    if not pipeline_name:
        return ""
    if num_queries <= 0:
        return pipeline_name

    if num_queries == 1:
        derived = f"{pipeline_name}-q1"
        weights = [1.0]
    else:
        if num_queries == 2:
            derived = pipeline_name
        else:
            derived = f"{pipeline_name}-q{num_queries}"
        bm25_w = 0.3
        vec_w = (1.0 - bm25_w) / float(num_queries - 1)
        weights = [bm25_w] + [vec_w] * (num_queries - 1)

    pipeline_body = {
        "description": f"Auto-generated hybrid pipeline for {num_queries} sub-queries",
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

    try:
        await client.http.put(f"/_search/pipeline/{derived}", body=pipeline_body)
    except Exception:
        return ""

    return derived


async def ensure_rrf_pipeline(
    client: Any,
    *,
    base_name: str,
    num_queries: int,
    weights: List[float],
    rank_constant: int = 60,
) -> str:
    """
    PUT a ``score-ranker-processor`` pipeline using ``technique: rrf``.
    ``weights`` length must equal ``hybrid.queries`` count (BM25 first, then each KNN).
    """
    if not base_name or num_queries <= 0:
        return ""
    w = [float(x) for x in weights]
    if len(w) != num_queries:
        log.warning(f"ensure_rrf_pipeline: weights len {len(w)} != num_queries {num_queries}")
        return ""
    ssum = sum(w) or 1.0
    w = [x / ssum for x in w]
    wkey = hashlib.md5(",".join(f"{x:.8f}" for x in w).encode()).hexdigest()[:10]
    derived = f"{base_name}-rrf-q{num_queries}-{wkey}"
    body: Dict[str, Any] = {
        "description": f"RRF fusion for {num_queries} hybrid sub-queries",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": int(rank_constant),
                        "parameters": {"weights": w},
                    },
                },
            },
        ],
    }
    try:
        await client.http.put(f"/_search/pipeline/{derived}", body=body)
    except Exception:
        log.exception("ensure_rrf_pipeline: PUT pipeline failed")
        return ""
    return derived


# ---------------------------------------------------------------------------
# mget helper
# ---------------------------------------------------------------------------


async def mget_sources_by_ids(client: Any, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    ids = [i for i in doc_ids if i]
    if not ids:
        return {}
    try:
        resp = await client.mget(index=INDEX_NAME, body={"ids": ids})
    except Exception as e:
        log.warning("mget failed for fill timeline: %s", e)
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for doc in (resp or {}).get("docs") or []:
        if not isinstance(doc, dict) or not doc.get("found"):
            continue
        did = str(doc.get("_id") or "")
        src = doc.get("_source")
        if did and isinstance(src, dict):
            out[did] = src
    return out


# ---------------------------------------------------------------------------
# Global BM25 chunked recall
# ---------------------------------------------------------------------------


async def global_bm25_chunked_merge_top_k(
    qb: QueryBuilder,
    client: Any,
    *,
    query_chunks: List[str],
    global_seg: Dict[str, Any],
    global_filters: Optional[List[Dict[str, Any]]] = None,
    global_k: int,
    text_fields: List[str],
    search_pipeline: Optional[str],
) -> List[str]:
    """
    BM25-only global recall per chunk (same should boosts on merged segment),
    merge hits by max score across chunks, return top *global_k* doc IDs.
    """
    should_g = build_should_boosts(global_seg)
    params_g = {"search_pipeline": search_pipeline} if search_pipeline else None
    size = max(1, int(global_k))

    async def one(chunk_q: str) -> Any:
        body = qb.build_bm25_only_search(
            model_class=CarInteriorAnalysisV2,
            query=chunk_q,
            size=size,
            search_fields=text_fields,
        )
        if (global_filters or []) or should_g:
            body["query"] = {
                "bool": {
                    "filter": global_filters or [],
                    "must": body["query"],
                    "should": should_g or [],
                    "minimum_should_match": 0,
                }
            }
        return await client.search(index=INDEX_NAME, body=body, params=params_g)

    if not query_chunks:
        return []

    if len(query_chunks) == 1:
        resp = await one(query_chunks[0])
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
        return [str(h.get("_id")) for h in hits if h.get("_id")][:size]

    responses = await asyncio.gather(*[one(cq) for cq in query_chunks])
    best: Dict[str, float] = {}
    for resp in responses:
        for h in (((resp or {}).get("hits") or {}).get("hits") or []):
            doc_id = str(h.get("_id") or "")
            if not doc_id:
                continue
            sc = float(h.get("_score") or 0.0)
            prev = best.get(doc_id)
            if prev is None or sc > prev:
                best[doc_id] = sc
    ranked = sorted(best.keys(), key=lambda x: best[x], reverse=True)
    return ranked[:size]


# ---------------------------------------------------------------------------
# Global KNN recall
# ---------------------------------------------------------------------------


async def global_knn_top_k(
    qb: QueryBuilder,
    client: Any,
    *,
    query_text: str,
    global_k: int,
    vector_field: str,
    search_pipeline: Optional[str],
) -> List[str]:
    q = truncate_chars(query_text, max_chars=GLOBAL_KNN_QUERY_MAX_CHARS)
    if not q:
        return []
    body = qb.build_knn_only_search(
        model_class=CarInteriorAnalysisV2,
        query=q,
        size=max(1, int(global_k)),
        vector_field=vector_field,
    )
    params = {"search_pipeline": search_pipeline} if search_pipeline else None
    resp = await client.search(index=INDEX_NAME, body=body, params=params)
    hits = (((resp or {}).get("hits") or {}).get("hits") or [])
    return [str(h.get("_id")) for h in hits if h.get("_id")]


# ---------------------------------------------------------------------------
# Timeline fill
# ---------------------------------------------------------------------------


async def fill_timeline_after_top1(
    client: Any,
    qb: QueryBuilder,
    *,
    q: str,
    top: List[Dict[str, Any]],
    seg_dur: float,
    text_fields: List[str],
    search_pipeline: Optional[str],
) -> Dict[str, Any]:
    """
    时长补全：

    1. **Anchor + follow**: 以 Top1 为锚，按 ``_scene_{id:03d}`` 顺序 mget 同一
       history 的后续分镜，直到累计 video_duration ≥ seg_dur。
    2. **路跑兜底**: 累计时长仍不足时（包括 ``top`` 为空的情况），发起
       ``generic_hq_road_run`` BM25 查询补足。

    ``top`` 为空（主查询 0 命中）时跳过锚点/跟随步骤直接进兜底，确保有时长需求
    的分镜至少能拿到路跑素材，而不是返回空列表。
    """
    empty: Dict[str, Any] = {
        "filled_hits": [],
        "filled_duration_seconds": 0.0,
        "segment_duration_seconds": float(seg_dur),
    }
    if seg_dur <= 0:
        return empty

    def _dur(s: Dict[str, Any]) -> float:
        try:
            return max(0.0, float(s.get("video_duration") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    filled: List[Dict[str, Any]] = []
    acc = 0.0

    # ------------------------------------------------------------------
    # Step 1: anchor + consecutive follow scenes from same video
    # ------------------------------------------------------------------
    if top:
        primary = top[0]
        anchor_id = str(primary.get("_id") or "")
        if anchor_id:
            hist, anchor_scene = parse_history_scene_from_doc_id(anchor_id)
            if hist and anchor_scene > 0:
                follow_ids = [
                    scene_doc_id(hist, sid)
                    for sid in range(anchor_scene + 1, anchor_scene + 1 + int(_FILL_MAX_FOLLOW_SCENES))
                ]
            else:
                follow_ids = []

            fetch_ids = [anchor_id] + [i for i in follow_ids if i != anchor_id]
            src_map = await mget_sources_by_ids(client, fetch_ids)

            anchor_src = src_map.get(anchor_id) or {}
            d0 = _dur(anchor_src)
            acc += d0
            filled.append(
                {
                    "_id": anchor_id,
                    "_score": primary.get("_score"),
                    "history_id": history_id_from_doc_id(anchor_id),
                    "video_duration": d0,
                    "role": "primary",
                }
            )

            if hist and anchor_scene > 0:
                for sid in range(anchor_scene + 1, anchor_scene + 1 + int(_FILL_MAX_FOLLOW_SCENES)):
                    if acc >= seg_dur:
                        break
                    did = scene_doc_id(hist, sid)
                    src = src_map.get(did)
                    if not isinstance(src, dict):
                        break
                    dv = _dur(src)
                    acc += dv
                    filled.append(
                        {
                            "_id": did,
                            "_score": None,
                            "history_id": hist,
                            "video_duration": dv,
                            "role": "follow",
                        }
                    )

    # ------------------------------------------------------------------
    # Step 2: road-run fallback — always runs when acc < seg_dur,
    #         including when top was empty (0 main-search hits).
    # ------------------------------------------------------------------
    used_ids = [str(x.get("_id") or "") for x in filled if x.get("_id")]

    if acc < seg_dur:
        fb_params = {"search_pipeline": search_pipeline} if search_pipeline else None
        fb_body = build_road_run_fallback_query_body(
            qb,
            q=q,
            text_fields=text_fields,
            seg_dur=float(seg_dur),
            used_ids=used_ids,
            use_duration_score=True,
        )
        try:
            fb_resp = await client.search(index=INDEX_NAME, body=fb_body, params=fb_params)
        except Exception as e:
            log.warning("road_run fallback search (with duration score) failed: %s", e)
            fb_body_plain = build_road_run_fallback_query_body(
                qb,
                q=q,
                text_fields=text_fields,
                seg_dur=float(seg_dur),
                used_ids=used_ids,
                use_duration_score=False,
            )
            try:
                fb_resp = await client.search(index=INDEX_NAME, body=fb_body_plain, params=fb_params)
            except Exception as e2:
                log.warning("road_run fallback search (plain) failed: %s", e2)
                fb_resp = {}

        for h in (((fb_resp or {}).get("hits") or {}).get("hits") or []):
            if acc >= seg_dur:
                break
            did = str(h.get("_id") or "")
            if not did or did in used_ids:
                continue
            src = h.get("_source") or {}
            dv = _dur(src if isinstance(src, dict) else {})
            used_ids.append(did)
            acc += dv
            filled.append(
                {
                    "_id": did,
                    "_score": h.get("_score"),
                    "history_id": history_id_from_doc_id(did),
                    "video_duration": dv,
                    "role": "fallback",
                }
            )

    if not filled:
        return empty

    return {
        "filled_hits": filled,
        "filled_duration_seconds": float(acc),
        "segment_duration_seconds": float(seg_dur),
    }
