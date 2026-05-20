"""
Script-match service — public entry point.

检索流程说明（match_script_tags_segments）
------------------------------------------
1) 模式前缀 global_then_segment_*（级联召回）
   a) 把所有分镜标签合并成一个 global_seg（列表字段做并集；标量如 topic 取首个非「未知」）。
   b) Global BM25：将 query 短语拆成多个 chunk，每 chunk 一次 BM25-only search，
      命中 doc 取各 chunk 上 _score 的最大值再排序，取 Top global_k _id。
      该步带 global_filters：bool.should 满足其一即可（minimum_should_match=1）——
      「movement AND product_status_scene AND car_model」三者皆有时为一支；并始终可与
      「generic_hq_road_run == true」支路 OR。
   c) Global KNN 辅助（可选 GLOBAL_USE_KNN_ASSIST）：对 description_vector 单独 KNN，
      与 BM25 列表做 RRF 融合。
   d) 得到 candidate_ids 后，每个分镜的检索在 filter 中追加 ids，限制只在候选集里打分。

2) 分镜阶段（对每个 seg）
   a) query_text：由 segment_text、description、各列表标签等拼成 BM25 查询串。
   b) filter（硬过滤）：relaxed 时 movement + video_usage；严格时分区为
      (movement∧product_status_scene∧car_model) ∨ generic_hq_road_run，另可加 video_usage。
   c) should（软加权）：镜头属性、环境、人物、key_words、text 等 term/terms boost。
   d) 主检索：seg_mode=zero 为纯 BM25；否则 build_dynamic_hybrid_search（BM25 + 1~多路 KNN）。
      hybrid 模式下把 filter 下推到 hybrid 的每个子查询 bool 里，确保硬过滤生效。

3) 补时长（内建，不做兄弟 vs 老二重排）：
   以 Top1 命中为锚，按 _scene_{id:03d} 在同一 history_id 下顺序 mget 后续分镜，直到
   cumulative video_duration 达到 seg["duration"]；仍不足则路跑兜底（function_score + script_score）。

4) 索引字段 generic_hq_road_run 由视频分析入库时打标；补时长兜底查询强制使用该字段。
"""
from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Tuple

from infra.storage.opensearch.query_builder import QueryBuilder
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.base_index import get_vector_fields
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from services.video_analysis_db_service import video_analysis_db_service

from services.script_match_query_builder import (
    INDEX_NAME,
    GLOBAL_BM25_FIELDS,
    GLOBAL_CHUNK_MAX_CHARS,
    GLOBAL_CHUNK_MAX_PHRASES,
    GLOBAL_MAX_CHUNKS,
    GLOBAL_KNN_VECTOR_FIELD,
    GLOBAL_RRF_K,
    GLOBAL_USE_KNN_ASSIST,
    build_field_aligned_bm25_query,
    build_field_aligned_hybrid_query,
    build_filters,
    build_should_boosts,
    cap_global_merged_lists,
    choose_vector_fields,
    history_id_from_doc_id,
    merge_segments_for_global,
    segment_query_parts,
    segment_query_text,
    segment_duration_seconds,
)
from services.script_match_recall import (
    ensure_hybrid_pipeline,
    ensure_rrf_pipeline,
    fill_timeline_after_top1,
    global_bm25_chunked_merge_top_k,
    global_knn_top_k,
)
from utils.search_utils import chunks_from_query_parts, rrf_fuse_ranked_lists
from infra.logging.logger import logger as log


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------


async def _fetch_video_paths(history_ids: List[str], shot_cards_version: str = "v1") -> Dict[str, str]:
    out: Dict[str, str] = {}
    ver: str = "v2" if (shot_cards_version or "v1").strip() == "v2" else "v1"
    for hid in [h for h in history_ids if h]:
        vp = await video_analysis_db_service.resolve_source_video_url_for_index_key(
            hid, shot_cards_version=ver  # type: ignore[arg-type]
        )
        if vp:
            out[hid] = vp
    return out


def _backfill_top_hits_video_paths_from_filled(result: Dict[str, Any]) -> None:
    """
    主检索 ``top_hits`` 的 video_path 来自首轮 path_map；若为空而补时长 ``filled_hits`` 已解析到 URL，
    按 history_id 回填，避免落库 hits 无 URL、前端「匹配结果 / Top5」全空。
    """
    tops = result.get("top_hits")
    if not isinstance(tops, list) or not tops:
        return
    filled = result.get("filled_hits") or []
    if not isinstance(filled, list) or not filled:
        return
    by_hid: Dict[str, str] = {}
    for h in filled:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("history_id") or "").strip()
        vp = str(h.get("video_path") or "").strip()
        if hid and vp:
            by_hid[hid] = vp
    if not by_hid:
        return
    for h in tops:
        if not isinstance(h, dict):
            continue
        if str(h.get("video_path") or "").strip():
            continue
        hid = str(h.get("history_id") or "").strip()
        if hid and hid in by_hid:
            h["video_path"] = by_hid[hid]


# ---------------------------------------------------------------------------
# Global recall + per-segment
# ---------------------------------------------------------------------------


def _merge_and_cap_global(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = merge_segments_for_global([s for s in segments if isinstance(s, dict)])
    return cap_global_merged_lists(merged)


def _query_body_is_hybrid(body: Dict[str, Any]) -> bool:
    q0 = body.get("query") or {}
    return isinstance(q0, dict) and "hybrid" in q0 and isinstance(q0.get("hybrid"), dict)


def _inner_seg_mode(outer_mode: str) -> str:
    if outer_mode == "global_then_segment":
        return "lite"
    if outer_mode.startswith("global_then_segment_"):
        return outer_mode.replace("global_then_segment_", "", 1) or "lite"
    return outer_mode


async def _compute_global_candidate_ids(
    segments: List[Dict[str, Any]],
    qb: QueryBuilder,
    c: Any,
    *,
    global_k: int,
    search_pipeline: Optional[str],
    text_fields: List[str],
) -> List[str]:
    global_seg = _merge_and_cap_global(segments)
    q_parts = segment_query_parts(global_seg)
    query_chunks = chunks_from_query_parts(
        q_parts,
        max_chars=GLOBAL_CHUNK_MAX_CHARS,
        max_phrases=GLOBAL_CHUNK_MAX_PHRASES,
        max_chunks=GLOBAL_MAX_CHUNKS,
    )
    global_bm25_fields = [f for f in GLOBAL_BM25_FIELDS if f in (text_fields or [])] or ["marketing_phrases"]
    global_filters = build_filters(global_seg, relax_partitions=False)

    bm25_ranked = await global_bm25_chunked_merge_top_k(
        qb,
        c,
        query_chunks=query_chunks,
        global_seg=global_seg,
        global_filters=global_filters,
        global_k=int(global_k),
        text_fields=global_bm25_fields,
        search_pipeline=search_pipeline,
    )

    if GLOBAL_USE_KNN_ASSIST:
        vec_query = " ".join([str(x) for x in q_parts[:60] if str(x).strip()])
        knn_ranked = await global_knn_top_k(
            qb,
            c,
            query_text=vec_query,
            global_k=int(global_k),
            vector_field=GLOBAL_KNN_VECTOR_FIELD,
            search_pipeline=search_pipeline,
        )
        return rrf_fuse_ranked_lists([bm25_ranked, knn_ranked], k=GLOBAL_RRF_K, top_n=int(global_k))
    return bm25_ranked


async def _match_one_segment(
    seg: Dict[str, Any],
    *,
    candidate_ids: Optional[List[str]],
    qb: QueryBuilder,
    c: Any,
    top_k: int,
    search_pipeline: Optional[str],
    vector_field: str,
    text_fields: List[str],
    outer_mode: str,
    bm25_factor: float,
    vector_factor: float,
    use_rrf: bool,
    shot_cards_version: str,
    enable_road_run_fallback: bool,
) -> Dict[str, Any]:
    q = segment_query_text(seg)
    filters = build_filters(seg, relax_partitions=False)
    should_boosts = build_should_boosts(seg)

    seg_mode = _inner_seg_mode(outer_mode)
    vector_fields = choose_vector_fields(seg, mode=seg_mode, primary=vector_field)

    if seg_mode == "field_aligned":
        body = build_field_aligned_bm25_query(seg, size=int(top_k))
    elif seg_mode == "field_aligned_hybrid":
        body = build_field_aligned_hybrid_query(seg, qb, size=int(top_k), vector_fields=vector_fields)
    elif seg_mode == "zero" or not vector_fields:
        body = qb.build_bm25_only_search(
            model_class=CarInteriorAnalysisV2,
            query=q,
            size=int(top_k),
            search_fields=text_fields,
        )
    else:
        body = qb.build_dynamic_hybrid_search(
            model_class=CarInteriorAnalysisV2,
            query=q,
            size=int(top_k),
            bm25_factor=float(bm25_factor),
            vector_factor=float(vector_factor),
            search_fields=text_fields,
            vector_fields=vector_fields,
        )

    if candidate_ids:
        filters = list(filters or [])
        filters.append({"ids": {"values": candidate_ids}})

    if filters or should_boosts:
        q0 = body.get("query") or {}
        if isinstance(q0, dict) and "hybrid" in q0 and isinstance(q0.get("hybrid"), dict):
            hybrid_obj: Dict[str, Any] = q0["hybrid"]
            subqs = hybrid_obj.get("queries") or []
            wrapped: List[Dict[str, Any]] = [
                {
                    "bool": {
                        "filter": filters or [],
                        "must": [subq],
                        "should": should_boosts or [],
                        "minimum_should_match": 0,
                    }
                }
                for subq in subqs
                if isinstance(subq, dict)
            ]
            body["query"] = {"hybrid": {"queries": wrapped}}
        else:
            body["query"] = {
                "bool": {
                    "filter": filters or [],
                    "must": [body["query"]],
                    "should": should_boosts or [],
                    "minimum_should_match": 0,
                }
            }

    params: Optional[Dict[str, str]] = None
    pipeline_base = search_pipeline or "nlp-search-pipeline"
    qobj = body.get("query") or {}
    if isinstance(qobj, dict) and "hybrid" in qobj and isinstance(qobj.get("hybrid"), dict):
        n_q = len(qobj["hybrid"].get("queries") or [])
        if use_rrf and n_q > 0:
            share = float(vector_factor) / max(1, n_q - 1) if n_q > 1 else float(vector_factor)
            raw_w = [float(bm25_factor)] + [share] * (n_q - 1)
            pipeline_name = await ensure_rrf_pipeline(
                c,
                base_name="script-match-rrf",
                num_queries=n_q,
                weights=raw_w,
            )
            if pipeline_name:
                params = {"search_pipeline": pipeline_name}
        else:
            pipeline_name = await ensure_hybrid_pipeline(c, pipeline_name=pipeline_base, num_queries=n_q)
            if pipeline_name:
                params = {"search_pipeline": pipeline_name}
    elif pipeline_base:
        params = {"search_pipeline": pipeline_base}

    # RRF hybrid 与 explain 不兼容；普通 hybrid / BM25 请求 explain（与 video_analysis 搜索一致）
    explain_requested = not (use_rrf and _query_body_is_hybrid(body))
    if explain_requested:
        body["explain"] = True

    resp = await c.search(index=INDEX_NAME, body=body, params=params)
    hits = (((resp or {}).get("hits") or {}).get("hits") or [])
    top: List[Dict[str, Any]] = []
    for h in hits[: int(top_k)]:
        item: Dict[str, Any] = {"_id": h.get("_id"), "_score": h.get("_score")}
        if explain_requested:
            ex = h.get("_explanation")
            if ex is not None:
                item["_explanation"] = ex
        top.append(item)

    history_ids = [history_id_from_doc_id(t.get("_id") or "") for t in top]
    path_map = await _fetch_video_paths(history_ids, shot_cards_version)

    seg_dur = segment_duration_seconds(seg)
    fill_block: Dict[str, Any] = {
        "filled_hits": [],
        "filled_duration_seconds": 0.0,
        "segment_duration_seconds": float(seg_dur),
    }
    if seg_dur > 0:
        fill_block = await fill_timeline_after_top1(
            c,
            qb,
            q=q,
            top=top,
            seg_dur=seg_dur,
            text_fields=text_fields,
            search_pipeline=pipeline_base,
            enable_road_run_fallback=enable_road_run_fallback,
        )
        fhs = fill_block.get("filled_hits") or []
        extra_hids = list(dict.fromkeys([str(h.get("history_id") or "") for h in fhs if h.get("history_id")]))
        path_map2 = await _fetch_video_paths(extra_hids, shot_cards_version)
        merged_paths = {**path_map, **path_map2}
        fill_block["filled_hits"] = [
            {**h, "video_path": merged_paths.get(str(h.get("history_id") or ""), "")} for h in fhs
        ]

    out = {
        "segment_id": seg.get("id"),
        "segment_text": seg.get("segment_text"),
        "query_text": q,
        "opensearch_body": copy.deepcopy(body),
        "search_params": dict(params) if params else {},
        "filters": filters,
        "should_boosts": should_boosts,
        "vector_fields": vector_fields,
        "top_hits": [
            {
                **t,
                "history_id": history_id_from_doc_id(t.get("_id") or ""),
                "video_path": path_map.get(history_id_from_doc_id(t.get("_id") or ""), ""),
            }
            for t in top
        ],
        "filled_hits": fill_block.get("filled_hits") or [],
        "filled_duration_seconds": float(fill_block.get("filled_duration_seconds") or 0.0),
        "segment_duration_seconds": float(fill_block.get("segment_duration_seconds") or 0.0),
    }
    _backfill_top_hits_video_paths_from_filled(out)
    return out


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


async def match_script_tags_segments(
    segments: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    global_k: int = 200,
    search_pipeline: Optional[str] = "nlp-search-pipeline",
    vector_field: str = "marketing_phrases_vector",
    text_fields: Optional[List[str]] = None,
    mode: str = "lite",
    shot_cards_version: str = "v1",
    concurrency: int = 1,
    bm25_factor: float = 0.5,
    vector_factor: float = 0.5,
    use_rrf: bool = False,
    with_timings: bool = False,
    enable_road_run_fallback: bool = True,
    on_segment_done: Optional[Callable[[int, Dict[str, Any]], Awaitable[None]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each script segment (Stage2 tags), search OpenSearch and map hits to DB paths.

    ``concurrency`` > 1 时并行分镜检索（每镜仍完整执行补时长等逻辑），适用于视频匹配等场景；
    ``concurrency`` == 1 保持与历史版本相同的串行顺序。

    ``shot_cards_version``: 解析 ``video_url`` 时传入 ``get_history_item`` 的 workspace 版本（v1/v2）。
    ``with_timings``: 为每条结果增加 ``elapsed_ms``（毫秒，含 Semaphore 排队等待）。
    ``on_segment_done``: 每个分镜检索完成后回调 ``(segment_index, match_dict)``，便于落库或 SSE。
    """
    text_fields = text_fields or [
        "marketing_phrases",
        "function_selling_points",
        "design_selling_points",
        "description",
        "subject",
        "object",
        "scene_location",
        "scenario_a",
        "scenario_b",
        "text",
    ]

    qb = QueryBuilder()
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    candidate_ids: Optional[List[str]] = None
    dict_segs = [s for s in segments if isinstance(s, dict)]
    if mode.startswith("global_then_segment") and dict_segs:
        candidate_ids = await _compute_global_candidate_ids(
            dict_segs,
            qb,
            c,
            global_k=global_k,
            search_pipeline=search_pipeline,
            text_fields=text_fields,
        )

    valid_indices = [i for i, s in enumerate(segments) if isinstance(s, dict)]

    async def run_segment_at_index(idx: int) -> Tuple[int, Dict[str, Any]]:
        seg = segments[idx]
        assert isinstance(seg, dict)
        t0 = time.perf_counter()
        r = await _match_one_segment(
            seg,
            candidate_ids=candidate_ids,
            qb=qb,
            c=c,
            top_k=top_k,
            search_pipeline=search_pipeline,
            vector_field=vector_field,
            text_fields=text_fields,
            outer_mode=mode,
            bm25_factor=bm25_factor,
            vector_factor=vector_factor,
            use_rrf=use_rrf,
            shot_cards_version=shot_cards_version,
            enable_road_run_fallback=enable_road_run_fallback,
        )
        if with_timings:
            r["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return idx, r

    if not valid_indices:
        return []

    if concurrency <= 1:
        out: List[Dict[str, Any]] = []
        for idx in valid_indices:
            ix, r = await run_segment_at_index(idx)
            if on_segment_done is not None:
                await on_segment_done(ix, r)
            out.append(r)
        return out

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def gated(idx: int) -> Tuple[int, Dict[str, Any]]:
        async with sem:
            ix, r = await run_segment_at_index(idx)
            if on_segment_done is not None:
                await on_segment_done(ix, r)
            return ix, r

    pairs = await asyncio.gather(*(gated(i) for i in valid_indices))
    pairs.sort(key=lambda x: x[0])
    return [p[1] for p in pairs]
