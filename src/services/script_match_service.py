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
   cumulative video_duration ≥ seg["duration"]；仍不足则路跑兜底（function_score + script_score）。

4) 索引字段 generic_hq_road_run 由视频分析入库时打标；补时长兜底查询强制使用该字段。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from infra.storage.opensearch.query_builder import QueryBuilder
from infra.storage.opensearch_connector import opensearch_connector
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
    fill_timeline_after_top1,
    global_bm25_chunked_merge_top_k,
    global_knn_top_k,
)
from utils.search_utils import chunks_from_query_parts, rrf_fuse_ranked_lists


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------


async def _fetch_video_paths(history_ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for hid in [h for h in history_ids if h]:
        try:
            item = await video_analysis_db_service.get_history_item(hid)
            if item and isinstance(item, dict):
                vp = str(item.get("video_url") or "")
                if vp:
                    out[hid] = vp
        except Exception:
            continue
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
) -> List[Dict[str, Any]]:
    """
    For each script segment (Stage2 tags), search OpenSearch and map hits to DB paths.

    ``mode`` controls the per-segment BM25/hybrid strategy:

    * ``"zero"`` — keyword-only BM25, no vectors.
    * ``"lite"`` — BM25 (multi_match) + 1 KNN sub-query (primary vector field).
    * ``"full"`` / other — BM25 + up to 3 KNN sub-queries (field-selected).
    * ``"field_aligned"`` — field-aligned BM25 (each seg field → matching index
      field, ``bool.should``), no vectors.  Avoids cross-field IDF pollution.
    * ``"field_aligned_hybrid"`` — field-aligned BM25 (above) + KNN sub-queries
      (field-selected by ``choose_vector_fields``).  Recommended mode when both
      precision and recall matter.
    * ``"global_then_segment[_<sub_mode>]"`` — first do a global cascade recall
      to get ``candidate_ids``, then run per-segment with sub_mode as the inner
      strategy (e.g. ``"global_then_segment_field_aligned_hybrid"``).

    When ``seg["duration"] > 0``, also returns ``filled_hits`` (ordered): Top1
    primary, then same-video following scenes by ``_scene_{id:03d}`` until
    cumulative ``video_duration`` reaches ``duration``; if still short, appends
    ``generic_hq_road_run`` BM25 hits with soft duration proximity scoring.
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

    # ------------------------------------------------------------------
    # Step 1: optional global cascade recall → candidate_ids
    # ------------------------------------------------------------------
    candidate_ids: Optional[List[str]] = None
    if mode.startswith("global_then_segment"):
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
            candidate_ids = rrf_fuse_ranked_lists([bm25_ranked, knn_ranked], k=GLOBAL_RRF_K, top_n=int(global_k))
        else:
            candidate_ids = bm25_ranked

    # ------------------------------------------------------------------
    # Step 2: per-segment recall
    # ------------------------------------------------------------------
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue

        q = segment_query_text(seg)
        filters = build_filters(seg, relax_partitions=False)
        should_boosts = build_should_boosts(seg)

        seg_mode = mode
        if mode == "global_then_segment":
            seg_mode = "lite"
        elif mode.startswith("global_then_segment_"):
            seg_mode = mode.replace("global_then_segment_", "", 1) or "lite"
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
                bm25_factor=0.5,
                vector_factor=0.5,
                search_fields=text_fields,
                vector_fields=vector_fields,
            )

        if candidate_ids:
            filters = list(filters or [])
            filters.append({"ids": {"values": candidate_ids}})

        if filters or should_boosts:
            # IMPORTANT: push filters into every sub-query inside hybrid.queries so that
            # hard filters reliably apply in OpenSearch hybrid mode (outer bool.filter
            # may be ignored in some OS versions when the top-level query is hybrid).
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

        params = None
        if search_pipeline:
            qobj = body.get("query") or {}
            if isinstance(qobj, dict) and "hybrid" in qobj and isinstance(qobj.get("hybrid"), dict):
                n_q = len(qobj["hybrid"].get("queries") or [])
                pipeline_name = await ensure_hybrid_pipeline(c, pipeline_name=search_pipeline, num_queries=n_q)
                if pipeline_name:
                    params = {"search_pipeline": pipeline_name}
            else:
                params = {"search_pipeline": search_pipeline}

        resp = await c.search(index=INDEX_NAME, body=body, params=params)
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
        top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[: int(top_k)]]

        history_ids = [history_id_from_doc_id(t.get("_id") or "") for t in top]
        path_map = await _fetch_video_paths(history_ids)

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
                search_pipeline=search_pipeline,
            )
            fhs = fill_block.get("filled_hits") or []
            extra_hids = list(dict.fromkeys([str(h.get("history_id") or "") for h in fhs if h.get("history_id")]))
            path_map2 = await _fetch_video_paths(extra_hids)
            merged_paths = {**path_map, **path_map2}
            fill_block["filled_hits"] = [
                {**h, "video_path": merged_paths.get(str(h.get("history_id") or ""), "")} for h in fhs
            ]

        out.append(
            {
                "segment_id": seg.get("id"),
                "segment_text": seg.get("segment_text"),
                "query_text": q,
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
        )

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _merge_and_cap_global(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = merge_segments_for_global([s for s in segments if isinstance(s, dict)])
    return cap_global_merged_lists(merged)
