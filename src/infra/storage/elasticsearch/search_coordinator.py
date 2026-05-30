from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from infra.logging.logger import logger as log
from infra.storage.elasticsearch_connector import elasticsearch_connector
from infra.storage.elasticsearch.query_builder import QueryBuilder
from models.elasticsearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.elasticsearch_index.base_index import (
    get_vector_fields,
    get_searchable_fields,
    get_field_weights,
    get_vector_weights,
)
from services.script_match_query_builder import (
    INDEX_NAME,
    segment_query_text,
    build_filters,
    build_should_boosts,
    build_must_nots,
    choose_vector_fields,
    segment_duration_seconds,
    history_id_from_doc_id,
    parse_history_scene_from_doc_id,
    scene_doc_id,
)
from services.script_match_service import _fetch_video_paths, _backfill_top_hits_video_paths_from_filled

# Timeline-fill constants mapped for ES
_FILL_MAX_FOLLOW_SCENES = 32
_FILL_ROAD_RUN_FALLBACK_SIZE = 24
_FILL_ROAD_RUN_DURATION_BOOST_WEIGHT = 5.0
_FILL_ROAD_RUN_DURATION_BOOST_TAU_SEC = 1.2

_ROAD_RUN_DURATION_SCORE_PAINLESS = (
    "double seg = params.seg_dur; "
    "double w = params.weight; "
    "double tau = params.tau > 1e-6 ? params.tau : 0.05; "
    "if (doc['video_duration'].size() == 0) { return 0.0; } "
    "double v = doc['video_duration'].value; "
    "double diff = Math.abs(v - seg); "
    "return w * Math.exp(-diff / tau);"
)


async def mget_sources_by_ids_es(client: Any, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    ids = [i for i in doc_ids if i]
    if not ids:
        return {}
    try:
        resp = await client.mget(index=INDEX_NAME, ids=ids)
    except Exception as e:
        log.warning("ES mget failed for fill timeline: %s", e)
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


def build_road_run_fallback_query_body_es(
    qb: QueryBuilder,
    *,
    q: str,
    text_fields: List[str],
    seg_dur: float,
    used_ids: List[str],
    use_duration_score: bool,
    car_model: str = "",
    frame_size: str = "",
    frame_orientation: str = "",
) -> Dict[str, Any]:
    """Build the Elasticsearch query body for the generic road-run fallback step."""
    # We use qb.build_es_keyword_query to build the base query
    fb_body = qb.build_es_keyword_query(
        CarInteriorAnalysisV2,
        q or "路跑 外观 行驶",
        size=int(_FILL_ROAD_RUN_FALLBACK_SIZE),
        search_fields=text_fields,
    )
    inner_q = fb_body.get("query") or {"match_all": {}}
    must_clauses: List[Dict[str, Any]] = [
        inner_q,
        {"term": {"generic_hq_road_run": True}},
    ]
    if car_model and car_model != "未知":
        must_clauses.append({"term": {"car_model": {"value": car_model}}})
    if frame_size and frame_size != "未知":
        must_clauses.append({"term": {"frame_size": {"value": frame_size}}})
    
    from utils.frame_orientation import infer_frame_orientation
    orient = frame_orientation if frame_orientation and frame_orientation != "未知" else infer_frame_orientation(frame_size)
    if orient and orient != "未知":
        must_clauses.append({"term": {"frame_orientation": {"value": orient}}})

    bool_inner: Dict[str, Any] = {
        "must": must_clauses,
    }
    if used_ids:
        bool_inner["must_not"] = [{"ids": {"values": used_ids}}]

    if not use_duration_score or seg_dur <= 0:
        fb_body["query"] = {"bool": bool_inner}
        return fb_body

    fb_body["query"] = {
        "function_score": {
            "query": {"bool": bool_inner},
            "boost_mode": "sum",
            "functions": [
                {
                    "script_score": {
                        "script": {
                            "source": _ROAD_RUN_DURATION_SCORE_PAINLESS,
                            "lang": "painless",
                            "params": {
                                "seg_dur": float(seg_dur),
                                "weight": float(_FILL_ROAD_RUN_DURATION_BOOST_WEIGHT),
                                "tau": float(_FILL_ROAD_RUN_DURATION_BOOST_TAU_SEC),
                            },
                        }
                    }
                }
            ],
        }
    }
    return fb_body


async def fill_timeline_after_top1_es(
    client: Any,
    qb: QueryBuilder,
    *,
    q: str,
    top: List[Dict[str, Any]],
    seg_dur: float,
    text_fields: List[str],
    enable_road_run_fallback: bool = True,
    seg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

    # Step 1: anchor + follow scenes
    if top:
        primary = top[0]
        anchor_id = str(primary.get("_id") or "")
        if anchor_id:
            hist, anchor_scene = parse_history_scene_from_doc_id(anchor_id)
            if hist and anchor_scene > 0:
                follow_ids: List[str] = []
                for sid in range(anchor_scene + 1, anchor_scene + 1 + int(_FILL_MAX_FOLLOW_SCENES)):
                    k1 = scene_doc_id(hist, sid)
                    k2 = f"{hist}_{sid}"
                    follow_ids.append(k1)
                    if k2 != k1:
                        follow_ids.append(k2)
            else:
                follow_ids = []

            fetch_ids = [anchor_id] + [i for i in follow_ids if i != anchor_id]
            src_map = await mget_sources_by_ids_es(client, fetch_ids)

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
                    candidates = [scene_doc_id(hist, sid), f"{hist}_{sid}"]
                    src = None
                    did = ""
                    for cand in candidates:
                        s0 = src_map.get(cand)
                        if isinstance(s0, dict):
                            src = s0
                            did = cand
                            break
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

    # Step 2: road-run fallback
    used_ids = [str(x.get("_id") or "") for x in filled if x.get("_id")]
    if len(filled) == 0 and enable_road_run_fallback:
        car_model = str((seg or {}).get("car_model") or "")
        frame_size = str((seg or {}).get("frame_size") or "")
        
        frame_orientation = str((seg or {}).get("frame_orientation") or "")
        tokens_json = (seg or {}).get("_search_tokens_json")
        if isinstance(tokens_json, list):
            for t in tokens_json:
                if isinstance(t, dict) and str(t.get("sourceField") or "").strip() == "frame_orientation":
                    frame_orientation = str(t.get("text") or "").strip()
        
        fb_body = build_road_run_fallback_query_body_es(
            qb,
            q=q,
            text_fields=text_fields,
            seg_dur=float(seg_dur),
            used_ids=used_ids,
            use_duration_score=True,
            car_model=car_model,
            frame_size=frame_size,
            frame_orientation=frame_orientation,
        )
        try:
            fb_resp = await client.search(index=INDEX_NAME, body=fb_body)
        except Exception as e:
            log.warning("ES road_run fallback search (with duration score) failed: %s", e)
            fb_body_plain = build_road_run_fallback_query_body_es(
                qb,
                q=q,
                text_fields=text_fields,
                seg_dur=float(seg_dur),
                used_ids=used_ids,
                use_duration_score=False,
                car_model=car_model,
                frame_size=frame_size,
                frame_orientation=frame_orientation,
            )
            try:
                fb_resp = await client.search(index=INDEX_NAME, body=fb_body_plain)
            except Exception as e2:
                log.warning("ES road_run fallback search (plain) failed: %s", e2)
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


def _inner_seg_mode(outer_mode: str) -> str:
    if outer_mode == "global_then_segment":
        return "lite"
    if outer_mode.startswith("global_then_segment_"):
        return outer_mode.replace("global_then_segment_", "", 1) or "lite"
    return outer_mode


async def execute_client_side_rrf_es(
    client: Any,
    qb: QueryBuilder,
    model_class: Type[BaseIndex],
    query_text: str,
    size: int,
    text_fields: List[str],
    vector_fields: List[str],
    query_vector: List[float],
    bm25_weight: float,
    vector_weight: float,
    text_weights: Optional[Dict[str, float]] = None,
    vector_weights: Optional[Dict[str, float]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    should_boosts: Optional[List[Dict[str, Any]]] = None,
    must_nots: Optional[List[Dict[str, Any]]] = None,
    highlight_fields: Optional[List[str]] = None,
    field_aligned_seg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Executes BM25 query and KNN queries in parallel, then applies client-side RRF fusion.
    This works on standard/basic licensed Elasticsearch without requiring Platinum/Enterprise RRF.
    """
    from utils.search_utils import reciprocal_rank_fuse
    
    # 1. Build BM25 query
    if field_aligned_seg:
        # Field-aligned BM25 query
        bm25_query = qb.build_es_field_aligned_bm25_query(field_aligned_seg, size=size)
        bool_inner: Dict[str, Any] = {"must": [bm25_query]}
        if filters:
            bool_inner["filter"] = filters
        if should_boosts:
            bool_inner["should"] = should_boosts
            bool_inner["minimum_should_match"] = 0
        if must_nots:
            bool_inner["must_not"] = must_nots
        bm25_body = {
            "size": max(size * 3, 100),
            "query": {"bool": bool_inner},
            "_source": {"excludes": get_vector_fields(model_class)}
        }
    else:
        # Standard multi-match BM25 query
        bm25_body = qb.build_es_keyword_query(
            model_class=model_class,
            query=query_text,
            size=max(size * 3, 100),
            search_fields=text_fields,
            field_boosts=text_weights,
            filters=filters,
            should_boosts=should_boosts,
            must_nots=must_nots,
        )
    
    if highlight_fields and "highlight" not in bm25_body:
        bm25_body["highlight"] = {
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "require_field_match": False,
            "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in highlight_fields},
        }

    # 2. Build KNN queries
    knn_bodies = []
    for vf in vector_fields:
        knn_body = {
            "size": max(size * 3, 100),
            "knn": {
                "field": vf,
                "query_vector": query_vector,
                "k": max(size * 3, 100),
                "num_candidates": max(size * 5, 200)
            },
            "_source": {"excludes": get_vector_fields(model_class)}
        }
        if filters or must_nots or should_boosts:
            knn_filter_bool = {}
            if filters:
                knn_filter_bool["filter"] = filters
            if must_nots:
                knn_filter_bool["must_not"] = must_nots
            if should_boosts:
                knn_filter_bool["should"] = should_boosts
                knn_filter_bool["minimum_should_match"] = 0
            knn_body["knn"]["filter"] = {"bool": knn_filter_bool}
            
        if highlight_fields:
            knn_body["highlight"] = {
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "require_field_match": False,
                "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in highlight_fields},
            }
        knn_bodies.append(knn_body)

    # 3. Execute in parallel
    tasks = [client.search(index=INDEX_NAME, body=bm25_body)]
    for kb in knn_bodies:
        tasks.append(client.search(index=INDEX_NAME, body=kb))

    responses = await asyncio.gather(*tasks)

    # 4. Extract ranked lists
    ranked_lists = []
    id_to_hit = {}
    
    for resp in responses:
        current_list = []
        for h in (resp.get("hits") or {}).get("hits") or []:
            did = h["_id"]
            current_list.append(did)
            if did not in id_to_hit:
                id_to_hit[did] = h
            else:
                if "highlight" in h:
                    if "highlight" not in id_to_hit[did]:
                        id_to_hit[did]["highlight"] = h["highlight"]
                    else:
                        id_to_hit[did]["highlight"].update(h["highlight"])
        ranked_lists.append(current_list)

    # 5. Setup weights
    weights = [float(bm25_weight)]
    for vf in vector_fields:
        v_weight = 1.0
        if vector_weights and vf in vector_weights:
            v_weight = float(vector_weights[vf])
        weights.append(v_weight * float(vector_weight))

    # 6. Apply RRF
    fused_results = reciprocal_rank_fuse(ranked_lists, weights, rank_constant=60, top_n=size)

    # 7. Reconstruct hits
    final_hits = []
    for doc_id, score in fused_results:
        hit = id_to_hit[doc_id]
        hit["_score"] = score
        final_hits.append(hit)
        
    return final_hits


async def _match_one_segment_es(
    seg: Dict[str, Any],
    *,
    qb: QueryBuilder,
    client: Any,
    top_k: int,
    vector_field: str,
    text_fields: List[str],
    outer_mode: str,
    bm25_factor: float,
    vector_factor: float,
    shot_cards_version: str,
    enable_road_run_fallback: bool,
) -> Dict[str, Any]:
    q = segment_query_text(seg)
    filters = build_filters(seg, relax_partitions=False)
    should_boosts = build_should_boosts(seg)
    must_nots = build_must_nots(seg)

    seg_mode = _inner_seg_mode(outer_mode)
    vector_fields = choose_vector_fields(seg, mode=seg_mode, primary=vector_field)

    explain_requested = False
    hits = []
    body_debug = {}
    
    if seg_mode == "field_aligned":
        # Field aligned BM25 only (no retriever RRF)
        bm25_query = qb.build_es_field_aligned_bm25_query(seg, size=top_k)
        bool_inner: Dict[str, Any] = {"must": [bm25_query]}
        if filters:
            bool_inner["filter"] = filters
        if should_boosts:
            bool_inner["should"] = should_boosts
            bool_inner["minimum_should_match"] = 0
        if must_nots:
            bool_inner["must_not"] = must_nots
        body = {
            "size": top_k,
            "query": {"bool": bool_inner},
            "_source": {"excludes": get_vector_fields(CarInteriorAnalysisV2)}
        }
        body_debug = body
        explain_requested = True
        if explain_requested:
            body["explain"] = True
        resp = await client.search(index=INDEX_NAME, body=body)
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
    elif seg_mode == "field_aligned_hybrid":
        # RRF hybrid using field-aligned BM25 standard retriever
        query_vector = qb._generate_embedding(q)
        hits = await execute_client_side_rrf_es(
            client=client,
            qb=qb,
            model_class=CarInteriorAnalysisV2,
            query_text=q,
            size=top_k,
            text_fields=text_fields,
            vector_fields=vector_fields,
            query_vector=query_vector,
            bm25_weight=bm25_factor,
            vector_weight=vector_factor,
            filters=filters,
            should_boosts=should_boosts,
            must_nots=must_nots,
            field_aligned_seg=seg
        )
        body_debug = {"client_side_rrf": True, "mode": "field_aligned_hybrid", "q": q, "vector_fields": vector_fields}
    elif seg_mode == "zero" or not vector_fields:
        # Standard keyword BM25 only
        body = qb.build_es_keyword_query(
            model_class=CarInteriorAnalysisV2,
            query=q,
            size=top_k,
            search_fields=text_fields,
            filters=filters,
            should_boosts=should_boosts,
            must_nots=must_nots
        )
        body_debug = body
        explain_requested = True
        if explain_requested:
            body["explain"] = True
        resp = await client.search(index=INDEX_NAME, body=body)
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
    else:
        # Standard RRF query
        query_vector = qb._generate_embedding(q)
        hits = await execute_client_side_rrf_es(
            client=client,
            qb=qb,
            model_class=CarInteriorAnalysisV2,
            query_text=q,
            size=top_k,
            text_fields=text_fields,
            vector_fields=vector_fields,
            query_vector=query_vector,
            bm25_weight=bm25_factor,
            vector_weight=vector_factor,
            filters=filters,
            should_boosts=should_boosts,
            must_nots=must_nots
        )
        body_debug = {"client_side_rrf": True, "mode": "standard_hybrid", "q": q, "vector_fields": vector_fields}

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
        fill_block = await fill_timeline_after_top1_es(
            client,
            qb,
            q=q,
            top=top,
            seg_dur=seg_dur,
            text_fields=text_fields,
            enable_road_run_fallback=enable_road_run_fallback,
            seg=seg,
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
        "elasticsearch_body": body_debug,
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


async def match_script_tags_segments_es(
    segments: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    global_k: int = 200,
    search_pipeline: Optional[str] = None,
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
    For each segment, search Elasticsearch and map hits.
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
    await elasticsearch_connector.ensure_init()
    client = await elasticsearch_connector.get_client()

    valid_indices = [i for i, s in enumerate(segments) if isinstance(s, dict)]

    async def run_segment_at_index(idx: int) -> Tuple[int, Dict[str, Any]]:
        seg = segments[idx]
        assert isinstance(seg, dict)
        t0 = time.perf_counter()
        r = await _match_one_segment_es(
            seg,
            qb=qb,
            client=client,
            top_k=top_k,
            vector_field=vector_field,
            text_fields=text_fields,
            outer_mode=mode,
            bm25_factor=bm25_factor,
            vector_factor=vector_factor,
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


async def search_cards_es(
    query_text: str,
    req: Any,  # VideoAnalysisSearchRequest
    term_filters: List[Dict[str, Any]],
    should_boosts: List[Dict[str, Any]],
    must_not_multi_matches: List[Dict[str, Any]],
    size: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Search cards via Elasticsearch using native or client-side RRF fallback.
    """
    await elasticsearch_connector.ensure_init()
    client = await elasticsearch_connector.get_client()
    qb = QueryBuilder()

    vec_fields = get_vector_fields(CarInteriorAnalysisV2)
    text_fields = get_searchable_fields(CarInteriorAnalysisV2)

    search_mode = "precise"
    explain_requested = False

    _HIGHLIGHT_FIELDS = [
        "description",
        "subject",
        "object",
        "design_selling_points",
        "function_selling_points",
        "scenario_a",
        "scenario_b",
        "marketing_phrases",
        "appealing_audience",
        "scene_location",
    ]

    if not req.fuzzy:
        body = qb.build_es_keyword_query(
            model_class=CarInteriorAnalysisV2,
            query=query_text,
            size=size,
            search_fields=text_fields,
            field_boosts=req.text_weights,
            filters=term_filters,
            should_boosts=should_boosts,
            must_nots=must_not_multi_matches,
        )
        body["highlight"] = {
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "require_field_match": False,
            "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in _HIGHLIGHT_FIELDS},
        }
        search_mode = "precise"
        explain_requested = True
        if explain_requested:
            body["explain"] = True
        resp = await client.search(index=INDEX_NAME, body=body)
        hits = ((resp.get("hits") or {}).get("hits") or [])
    else:
        # Fuzzy RRF retriever search (using client-side RRF for license compliance)
        vec_weight_map = get_vector_weights(CarInteriorAnalysisV2).copy()
        if req.vector_weights:
            vec_weight_map.update(req.vector_weights)

        active_vecs = [f for f in vec_fields if float(vec_weight_map.get(f, 1.0) or 0) > 0]
        ordered_vecs = sorted(
            active_vecs, key=lambda f: float(vec_weight_map.get(f, 1.0)), reverse=True
        )

        q_vec = qb._generate_embedding(query_text)

        hits = await execute_client_side_rrf_es(
            client=client,
            qb=qb,
            model_class=CarInteriorAnalysisV2,
            query_text=query_text,
            size=size,
            text_fields=text_fields,
            vector_fields=ordered_vecs,
            query_vector=q_vec,
            bm25_weight=req.bm25_weight,
            vector_weight=req.vector_weight,
            text_weights=req.text_weights,
            vector_weights=req.vector_weights,
            filters=term_filters,
            should_boosts=should_boosts,
            must_nots=must_not_multi_matches,
            highlight_fields=_HIGHLIGHT_FIELDS,
        )
        search_mode = "fuzzy_rrf"

    # Handle Stage 1.5 Soft Fallback
    _STRICT_FIELDS = {"car_model", "frame_size", "frame_orientation"}
    non_strict_filters = [
        tf for tf in term_filters
        if "term" in tf and any(f not in _STRICT_FIELDS for f in tf["term"])
    ]
    if len(hits) == 0 and non_strict_filters:
        # Re-build query with soft fallback: non-strict fields removed from filters and optionally added as should boosts
        soft_filters = [
            tf for tf in term_filters
            if "term" in tf and any(f in _STRICT_FIELDS for f in tf["term"])
        ]
        
        # Soft fallback body
        if not req.fuzzy:
            body = qb.build_es_keyword_query(
                model_class=CarInteriorAnalysisV2,
                query=query_text,
                size=size,
                search_fields=text_fields,
                field_boosts=req.text_weights,
                filters=soft_filters,
                should_boosts=should_boosts,
                must_nots=must_not_multi_matches,
            )
            body["highlight"] = {
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "require_field_match": False,
                "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in _HIGHLIGHT_FIELDS},
            }
            if explain_requested:
                body["explain"] = True
                
            resp = await client.search(index=INDEX_NAME, body=body)
            hits = ((resp.get("hits") or {}).get("hits") or [])
        else:
            hits = await execute_client_side_rrf_es(
                client=client,
                qb=qb,
                model_class=CarInteriorAnalysisV2,
                query_text=query_text,
                size=size,
                text_fields=text_fields,
                vector_fields=ordered_vecs,
                query_vector=q_vec,
                bm25_weight=req.bm25_weight,
                vector_weight=req.vector_weight,
                text_weights=req.text_weights,
                vector_weights=req.vector_weights,
                filters=soft_filters,
                should_boosts=should_boosts,
                must_nots=must_not_multi_matches,
                highlight_fields=_HIGHLIGHT_FIELDS,
            )
        if hits:
            search_mode = f"{search_mode}_soft"

    # Handle Stage 2 Fallback (Road run fallback)
    if len(hits) == 0 and req.enable_road_run_fallback:
        strict_filters = [
            tf for tf in term_filters
            if "term" in tf and any(f in _STRICT_FIELDS for f in tf["term"])
        ]
        fallback_filters = strict_filters + [{"term": {"generic_hq_road_run": True}}]
        
        if not req.fuzzy:
            body = qb.build_es_keyword_query(
                model_class=CarInteriorAnalysisV2,
                query=query_text,
                size=size,
                search_fields=text_fields,
                field_boosts=req.text_weights,
                filters=fallback_filters,
                should_boosts=should_boosts,
                must_nots=must_not_multi_matches,
            )
            body["highlight"] = {
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "require_field_match": False,
                "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in _HIGHLIGHT_FIELDS},
            }
            if explain_requested:
                body["explain"] = True
                
            resp = await client.search(index=INDEX_NAME, body=body)
            hits = ((resp.get("hits") or {}).get("hits") or [])
        else:
            hits = await execute_client_side_rrf_es(
                client=client,
                qb=qb,
                model_class=CarInteriorAnalysisV2,
                query_text=query_text,
                size=size,
                text_fields=text_fields,
                vector_fields=ordered_vecs,
                query_vector=q_vec,
                bm25_weight=req.bm25_weight,
                vector_weight=req.vector_weight,
                text_weights=req.text_weights,
                vector_weights=req.vector_weights,
                filters=fallback_filters,
                should_boosts=should_boosts,
                must_nots=must_not_multi_matches,
                highlight_fields=_HIGHLIGHT_FIELDS,
            )
        search_mode = f"{search_mode}_fallback"

    return hits, search_mode
