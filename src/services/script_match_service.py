from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from infra.logging.logger import logger as log
from infra.storage.opensearch.query_builder import QueryBuilder
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index import index_v2_enums
from services.video_analysis_db_service import video_analysis_db_service


INDEX_NAME = "car_interior_analysis_v2"


async def _ensure_hybrid_pipeline(client, *, pipeline_name: str, num_queries: int) -> str:
    """
    Cluster constraint: the search pipeline's normalization processor `weights` length must
    match the number of sub-queries in `hybrid.queries`, otherwise OpenSearch errors:
    "number of weights [x] must match number of sub-queries [y] in hybrid query".

    We treat `pipeline_name` as the default for the common 2-route hybrid (BM25 + 1 KNN).
    For other `num_queries`, we use a derived pipeline name and best-effort upsert it.
    """
    if not pipeline_name:
        return ""
    if num_queries <= 0:
        return pipeline_name
    if num_queries == 2:
        return pipeline_name

    derived = f"{pipeline_name}-q{num_queries}"
    if num_queries == 1:
        weights = [1.0]
    else:
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

# Stage2 script segments schema: models/pydantic/model_output_schema/seedtext_script_segments_schema.py
# (SeedtextIndexTagsSegment). Index vectors on CarInteriorAnalysisV2:
#   scenario_vector           ← embed(join(scenario_a) + join(scenario_b))
#   design_adjectives_vector  ← embed(design_adjectives)
#   function_adjectives_vector ← embed(function_adjectives)

# Global BM25 step (global_then_segment_*): merged segment text can become huge; IK analysis then
# expands to > Lucene's BooleanQuery.maxClauseCount (default 1024) → TransportError 500.
# Mitigation: split deduped query phrases into multiple chunks → parallel BM25 → merge by max(_score).
GLOBAL_CHUNK_MAX_CHARS = 160
GLOBAL_CHUNK_MAX_PHRASES = 18
GLOBAL_MAX_CHUNKS = 16
GLOBAL_STAGE_MAX_ITEMS_PER_LIST_FIELD = 80

# Global vector assist (optional): run a separate KNN on description_vector and fuse with BM25 via RRF.
GLOBAL_USE_KNN_ASSIST = True
GLOBAL_KNN_VECTOR_FIELD = "description_vector"
GLOBAL_KNN_QUERY_MAX_CHARS = 260
GLOBAL_RRF_K = 60

# To reduce maxClauseCount risk, keep global BM25 fields small.
GLOBAL_BM25_FIELDS = ["marketing_phrases", "function_selling_points", "design_selling_points"]


def _truthy_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _history_id_from_doc_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    if "_scene_" in doc_id:
        return doc_id.split("_scene_", 1)[0]
    return doc_id


def _segment_query_parts(seg: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for k in ["segment_text", "description"]:
        v = str(seg.get(k) or "").strip()
        if v:
            parts.append(v)

    for k in [
        "marketing_phrases",
        "marketing_tags",
        "function_selling_points",
        "design_selling_points",
        "design_adjectives",
        "function_adjectives",
        "scene_location",
        "scenario_a",
        "scenario_b",
        "extra_tags",
    ]:
        v = seg.get(k)
        if isinstance(v, list):
            parts.extend([str(x).strip() for x in v if str(x).strip()])
        elif isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # Optional: use topic/text signals if provided by the script rewriting stage.
    tp = seg.get("topic")
    if isinstance(tp, str) and tp.strip() and tp.strip() != "未知":
        parts.append(tp.strip())
    elif isinstance(tp, list) and tp:
        parts.extend([str(x).strip() for x in tp if str(x).strip() and str(x).strip() != "未知"])

    tx = seg.get("text")
    if isinstance(tx, list) and tx:
        parts.extend([str(x).strip() for x in tx if str(x).strip()])

    seen = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _segment_query_text(seg: Dict[str, Any]) -> str:
    return " ".join(_segment_query_parts(seg))


def _chunks_from_query_parts(
    parts: List[str],
    *,
    max_chars: int = GLOBAL_CHUNK_MAX_CHARS,
    max_phrases: int = GLOBAL_CHUNK_MAX_PHRASES,
    max_chunks: int = GLOBAL_MAX_CHUNKS,
) -> List[str]:
    """
    Pack deduped phrases into <=max_chunks strings, each under ~max_chars, for independent BM25 calls.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0

    for p in parts:
        if len(chunks) >= max_chunks:
            break
        piece = str(p or "").strip()
        if not piece:
            continue

        if len(piece) > max_chars:
            flush()
            for i in range(0, len(piece), max_chars):
                if len(chunks) >= max_chunks:
                    break
                chunks.append(piece[i : i + max_chars])
            continue

        add_len = len(piece) + (1 if cur else 0)
        if (cur and cur_len + add_len > max_chars) or (cur and len(cur) >= max_phrases):
            flush()

        if len(chunks) >= max_chunks:
            break

        cur.append(piece)
        cur_len += add_len

    flush()
    return [c for c in chunks if c.strip()]


def _truncate_chars(s: str, *, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def _rrf_fuse_ranked_lists(
    ranked_lists: List[List[str]],
    *,
    k: int = GLOBAL_RRF_K,
    top_n: int = 200,
) -> List[str]:
    """
    Reciprocal Rank Fusion (RRF):
      score(d) = sum_i 1 / (k + rank_i(d)), rank is 1-based.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for idx, doc_id in enumerate(lst or []):
            if not doc_id:
                continue
            rank = idx + 1
            scores[doc_id] = float(scores.get(doc_id, 0.0)) + 1.0 / (float(k) + float(rank))
    out = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return out[: int(top_n)]


def _merge_segments_for_global(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a pseudo segment dict for a single "global" recall.
    We union list-like fields across segments so global query has maximum coverage.
    """
    merged: Dict[str, Any] = {}

    # scalar-ish fields: keep first non-empty (and not "未知")
    for k in [
        "topic",
        "car_color",
        "weather",
        "time",
        "shot_style",
        "shot_type",
        "footage_type",
        "product_status_scene",
    ]:
        for seg in segments:
            v = seg.get(k)
            if isinstance(v, str) and v.strip() and v.strip() != "未知":
                merged[k] = v.strip()
                break

    # list-ish fields: union
    list_keys = [
        "marketing_phrases",
        "marketing_tags",
        "function_selling_points",
        "design_selling_points",
        "design_adjectives",
        "function_adjectives",
        "scene_location",
        "scenario_a",
        "scenario_b",
        "extra_tags",
        "object",
        "text",
        "video_usage",
        "person_detail",
        "key_traits",
    ]
    for k in list_keys:
        acc: List[str] = []
        for seg in segments:
            acc.extend(_truthy_list(seg.get(k)))
        # de-dup preserve order
        seen = set()
        out = []
        for x in acc:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        if out:
            merged[k] = out

    # movement: keep first non-empty for boosting (do not hard-filter globally)
    for seg in segments:
        mv = str(seg.get("movement") or "").strip()
        if mv and mv != "未知":
            merged["movement"] = mv
            break

    return merged


def _cap_global_merged_lists(
    merged: Dict[str, Any],
    *,
    max_each: int = GLOBAL_STAGE_MAX_ITEMS_PER_LIST_FIELD,
) -> Dict[str, Any]:
    """Trim unioned list fields so terms{} boosts stay small."""
    out = dict(merged)
    for k, v in list(out.items()):
        if isinstance(v, list) and len(v) > max_each:
            out[k] = v[:max_each]
    return out


async def _global_bm25_chunked_merge_top_k(
    qb: QueryBuilder,
    client: Any,
    *,
    query_chunks: List[str],
    global_seg: Dict[str, Any],
    global_k: int,
    text_fields: List[str],
    search_pipeline: Optional[str],
) -> List[str]:
    """
    Run BM25-only global recall per chunk (same should boosts on merged segment),
    merge hits by max score across chunks, return top global_k doc _ids.
    """
    should_g = _build_should_boosts(global_seg)
    params_g = {"search_pipeline": search_pipeline} if search_pipeline else None
    size = max(1, int(global_k))

    async def one(chunk_q: str):
        body = qb.build_bm25_only_search(
            model_class=CarInteriorAnalysisV2,
            query=chunk_q,
            size=size,
            search_fields=text_fields,
        )
        if should_g:
            body["query"] = {
                "bool": {"must": body["query"], "should": should_g, "minimum_should_match": 0}
            }
        return await client.search(index=INDEX_NAME, body=body, params=params_g)

    if not query_chunks:
        return []

    if len(query_chunks) == 1:
        resp = await one(query_chunks[0])
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
        out = [str(h.get("_id")) for h in hits if h.get("_id")]
        return out[:size]

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


async def _global_knn_top_k(
    qb: QueryBuilder,
    client: Any,
    *,
    query_text: str,
    global_k: int,
    vector_field: str,
    search_pipeline: Optional[str],
) -> List[str]:
    q = _truncate_chars(query_text, max_chars=GLOBAL_KNN_QUERY_MAX_CHARS)
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


def _build_filters(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []

    mv = str(seg.get("movement") or "").strip()
    if mv and mv != "未知":
        filters.append({"term": {"movement": {"value": mv}}})

    # Hard partitions (filter semantics are stable with hybrid):
    # - product_status_scene: 静态内饰/静态外观/路跑外观/... must not mix
    pss = str(seg.get("product_status_scene") or "").strip()
    if pss and pss != "未知":
        filters.append({"term": {"product_status_scene": {"value": pss}}})

    # - topic: treat as disjoint buckets in script-tag matching
    tp = seg.get("topic")
    if isinstance(tp, str):
        tv = tp.strip()
        if tv and tv != "未知":
            filters.append({"term": {"topic": {"value": tv}}})
    elif isinstance(tp, list) and tp:
        tv = str(tp[0]).strip()
        if tv and tv != "未知":
            filters.append({"term": {"topic": {"value": tv}}})

    vu = seg.get("video_usage")
    if isinstance(vu, list):
        vu2 = [str(x).strip() for x in vu if str(x).strip() and str(x).strip() != "未知"]
        if vu2:
            filters.append({"terms": {"video_usage": vu2}})

    return filters


def _build_should_boosts(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Soft constraints:
    - Keep recall high by using `should` with boosts instead of hard filters.
    """
    should: List[Dict[str, Any]] = []

    def add_term(field: str, val: str, boost: float):
        v = (val or "").strip()
        if not v or v == "未知":
            return
        should.append({"term": {field: {"value": v, "boost": boost}}})

    def add_terms(field: str, vals: List[str], boost: float):
        vs = [v for v in (vals or []) if v and v != "未知"]
        if not vs:
            return
        should.append({"terms": {field: vs, "boost": boost}})

    add_term("shot_style", str(seg.get("shot_style") or ""), 1.2)
    add_term("shot_type", str(seg.get("shot_type") or ""), 1.1)
    add_term("footage_type", str(seg.get("footage_type") or ""), 1.1)
    add_term("car_color", str(seg.get("car_color") or ""), 1.1)
    add_term("time", str(seg.get("time") or ""), 1.05)
    add_term("weather", str(seg.get("weather") or ""), 1.05)

    add_terms("person_detail", _truthy_list(seg.get("person_detail")), 1.05)

    # Boost key_traits recall: any script tag that matches the KEY_TRAITS enum should boost.
    trait_candidates: List[str] = []
    for t in (
        _truthy_list(seg.get("extra_tags"))
        + _truthy_list(seg.get("marketing_phrases"))
        + _truthy_list(seg.get("function_selling_points"))
        + _truthy_list(seg.get("design_selling_points"))
        + _truthy_list(seg.get("scene_location"))
        + _truthy_list(seg.get("object"))
    ):
        if t in index_v2_enums.KEY_TRAITS_CHOICES and t not in trait_candidates:
            trait_candidates.append(t)
    add_terms("key_traits", trait_candidates, 1.3)

    # Boost text recall if script provides a list of key on-screen strings/numbers.
    add_terms("text", _truthy_list(seg.get("text")), 1.25)

    return should




def _choose_vector_fields(seg: Dict[str, Any], *, mode: str, primary: str) -> List[str]:
    """
    Pick knn fields aligned with index mapping: scenario_vector uses scenario_a/scenario_b lists;
    design/function adjective vectors use their respective lists.
    """
    if mode == "zero":
        return []
    if mode == "lite":
        log.info(f"Mode=lite: using only primary vector field {primary} for stability and cost control.")
        return [primary]

    def has_list(k: str) -> bool:
        return bool(_truthy_list(seg.get(k)))

    candidates = [
        "marketing_phrases_vector",
        "function_selling_points_vector",
        "design_selling_points_vector",
        "description_vector",
        "scenario_vector",
        "design_adjectives_vector",
        "function_adjectives_vector",
    ]

    pruned: List[str] = []
    for f in candidates:
        if f == "function_selling_points_vector" and not has_list("function_selling_points"):
            continue
        if f == "design_selling_points_vector" and not has_list("design_selling_points"):
            continue
        if f == "scenario_vector" and not (has_list("scenario_a") or has_list("scenario_b")):
            continue
        if f == "design_adjectives_vector" and not has_list("design_adjectives"):
            continue
        if f == "function_adjectives_vector" and not has_list("function_adjectives"):
            continue
        pruned.append(f)

    out = [primary] + [f for f in pruned if f != primary]
    # Cap to 3 vector sub-queries for stability/cost
    return out[:3]


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
    For each script segment (Stage2 tags, SeedtextIndexTagsSegment), search OpenSearch and map hits to DB paths.
    Vector sub-queries follow CarInteriorAnalysisV2 (_choose_vector_fields): merged scenario_vector plus separate
    design/function adjective vectors when the segment lists are non-empty.
    Returns a list of segment-level results with top_k hits.
    """
    # Text fields used for BM25 multi_match within hybrid.
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
        # New fields may exist in newer index versions; harmless if unmapped in some clusters.
        "text",
    ]

    qb = QueryBuilder()
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    # Optional: global->segment cascade recall.
    # Step A: global recall once (k=global_k) to build candidate pool of doc _id
    # Step B: per-segment recall restricted to candidate pool (k=top_k)
    candidate_ids: Optional[List[str]] = None
    if mode.startswith("global_then_segment"):
        global_seg = _merge_segments_for_global([s for s in segments if isinstance(s, dict)])
        global_seg = _cap_global_merged_lists(global_seg)
        q_parts = _segment_query_parts(global_seg)
        query_chunks = _chunks_from_query_parts(
            q_parts,
            max_chars=GLOBAL_CHUNK_MAX_CHARS,
            max_phrases=GLOBAL_CHUNK_MAX_PHRASES,
            max_chunks=GLOBAL_MAX_CHUNKS,
        )
        # Global BM25 fields: keep small to avoid maxClauseCount
        global_bm25_fields = [f for f in GLOBAL_BM25_FIELDS if f in (text_fields or [])] or ["marketing_phrases"]

        bm25_ranked = await _global_bm25_chunked_merge_top_k(
            qb,
            c,
            query_chunks=query_chunks,
            global_seg=global_seg,
            global_k=int(global_k),
            text_fields=global_bm25_fields,
            search_pipeline=search_pipeline,
        )

        if GLOBAL_USE_KNN_ASSIST:
            # Use a short query for vector recall (prefer description/segment_text if present, else fall back).
            vec_query = " ".join([str(x) for x in q_parts[:60] if str(x).strip()])
            knn_ranked = await _global_knn_top_k(
                qb,
                c,
                query_text=vec_query,
                global_k=int(global_k),
                vector_field=GLOBAL_KNN_VECTOR_FIELD,
                search_pipeline=search_pipeline,
            )
            fused = _rrf_fuse_ranked_lists([bm25_ranked, knn_ranked], k=GLOBAL_RRF_K, top_n=int(global_k))
            candidate_ids = fused
        else:
            candidate_ids = bm25_ranked

    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        q = _segment_query_text(seg)
        filters = _build_filters(seg)
        should_boosts = _build_should_boosts(seg)
        # If we are in cascade mode, segment stage can choose the underlying segment mode.
        seg_mode = mode
        if mode == "global_then_segment":
            seg_mode = "lite"
        elif mode.startswith("global_then_segment_"):
            seg_mode = mode.replace("global_then_segment_", "", 1) or "lite"
        vector_fields = _choose_vector_fields(seg, mode=seg_mode, primary=vector_field)

        if seg_mode == "zero" or not vector_fields:
            # Keyword-only route (no vectors). Useful for quick debugging and exact-match heavy workloads.
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

        # Restrict segment recall to global candidate pool if present.
        if candidate_ids:
            filters = list(filters or [])
            filters.append({"ids": {"values": candidate_ids}})

        if filters or should_boosts:
            # IMPORTANT: OpenSearch `hybrid` queries do not reliably respect an outer `bool.filter`
            # wrapper in some versions/configs. To ensure hard filters apply in hybrid mode, we
            # push the same filters into EACH sub-query inside `hybrid.queries`.
            #
            # Symptom this fixes: `topic` / `product_status_scene` filters work in BM25-only,
            # but appear ignored when the query becomes `hybrid` (BM25 + kNN).
            q0 = body.get("query") or {}
            if isinstance(q0, dict) and "hybrid" in q0 and isinstance(q0.get("hybrid"), dict):
                hybrid_obj: Dict[str, Any] = q0["hybrid"]
                subqs = hybrid_obj.get("queries") or []
                wrapped_subqs: List[Dict[str, Any]] = []
                for subq in subqs:
                    if not isinstance(subq, dict):
                        continue
                    wrapped_subqs.append(
                        {
                            "bool": {
                                "filter": filters or [],
                                "must": [subq],
                                "should": should_boosts or [],
                                "minimum_should_match": 0,
                            }
                        }
                    )
                # If the original hybrid object carried `weights`, they may no longer match after
                # wrapping. Dropping them is safer than risking a 400 due to mismatch.
                body["query"] = {"hybrid": {"queries": wrapped_subqs}}
            else:
                must_list: List[Dict[str, Any]] = [body["query"]]
                body["query"] = {
                    "bool": {
                        "filter": filters or [],
                        "must": must_list,
                        "should": should_boosts or [],
                        "minimum_should_match": 0,
                    }
                }

        params = None
        if search_pipeline:
            qobj = body.get("query") or {}
            if isinstance(qobj, dict) and "hybrid" in qobj and isinstance(qobj.get("hybrid"), dict):
                n_q = len(qobj["hybrid"].get("queries") or [])
                pipeline_name = await _ensure_hybrid_pipeline(c, pipeline_name=search_pipeline, num_queries=n_q)
                if pipeline_name:
                    params = {"search_pipeline": pipeline_name}
            else:
                params = {"search_pipeline": search_pipeline}
        resp = await c.search(index=INDEX_NAME, body=body, params=params)
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
        top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[: int(top_k)]]

        history_ids = [_history_id_from_doc_id(t.get("_id") or "") for t in top]
        path_map = await _fetch_video_paths(history_ids)

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
                        "history_id": _history_id_from_doc_id(t.get("_id") or ""),
                        "video_path": path_map.get(_history_id_from_doc_id(t.get("_id") or ""), ""),
                    }
                    for t in top
                ],
            }
        )

    return out

