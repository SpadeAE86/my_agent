from __future__ import annotations

from typing import Any, Dict, List, Optional

from infra.storage.opensearch.query_builder import QueryBuilder
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index import index_v2_enums
from services.video_analysis_db_service import video_analysis_db_service


INDEX_NAME = "car_interior_analysis_v2"

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


def _segment_query_text(seg: Dict[str, Any]) -> str:
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
    out = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return " ".join(out)


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


def _build_filters(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []

    mv = str(seg.get("movement") or "").strip()
    if mv and mv != "未知":
        filters.append({"term": {"movement": mv}})

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
    add_term("product_status_scene", str(seg.get("product_status_scene") or ""), 1.1)
    add_term("car_color", str(seg.get("car_color") or ""), 1.1)
    add_term("time", str(seg.get("time") or ""), 1.05)
    add_term("weather", str(seg.get("weather") or ""), 1.05)
    # topic is a single keyword in the index (string).
    tp = seg.get("topic")
    if isinstance(tp, str):
        add_term("topic", tp, 1.25)
    elif isinstance(tp, list) and tp:
        add_term("topic", str(tp[0]), 1.25)

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
    if mode == "zero":
        return []
    if mode == "lite":
        return [primary]

    def has_list(k: str) -> bool:
        return bool(_truthy_list(seg.get(k)))

    candidates = [
        "marketing_phrases_vector",
        "function_selling_points_vector",
        "design_selling_points_vector",
        "description_vector",
        "scenario_a_vector",
        "scenario_b_vector",
    ]

    pruned: List[str] = []
    for f in candidates:
        if f == "function_selling_points_vector" and not has_list("function_selling_points"):
            continue
        if f == "design_selling_points_vector" and not has_list("design_selling_points"):
            continue
        if f == "scenario_a_vector" and not has_list("scenario_a"):
            continue
        if f == "scenario_b_vector" and not has_list("scenario_b"):
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
    For each script segment (Stage2 tags), search OpenSearch and map hits to DB paths.
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
        qg = _segment_query_text(global_seg)
        should_g = _build_should_boosts(global_seg)

        # Keep global recall cheap & stable: BM25-only + should boosts.
        body_g = qb.build_bm25_only_search(
            model_class=CarInteriorAnalysisV2,
            query=qg,
            size=int(global_k),
            search_fields=text_fields,
        )
        if should_g:
            body_g["query"] = {
                "bool": {"must": body_g["query"], "should": should_g, "minimum_should_match": 0}
            }
        params_g = {"search_pipeline": search_pipeline} if search_pipeline else None
        resp_g = await c.search(index=INDEX_NAME, body=body_g, params=params_g)
        hits_g = (((resp_g or {}).get("hits") or {}).get("hits") or [])
        candidate_ids = [str(h.get("_id")) for h in hits_g if h.get("_id")]

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
            body["query"] = {
                "bool": {
                    "filter": filters or [],
                    "must": body["query"],
                    "should": should_boosts or [],
                    "minimum_should_match": 0,
                }
            }

        params = {"search_pipeline": search_pipeline} if search_pipeline else None
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

