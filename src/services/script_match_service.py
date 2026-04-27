from __future__ import annotations

from typing import Any, Dict, List, Optional

from infra.storage.opensearch.query_builder import QueryBuilder
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
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
        "function_selling_points",
        "design_selling_points",
        "scene_location",
        "scenario_a",
        "scenario_b",
    ]:
        v = seg.get(k)
        if isinstance(v, list):
            parts.extend([str(x).strip() for x in v if str(x).strip()])

    seen = set()
    out = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return " ".join(out)


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

    add_terms("person_detail", _truthy_list(seg.get("person_detail")), 1.05)

    return should


def _choose_vector_fields(seg: Dict[str, Any], *, mode: str, primary: str) -> List[str]:
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
    search_pipeline: Optional[str] = "nlp-search-pipeline",
    vector_field: str = "marketing_phrases_vector",
    text_fields: Optional[List[str]] = None,
    mode: str = "lite",
) -> List[Dict[str, Any]]:
    """
    For each script segment (Stage2 tags), search OpenSearch and map hits to DB paths.
    Returns a list of segment-level results with top_k hits.
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
    ]

    qb = QueryBuilder()
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        q = _segment_query_text(seg)
        filters = _build_filters(seg)
        should_boosts = _build_should_boosts(seg)
        vector_fields = _choose_vector_fields(seg, mode=mode, primary=vector_field)

        body = qb.build_dynamic_hybrid_search(
            model_class=CarInteriorAnalysisV2,
            query=q,
            size=int(top_k),
            bm25_factor=0.5,
            vector_factor=0.5,
            search_fields=text_fields,
            vector_fields=vector_fields,
        )
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

