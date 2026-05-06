"""
Script-match domain: OpenSearch DSL query builders.

All functions here are pure (no async, no client calls).  They only build
Python dicts that represent OpenSearch query DSL fragments.

Import from this module instead of from script_match_service when you need
to compose or test individual filter / boost clauses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from infra.storage.opensearch.query_builder import QueryBuilder
from models.pydantic.opensearch_index.base_index import get_vector_fields, get_vector_weights
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index import index_v2_enums
from utils.search_utils import truthy_list, chunks_from_query_parts  # noqa: F401 (re-exported for convenience)

# Single source-of-truth for the "unknown / not set" sentinel used across enum fields.
_UNKNOWN = index_v2_enums.UNKNOWN  # "未知"

# Pre-computed at import time (constant for the lifetime of the process).
_ALL_VECTOR_FIELDS: List[str] = get_vector_fields(CarInteriorAnalysisV2)
_VECTOR_WEIGHTS: Dict[str, float] = get_vector_weights(CarInteriorAnalysisV2)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

INDEX_NAME = "car_interior_analysis_v2"

# ---------------------------------------------------------------------------
# Global recall hyper-params
# ---------------------------------------------------------------------------

# BM25 chunking (avoids Lucene maxClauseCount > 1024 after IK expansion)
GLOBAL_CHUNK_MAX_CHARS = 160
GLOBAL_CHUNK_MAX_PHRASES = 18
GLOBAL_MAX_CHUNKS = 16
GLOBAL_STAGE_MAX_ITEMS_PER_LIST_FIELD = 80

# KNN assist
GLOBAL_USE_KNN_ASSIST = True
GLOBAL_KNN_VECTOR_FIELD = "description_vector"
GLOBAL_KNN_QUERY_MAX_CHARS = 260
GLOBAL_RRF_K = 60

# Fields used in global BM25 step (kept small to limit clause count)
GLOBAL_BM25_FIELDS = ["marketing_phrases", "function_selling_points", "design_selling_points"]

# ---------------------------------------------------------------------------
# Timeline-fill hyper-params
# ---------------------------------------------------------------------------

# Max follow scenes from the same video to attempt after Top1
_FILL_MAX_FOLLOW_SCENES = 32
# Max road-run fallback hits
_FILL_ROAD_RUN_FALLBACK_SIZE = 24
# Duration proximity boost — exponential decay: w * exp(-|v - seg| / tau)
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

# ---------------------------------------------------------------------------
# Should-boost weights — tweak values here to adjust per-field ranking impact
# ---------------------------------------------------------------------------

SHOULDBOOST_WEIGHTS: Dict[str, float] = {
    "shot_style": 1.2,
    "shot_type": 1.1,
    "footage_type": 1.1,
    "car_color": 1.1,
    "car_model": 1.1,
    "topic": 1.1,
    "design_adjectives": 1.0,
    "function_adjectives": 1.0,
    "design_selling_points": 1.1,
    "function_selling_points": 1.1,
    "scene_location": 1.05,
    "time": 1.05,
    "weather": 1.05,
    "person_detail": 1.05,
    "key_words": 1.3,
    "text": 1.25,
    "camera_movement": 1.15,
    # video_usage：脚本端希望的素材用途 vs 索引端 AI 打标，两套标注来源不同，
    # 不适合做硬过滤 → 移到 should 软加权。
    "video_usage": 0.9,
}

# ---------------------------------------------------------------------------
# Doc-ID helpers
# ---------------------------------------------------------------------------


def history_id_from_doc_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    if "_scene_" in doc_id:
        return doc_id.split("_scene_", 1)[0]
    return doc_id


def parse_history_scene_from_doc_id(doc_id: str) -> tuple[str, int]:
    if not doc_id or "_scene_" not in doc_id:
        return "", 0
    hid, rest = doc_id.split("_scene_", 1)
    try:
        return hid, int(rest, 10)
    except ValueError:
        return hid or "", 0


def scene_doc_id(history_id: str, scene_id: int) -> str:
    return f"{history_id}_scene_{int(scene_id):03d}"


def segment_duration_seconds(seg: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(seg.get("duration") or 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Segment query text extraction
# ---------------------------------------------------------------------------


def _text_for_bm25(v: Any) -> str:
    """
    Extract a BM25-safe text string from a segment field value.

    - **List**: join non-empty items, skipping any that equal the "未知" sentinel.
      "未知" is the model's output for "not applicable / not observed"; treating
      it as a search term would match other docs that also have "未知", which is
      meaningless (two unknowns do NOT constitute a real match).
    - **Str**: return as-is unless the ENTIRE value IS "未知".
      A free-text field like ``description`` may legitimately contain the word
      "未知" mid-sentence ("速度未知的车辆…"); only skip the degenerate case
      where the whole field collapsed to the sentinel.
    """
    if isinstance(v, list):
        return " ".join([s for x in v for s in [str(x).strip()] if s and s != _UNKNOWN])
    if isinstance(v, str):
        t = v.strip()
        return "" if t == _UNKNOWN else t
    return ""


def segment_query_parts(seg: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for k in ["segment_text", "description"]:
        v = _text_for_bm25(seg.get(k))
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
            parts.extend([s for x in v for s in [str(x).strip()] if s and s != _UNKNOWN])
        elif isinstance(v, str):
            t = v.strip()
            if t and t != _UNKNOWN:
                parts.append(t)

    tp = seg.get("topic")
    if isinstance(tp, str) and tp.strip() and tp.strip() != _UNKNOWN:
        parts.append(tp.strip())
    elif isinstance(tp, list) and tp:
        parts.extend([str(x).strip() for x in tp if str(x).strip() and str(x).strip() != _UNKNOWN])

    tx = seg.get("text")
    if isinstance(tx, list) and tx:
        parts.extend([str(x).strip() for x in tx if str(x).strip()])

    seen: set = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def segment_query_text(seg: Dict[str, Any]) -> str:
    return " ".join(segment_query_parts(seg))


def extract_key_words(seg: Dict[str, Any]) -> List[str]:
    """
    Extract ``KEY_WORDS_CHOICES`` matches from the segment.

    Priority:
    1. Explicit ``seg["key_words"]`` — validated against the enum set.
    2. Inferred — scan **every** segment text / tag field for tokens that
       happen to match an enum value.  ``key_words`` in the index is the
       high-signal intent channel, so we want any field's signal to contribute
       (e.g. "降噪" in ``marketing_phrases`` or "安静" in ``extra_tags`` both
       tell us the clip is about NVH).
    """

    def _dedup(xs: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for x in xs:
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    explicit = [t for t in truthy_list(seg.get("key_words")) if t in index_v2_enums.KEY_WORDS_CHOICES]
    if explicit:
        return _dedup(explicit)

    # Scan ALL text/tag fields — the enum check prevents false positives.
    candidate_tokens: List[str] = []
    for field in [
        "extra_tags",
        "marketing_phrases",
        "marketing_tags",
        "function_selling_points",
        "design_selling_points",
        "scene_location",
        "scenario_a",
        "scenario_b",
        "design_adjectives",
        "function_adjectives",
        "object",
        "segment_text",
        "description",
        "subject",
    ]:
        v = seg.get(field)
        if isinstance(v, list):
            candidate_tokens.extend([str(x).strip() for x in v if str(x).strip()])
        elif isinstance(v, str) and v.strip():
            # Free-text: split on common delimiters so sub-phrases can match
            candidate_tokens.extend([t.strip() for t in v.replace("，", " ").replace("。", " ").split() if t.strip()])

    inferred = [t for t in candidate_tokens if t in index_v2_enums.KEY_WORDS_CHOICES]
    return _dedup(inferred)


# ---------------------------------------------------------------------------
# Global segment merging
# ---------------------------------------------------------------------------


def merge_segments_for_global(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a pseudo-segment dict by unioning list fields and picking first scalar."""
    merged: Dict[str, Any] = {}

    for k in [
        "topic",
        "car_model",
        "car_color",
        "weather",
        "time",
        "shot_style",
        "shot_type",
        "footage_type",
        "product_status_scene",
        "camera_movement",
    ]:
        for seg in segments:
            v = seg.get(k)
            if isinstance(v, str) and v.strip() and v.strip() != _UNKNOWN:
                merged[k] = v.strip()
                break

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
        "key_words",
    ]
    for k in list_keys:
        acc: List[str] = []
        for seg in segments:
            acc.extend(truthy_list(seg.get(k)))
        seen: set = set()
        out: List[str] = []
        for x in acc:
            if x in seen or x == _UNKNOWN:
                continue
            seen.add(x)
            out.append(x)
        if out:
            merged[k] = out

    for seg in segments:
        mv = str(seg.get("movement") or "").strip()
        if mv and mv != _UNKNOWN:
            merged["movement"] = mv
            break

    return merged


def cap_global_merged_lists(
    merged: Dict[str, Any],
    *,
    max_each: int = GLOBAL_STAGE_MAX_ITEMS_PER_LIST_FIELD,
) -> Dict[str, Any]:
    """Trim unioned list fields to keep terms{} boosts small."""
    out = dict(merged)
    for k, v in list(out.items()):
        if isinstance(v, list) and len(v) > max_each:
            out[k] = v[:max_each]
    return out


# ---------------------------------------------------------------------------
# OpenSearch DSL builders (pure, return dicts only)
# ---------------------------------------------------------------------------


def build_filters(seg: Dict[str, Any], *, relax_partitions: bool = False) -> List[Dict[str, Any]]:
    """
    relaxed (``relax_partitions=True``): movement + video_usage.

    strict (``relax_partitions=False``):
      (any non-empty subset of {movement, product_status_scene, car_model}) ∨ generic_hq_road_run
      optionally AND-ed with video_usage terms.

    The partition arm uses **whatever terms are available** (1-3), not only when
    all three are present.  Requiring all three caused segments that lack
    ``car_model`` (often supplied at the script level, not inside each segment
    dict) to collapse to ``road_run=True`` only — filtering out all non-road-run
    clips and returning zero hits.
    """
    filters: List[Dict[str, Any]] = []

    if relax_partitions:
        mv = str(seg.get("movement") or "").strip()
        if mv and mv != _UNKNOWN:
            filters.append({"term": {"movement": {"value": mv}}})
        vu = seg.get("video_usage")
        if isinstance(vu, list):
            vu2 = [str(x).strip() for x in vu if str(x).strip() and str(x).strip() != _UNKNOWN]
            if vu2:
                filters.append({"terms": {"video_usage": vu2}})
        return filters

    mv = str(seg.get("movement") or "").strip()
    pss = str(seg.get("product_status_scene") or "").strip()
    cm = str(seg.get("car_model") or "").strip()

    partition_terms: List[Dict[str, Any]] = []
    if mv and mv != _UNKNOWN:
        partition_terms.append({"term": {"movement": {"value": mv}}})
    if pss and pss != _UNKNOWN:
        partition_terms.append({"term": {"product_status_scene": {"value": pss}}})
    if cm and cm != _UNKNOWN:
        partition_terms.append({"term": {"car_model": {"value": cm}}})

    # Build the OR: (available partition terms) OR road_run.
    # Always include road_run so clips without perfect partition match can still
    # be retrieved as a last resort.
    should_parts: List[Dict[str, Any]] = []
    if partition_terms:
        # Use whatever partition fields are available — even 1 or 2 is useful.
        should_parts.append({"bool": {"filter": partition_terms}})
    should_parts.append({"term": {"generic_hq_road_run": True}})

    if len(should_parts) == 1:
        # Only road_run available (0 partition fields) → hard-require it.
        filters.append(should_parts[0])
    else:
        filters.append({"bool": {"should": should_parts, "minimum_should_match": 1}})

    # video_usage is intentionally NOT added as a hard filter here.
    # The script-side video_usage ("希望用来做什么") and the index-side video_usage
    # ("AI labelled this clip as") come from different labelling contexts and often
    # don't align exactly.  Adding it as AND would silently kill results for valid
    # clips (e.g. script says "烘托氛围" but indexed as "使用场景展示").
    # It is instead added as a soft should-boost in build_should_boosts().

    return filters


def build_should_boosts(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Soft constraints via ``should`` clauses with per-field boost weights.
    Weights are centralised in ``SHOULDBOOST_WEIGHTS``.
    """
    should: List[Dict[str, Any]] = []
    w = SHOULDBOOST_WEIGHTS

    def add_term(field: str, val: str, boost: float) -> None:
        v = (val or "").strip()
        if not v or v == _UNKNOWN:
            return
        should.append({"term": {field: {"value": v, "boost": boost}}})

    def add_terms(field: str, vals: List[str], boost: float) -> None:
        vs = [v for v in (vals or []) if v and v != _UNKNOWN]
        if not vs:
            return
        should.append({"terms": {field: vs, "boost": boost}})

    add_term("shot_style", str(seg.get("shot_style") or ""), float(w["shot_style"]))
    add_term("shot_type", str(seg.get("shot_type") or ""), float(w["shot_type"]))
    add_term("footage_type", str(seg.get("footage_type") or ""), float(w["footage_type"]))
    add_term("car_color", str(seg.get("car_color") or ""), float(w["car_color"]))
    add_term("car_model", str(seg.get("car_model") or ""), float(w["car_model"]))

    tp = seg.get("topic")
    if isinstance(tp, str):
        add_term("topic", tp, float(w["topic"]))
    elif isinstance(tp, list) and tp:
        add_term("topic", str(tp[0]).strip(), float(w["topic"]))

    add_terms("design_adjectives", truthy_list(seg.get("design_adjectives")), float(w["design_adjectives"]))
    add_terms("function_adjectives", truthy_list(seg.get("function_adjectives")), float(w["function_adjectives"]))
    add_terms("design_selling_points", truthy_list(seg.get("design_selling_points")), float(w["design_selling_points"]))
    add_terms("function_selling_points", truthy_list(seg.get("function_selling_points")), float(w["function_selling_points"]))
    add_terms("scene_location", truthy_list(seg.get("scene_location")), float(w["scene_location"]))
    add_term("time", str(seg.get("time") or ""), float(w["time"]))
    add_term("weather", str(seg.get("weather") or ""), float(w["weather"]))
    add_term("camera_movement", str(seg.get("camera_movement") or ""), float(w["camera_movement"]))
    add_terms("person_detail", truthy_list(seg.get("person_detail")), float(w["person_detail"]))
    add_terms("key_words", extract_key_words(seg), float(w["key_words"]))
    add_terms("text", truthy_list(seg.get("text")), float(w["text"]))
    add_terms("video_usage", truthy_list(seg.get("video_usage")), float(w["video_usage"]))

    return should


def choose_vector_fields(seg: Dict[str, Any], *, mode: str, primary: str) -> List[str]:
    """
    Pick KNN vector fields aligned with index mapping.
    - ``zero``: no vectors.
    - ``lite``: primary field only.
    - otherwise: up to 3 fields based on which segment lists are populated.
    """
    if mode == "zero":
        return []
    if mode == "lite":
        return [primary]

    def has_list(k: str) -> bool:
        return bool(truthy_list(seg.get(k)))

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
    return out[:3]


def build_road_run_fallback_query_body(
    qb: QueryBuilder,
    *,
    q: str,
    text_fields: List[str],
    seg_dur: float,
    used_ids: List[str],
    use_duration_score: bool,
) -> Dict[str, Any]:
    """Build the OpenSearch query body for the generic road-run fallback step."""
    fb_body = qb.build_bm25_only_search(
        CarInteriorAnalysisV2,
        q or "路跑 外观 行驶",
        size=int(_FILL_ROAD_RUN_FALLBACK_SIZE),
        search_fields=text_fields,
    )
    inner_q = fb_body.get("query") or {"match_all": {}}
    bool_inner: Dict[str, Any] = {
        "must": [
            inner_q,
            {"term": {"generic_hq_road_run": True}},
        ],
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


# ---------------------------------------------------------------------------
# Field-aligned BM25 query building
# ---------------------------------------------------------------------------
#
# Why: the current multi_match approach concatenates everything into one big
# string and fans it out across all index fields — correct but noisy.  IDF for
# "ERNC主动降噪" computed inside `marketing_phrases` (high-specificity field)
# is very different from IDF computed inside `description` (broad narrative).
# Field-aligned BM25 mirrors the KNN approach: each segment field queries the
# semantically equivalent index field, so scores are computed in the right
# vocabulary context.
#
# Keyword/enum fields (shot_style, weather, movement, …) are ALREADY handled
# by build_should_boosts → no BM25 clause needed for them here.
# ---------------------------------------------------------------------------

# (seg_field, index_field, bm25_boost)
# Ordered roughly by signal quality.
# ``extra_tags`` is NOT listed here — it is a catch-all field that should fan
# out to ALL BM25 text channels rather than being pinned to one (see
# ``_FIELD_ALIGN_BROADCAST`` below).
_FIELD_ALIGN_TEXT_MAP: List[Tuple[str, str, float]] = [
    ("marketing_phrases",       "marketing_phrases",       1.5),
    ("function_selling_points", "function_selling_points", 1.3),
    ("design_selling_points",   "design_selling_points",   1.3),
    ("description",             "description",             1.2),
    ("scene_location",          "scene_location",          1.1),
    ("scenario_a",              "scenario_a",              1.1),
    ("scenario_b",              "scenario_b",              1.0),
    # Looser cross-mappings (segment field → nearest index field)
    ("segment_text",            "description",             1.0),
    ("subject",                 "subject",                 1.0),
    ("object",                  "object",                  0.9),
    ("marketing_tags",          "marketing_phrases",       0.8),
]

# Fields that should broadcast to ALL BM25 text index fields via multi_match.
# This lets catch-all signals (extra_tags, future wildcard fields) surface
# wherever they have the highest TF-IDF relevance without pre-committing to a
# single channel.
_FIELD_ALIGN_BROADCAST: List[Tuple[str, float]] = [
    ("extra_tags", 0.7),
]

# The BM25 text fields in the index that broadcast clauses fan out to.
_BROADCAST_INDEX_TEXT_FIELDS: List[str] = [
    "marketing_phrases",
    "function_selling_points",
    "design_selling_points",
    "description",
    "scene_location",
    "scenario_a",
    "scenario_b",
    "subject",
    "object",
]


def build_field_aligned_bm25_query(
    seg: Dict[str, Any],
    *,
    size: int,
) -> Dict[str, Any]:
    """
    Build a BM25-only query body where each populated segment field is queried
    against the matching index field (``bool.should`` of per-field ``match``
    clauses, ``minimum_should_match=1``).

    Two phases:

    **Phase 1 — aligned**: each segment field → its semantically equivalent
    index field.  ``_text_for_bm25`` strips "未知" tokens so two documents that
    both have ``scenario_a = ["未知"]`` do NOT score as a match.

    **Phase 2 — broadcast**: catch-all segment fields (``extra_tags``) are
    fanned out to ALL BM25 text index fields via ``multi_match / best_fields``.
    This lets a tag like "山路" in ``extra_tags`` surface wherever it has the
    highest TF-IDF relevance (``scene_location``, ``scenario_a``, ``marketing_phrases``…)
    without pre-committing to a single channel.

    Returns a full OpenSearch search body (``size``, ``query``, ``_source``).
    """
    should_clauses: List[Dict[str, Any]] = []

    # --- Phase 1: per-field aligned match ---
    for seg_field, index_field, boost in _FIELD_ALIGN_TEXT_MAP:
        text = _text_for_bm25(seg.get(seg_field))
        if not text:
            continue
        should_clauses.append(
            {"match": {index_field: {"query": text, "boost": float(boost)}}}
        )

    # --- Phase 2: broadcast catch-all fields → multi_match across all text fields ---
    for seg_field, boost in _FIELD_ALIGN_BROADCAST:
        text = _text_for_bm25(seg.get(seg_field))
        if not text:
            continue
        should_clauses.append(
            {
                "multi_match": {
                    "query": text,
                    "fields": _BROADCAST_INDEX_TEXT_FIELDS,
                    "type": "best_fields",
                    "boost": float(boost),
                }
            }
        )

    query: Dict[str, Any] = (
        {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        if should_clauses
        else {"match_all": {}}
    )

    return {
        "size": size,
        "query": query,
        "_source": {"excludes": _ALL_VECTOR_FIELDS},
    }


def build_field_aligned_hybrid_query(
    seg: Dict[str, Any],
    qb: QueryBuilder,
    *,
    size: int,
    vector_fields: List[str],
) -> Dict[str, Any]:
    """
    Build a hybrid query body where:
    - BM25 sub-query: ``build_field_aligned_bm25_query`` (field-aligned
      ``bool.should``) — replaces the single ``multi_match`` sub-query.
    - KNN sub-queries: one per entry in *vector_fields*, each using the full
      ``segment_query_text`` embedding (existing behaviour, already field-selected
      by ``choose_vector_fields``).

    Pipeline weight management (``ensure_hybrid_pipeline``) is unchanged;
    the BM25 sub-query is still position-0 so the 0.3/0.7 split applies.
    """
    bm25_body = build_field_aligned_bm25_query(seg, size=size)
    bm25_subq = bm25_body["query"]

    q = segment_query_text(seg)
    query_vector = qb._generate_embedding(q)

    knn_subqs: List[Dict[str, Any]] = []
    for vf in vector_fields:
        if vf not in _ALL_VECTOR_FIELDS:
            continue
        weight = float(_VECTOR_WEIGHTS.get(vf, 1.0))
        knn_subqs.append(
            {"knn": {vf: {"vector": query_vector, "k": size, "boost": weight}}}
        )

    return {
        "size": size,
        "query": {"hybrid": {"queries": [bm25_subq] + knn_subqs}},
        "_source": {"excludes": _ALL_VECTOR_FIELDS},
    }

