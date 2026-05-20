# routers/video_analysis.py — 视频分析路由
# 端点:
#   POST /video-analysis           — 上传视频并执行分析流水线, 返回分镜卡片列表
#   GET  /video-analysis/history   — 获取所有历史分析记录
#   POST /video-analysis/history   — 覆盖写入全部历史记录
#   POST /video-analysis/history/update — 追加/更新单条历史记录

import asyncio
import functools
import hashlib
import os
import sys
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_material_match import VideoMaterialMatchHistory

from models.pydantic.video_analysis_request import (
    HistorySaveRequest,
    HistoryUpdateRequest,
    VideoAnalysisHistoryItem,
    ShotCard as PydShotCard,
)
from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextIndexTagsEnvelope
from services.analysis_video import analyze_video, index_shotcards_to_opensearch
from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags
from utils.frame_orientation import infer_frame_orientation
from services.video_analysis_db_service import video_analysis_db_service
from services.video_match_service import _load_strategy_by_name
from services.http_request_trace_service import http_request_trace_service
from infra.logging.logger import logger as log
from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import query_builder
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import (
    get_index_name, get_vector_fields, get_searchable_fields, get_field_weights, get_vector_weights,
)
from services.script_match_recall import ensure_hybrid_pipeline, ensure_rrf_pipeline
from services.video_match_http_trace import trace_response_top_hits_with_explain
from services.token_join_template_service import (
    TOKEN_JOIN_TERM_FIELDS_V2,
    normalize_v2_term_filter_value,
    create_template,
    delete_template,
    get_default_and_fields,
    list_templates,
    set_default_template,
    update_template,
)
from utils.search_utils import reciprocal_rank_fuse
from core.workspace import list_workspaces, DEFAULT_WORKSPACE_KEY, get_workspace


def _va_ordered_cards_to_trace_hits(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """与分镜素材检索 trace 同源字段，供 trace_response_top_hits_with_explain 写入 HTTP 记录。"""
    out: List[Dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        hid = c.get("history_id")
        sid = c.get("scene_id")
        doc_id: Optional[str] = None
        if hid is not None and sid is not None:
            try:
                doc_id = f"{hid}_{int(sid)}"
            except (TypeError, ValueError):
                doc_id = f"{hid}_{sid}"
        vp = str(c.get("obs_video_url") or c.get("thumbnail") or c.get("video_url") or "").strip()
        out.append(
            {
                "_id": doc_id or c.get("_id"),
                "_score": c.get("_score"),
                "history_id": hid,
                "video_path": vp[:512] if vp else None,
                "_explanation": c.get("_explanation"),
            }
        )
    return out

search_router = APIRouter()

class VideoAnalysisSearchToken(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    join: Optional[str] = "AND"
    not_: bool = Field(False, alias="not")
    type: Optional[str] = "keyword"
    source_field: Optional[str] = Field(
        default=None,
        description="对应 segment / v2 索引 keyword 字段名；AND 时用于 term filter",
    )

class VideoAnalysisSearchRequest(BaseModel):
    tokens: List[VideoAnalysisSearchToken] = Field(default_factory=list)
    strategy_name: Optional[str] = Field(default=None, description="搜索策略名称。若提供，则以后端策略为主")
    fuzzy: bool = False
    enable_road_run_fallback: bool = Field(default=True, description="开启路跑兜底")
    history_id: Optional[str] = Field(
        default=None,
        description="已忽略：搜索不按历史收窄，仅在 workspace 对应索引内全量检索；保留字段仅为兼容旧客户端。",
    )
    size: int = 50
    workspace: Optional[str] = None  # "v1" | "v2"
    bm25_weight: float = Field(default=0.3, description="BM25 搜索权重")
    vector_weight: float = Field(default=0.7, description="向量搜索权重")
    text_weights: Optional[dict] = Field(default=None, description="文本字段权重")
    vector_weights: Optional[dict] = Field(default=None, description="向量字段权重")
    use_rrf: bool = Field(default=False, description="模糊检索时使用 RRF（关闭则用 min_max+加权平均融合）")


def _va_tokens_for_storage(tokens: List[VideoAnalysisSearchToken]) -> List[Dict[str, Any]]:
    """完整入参 token 列表，供 material history / HTTP 审计还原标签（含 join、not、type、source_field）。"""
    out: List[Dict[str, Any]] = []
    for t in tokens:
        text = (t.text or "").strip()
        if not text:
            continue
        join_raw = (t.join or "AND").strip().upper()
        join = join_raw if join_raw in ("AND", "OR") else "AND"
        d: Dict[str, Any] = {
            "text": text,
            "join": join,
            "not": bool(t.not_),
            "type": (t.type or "keyword").strip() or "keyword",
        }
        sf = (t.source_field or "").strip()
        if sf:
            d["source_field"] = sf
        out.append(d)
    return out


def _video_analysis_split_tokens(
    raw_tokens: List[VideoAnalysisSearchToken],
    *,
    index_is_v2: bool,
) -> tuple[str, List[dict], List[str]]:
    allowed = TOKEN_JOIN_TERM_FIELDS_V2 if index_is_v2 else frozenset()
    term_filters: List[dict] = []
    rel_parts: List[str] = []
    must_not_texts: List[str] = []

    for tok in raw_tokens or []:
        text = (tok.text or "").strip()
        if not text or text == "未知":
            continue
        if tok.not_:
            must_not_texts.append(text)
            continue
        join = (tok.join or "AND").strip().upper()
        is_and = join == "AND"
        sf = (tok.source_field or "").strip()
        if is_and and sf and sf in allowed:
            term_val = normalize_v2_term_filter_value(sf, text)
            term_filters.append({"term": {sf: term_val}})
        else:
            rel_parts.append(text)

    query_text = " ".join(rel_parts).strip()
    if not query_text:
        query_text = "素材"
    return query_text, term_filters, must_not_texts


def _must_not_clauses_from_texts(texts: List[str], weighted_fields: List[str]) -> List[dict]:
    out: List[dict] = []
    for t in texts:
        tt = (t or "").strip()
        if not tt:
            continue
        out.append(
            {"multi_match": {"query": tt, "fields": weighted_fields, "type": "best_fields"}}
        )
    return out


def _build_fallback_filter_block(
    term_filters: List[dict],
    history_prefix: Optional[str],
    enable_road_run_fallback: bool
) -> List[dict]:
    filt = []
    if history_prefix:
        filt.append({"prefix": {"id": history_prefix}})

    if not term_filters:
        return filt

    if not enable_road_run_fallback:
        filt.extend(term_filters)
        return filt

    fallback_block = {
        "bool": {
            "minimum_should_match": 1,
            "should": [
                {
                    "bool": {
                        "filter": list(term_filters),
                        "_name": "strict_match"
                    }
                },
                {
                    "bool": {
                        "filter": [{"term": {"generic_hq_road_run": True}}],
                        "_name": "road_run_fallback"
                    }
                }
            ]
        }
    }
    filt.append(fallback_block)
    return filt


def _build_fallback_should_boosts(term_filters: List[dict], enable_road_run_fallback: bool) -> List[dict]:
    if not enable_road_run_fallback or not term_filters:
        return []
    should_boosts = []
    for tf in term_filters:
        if "term" in tf:
            for field, val in tf["term"].items():
                should_boosts.append({"term": {field: {"value": val, "boost": 1.1}}})
    return should_boosts


def _wrap_bool_query(
    inner: dict,
    *,
    history_prefix: Optional[str],
    term_filters: List[dict],
    must_not: Optional[List[dict]] = None,
    enable_road_run_fallback: bool = False,
) -> dict:
    filt = _build_fallback_filter_block(term_filters, history_prefix, enable_road_run_fallback)
    should = _build_fallback_should_boosts(term_filters, enable_road_run_fallback)
    
    if not filt and not must_not and not should:
        return inner
        
    b: dict = {"must": [inner]}
    if filt:
        b["filter"] = filt
    if must_not:
        b["must_not"] = must_not
    if should:
        b["should"] = should
    return {"bool": b}


def _wrap_hybrid_query_with_filters(
    hybrid_query: dict,
    *,
    history_prefix: Optional[str],
    term_filters: List[dict],
    must_not: Optional[List[dict]] = None,
    enable_road_run_fallback: bool = False,
) -> dict:
    """
    OpenSearch 要求 ``hybrid`` 为顶层 query，不能包在 ``bool.must`` 里。
    将 filter / must_not 下推到 hybrid 的每个子查询外层的 ``bool``（与 script_match 一致）。
    """
    filt = _build_fallback_filter_block(term_filters, history_prefix, enable_road_run_fallback)
    should = _build_fallback_should_boosts(term_filters, enable_road_run_fallback)

    hy = hybrid_query.get("hybrid") if isinstance(hybrid_query, dict) else None
    if not isinstance(hy, dict):
        return hybrid_query
    subqs = hy.get("queries") or []
    wrapped: List[dict] = []
    for subq in subqs:
        if not isinstance(subq, dict):
            continue
        if not filt and not must_not and not should:
            wrapped.append(subq)
            continue
        b: Dict[str, Any] = {"must": [subq]}
        if filt:
            b["filter"] = filt
        if must_not:
            b["must_not"] = must_not
        if should:
            b["should"] = should
        wrapped.append({"bool": b})
    return {"hybrid": {"queries": wrapped}}

def _parse_doc_id(doc_id: str) -> Optional[tuple[str, int]]:
    """
    doc_id format: "{history_id}_{scene_id}"
    """
    try:
        if not doc_id:
            return None
        parts = doc_id.rsplit("_", 1)
        if len(parts) != 2:
            return None
        return parts[0], int(parts[1])
    except Exception:
        return None


HYBRID_MAX_KNN = 4
RRF_RANK_CONSTANT = 60


async def _video_analysis_client_rrf_hits(
    client: Any,
    *,
    index_name: str,
    IndexModel: type,
    query_text: str,
    q_vec: List[float],
    ordered_vec_fields: List[str],
    vec_weight_map: dict,
    text_weights: Optional[dict],
    size: int,
    hist_prefix_opt: Optional[str],
    extra_term_filters: Optional[List[dict]] = None,
    must_not_multi_matches: Optional[List[dict]] = None,
) -> List[dict]:
    """
    When active KNN routes exceed OpenSearch hybrid cap, run BM25 + one KNN search per field
    via ``_msearch`` and merge with weighted RRF in-process.
    """
    import json

    recall = min(500, max(size * 5, 100))

    text_fields = get_searchable_fields(IndexModel)
    weights = get_field_weights(IndexModel).copy()
    if text_weights:
        weights.update(text_weights)
    weighted_fields = [f"{f}^{weights.get(f, 1.0)}" for f in text_fields]

    def wrap_clause(inner: dict) -> dict:
        return _wrap_bool_query(
            inner,
            history_prefix=hist_prefix_opt,
            term_filters=list(extra_term_filters or []),
            must_not=must_not_multi_matches if must_not_multi_matches else None,
        )

    mm = {
        "multi_match": {
            "query": query_text,
            "fields": weighted_fields,
            "type": "best_fields",
            "_name": "bm25_text_match",
        }
    }
    bm25_query = wrap_clause(mm)
    bm25_body = {"size": recall, "query": bm25_query, "_source": False}

    knn_bodies: List[dict] = []
    for field in ordered_vec_fields:
        boost = float(vec_weight_map.get(field, 1.0))
        knn_clause = {
            "knn": {
                field: {
                    "vector": q_vec,
                    "k": recall,
                    "boost": boost,
                },
            },
        }
        knn_q = wrap_clause(knn_clause)
        knn_bodies.append({"size": recall, "query": knn_q, "_source": False})

    nd_parts: List[str] = []
    hdr = json.dumps({"index": index_name})
    for b in [bm25_body] + knn_bodies:
        nd_parts.append(hdr)
        nd_parts.append(json.dumps(b))
    nd_body = "\n".join(nd_parts) + "\n"

    resp = await client.msearch(body=nd_body)
    responses = resp.get("responses") or []
    ranked_lists: List[List[str]] = []
    for r in responses:
        hh = ((r or {}).get("hits") or {}).get("hits") or []
        ranked_lists.append([str(h.get("_id") or "") for h in hh if h.get("_id")])

    raw_w = [1.0] + [float(vec_weight_map.get(f, 1.0)) for f in ordered_vec_fields]
    ssum = sum(raw_w) or 1.0
    rrf_weights = [x / ssum for x in raw_w]

    fused = reciprocal_rank_fuse(
        ranked_lists,
        rrf_weights,
        rank_constant=RRF_RANK_CONSTANT,
        top_n=size,
    )
    return [{"_id": doc_id, "_score": sc, "matched_queries": []} for doc_id, sc in fused]


@search_router.post("/search")
async def search_cards(req: VideoAnalysisSearchRequest):
    """
    Search cards via OpenSearch (hybrid: keyword + vector).
    Returns full ShotCard payloads from DB (source of truth) ordered by OpenSearch score.
    精准匹配(fuzzy=False): BM25 only  /  模糊匹配(fuzzy=True): BM25 + KNN hybrid
    （可选 use_rrf：RRF 排名融合；宏观 bm25_weight/vector_weight 不参与，仅以字段级权重推导子路权重）

    AND + ``source_field``（v2 keyword 白名单）在 OpenSearch 中作 term filter；
    OR（及无 source_field 的 AND）进入 ``query_text`` 相关性；``not`` → must_not。

    检索范围：仅由 ``workspace`` 决定索引（v1/v2 模型）；**不按** ``history_id`` 收窄文档，
    即在当前 workspace 对应索引内全量匹配（与视频匹配页一致的全库召回语义）。
    """
    raw_tokens = [t for t in (req.tokens or []) if (t.text or "").strip()]
    if not raw_tokens:
        return {"success": True, "cards": []}

    size = max(1, min(int(req.size or 50), 200))
    ws = (req.workspace or "").strip() or "default"
    history_id = (req.history_id or "").strip()
    index_is_v2 = ws == "v2"

    if req.strategy_name:
        st = await _load_strategy_by_name(req.strategy_name)
        if st:
            req.bm25_weight = float(st.bm25_weight or 0.3)
            req.vector_weight = float(st.vector_weight or 0.7)
            req.text_weights = st.text_weights
            req.vector_weights = st.vector_weights
            req.use_rrf = bool(st.use_rrf)
            req.fuzzy = req.vector_weight > 0

    query_text, term_filters, must_not_texts = _video_analysis_split_tokens(
        raw_tokens, index_is_v2=index_is_v2
    )

    hist_prefix_opt: Optional[str] = None

    token_texts = [t.text for t in raw_tokens[:10]]
    log.info(
        f"[search] query={query_text!r} fuzzy={req.fuzzy} use_rrf={req.use_rrf} size={size} "
        f"workspace={ws!r} history_id_param={history_id or '*'} (ignored for scope) "
        f"term_filters={term_filters!r} tokens={token_texts!r}"
    )

    t_search0 = time.perf_counter()
    token_payload = _va_tokens_for_storage(raw_tokens)
    audit_body: Dict[str, Any] = {
        "workspace": ws,
        "fuzzy": req.fuzzy,
        "use_rrf": req.use_rrf,
        "size": size,
        "query_text": query_text,
        "token_count": len(raw_tokens),
        "tokens": token_payload,
        "history_id": history_id or None,
        "bm25_weight": req.bm25_weight,
        "vector_weight": req.vector_weight,
    }
    trace_rid = await http_request_trace_service.create_initial(
        request_url="/video-analysis/search",
        method_name="POST /video-analysis/search",
        business_type="VIDEO_ANALYSIS_CARD_SEARCH",
        request_body=audit_body,
    )
    match_hist_id = str(uuid.uuid4())

    async def _audit_va_search_finish(
        *,
        api_ok: bool,
        search_mode_final: str,
        cards_result: Optional[List[Dict[str, Any]]] = None,
        err_msg: Optional[str] = None,
    ) -> None:
        elapsed_ms = int((time.perf_counter() - t_search0) * 1000)
        cards = cards_result or []
        hit_n = len(cards)
        top1: Optional[str] = None
        if cards:
            c0 = cards[0]
            if isinstance(c0, dict):
                top1 = (
                    str(
                        c0.get("obs_video_url")
                        or c0.get("thumbnail")
                        or c0.get("video_url")
                        or ""
                    ).strip()[:2048]
                    or None
                )
        st_snap: Dict[str, Any] = {
            "bm25_weight": req.bm25_weight,
            "vector_weight": req.vector_weight,
            "use_rrf": req.use_rrf,
            "fuzzy": req.fuzzy,
            "text_weights": req.text_weights,
            "vector_weights": req.vector_weights,
            "search_tokens": token_payload,
        }
        trace_hit_payload: Any = None
        if api_ok and cards:
            trace_hit_payload = trace_response_top_hits_with_explain(
                _va_ordered_cards_to_trace_hits(cards)
            )
        resp_trace: Dict[str, Any] = {
            "hit_count": hit_n,
            "search_mode": search_mode_final,
            "success": api_ok,
            "match_history_id": match_hist_id,
        }
        if trace_hit_payload is not None:
            resp_trace["top_hits_explain"] = trace_hit_payload
        try:
            await http_request_trace_service.finalize(
                trace_rid,
                status_code=200 if api_ok else 500,
                response_body=resp_trace if api_ok else {"success": False, "error": err_msg},
                error_message=err_msg if not api_ok else None,
                business_success=api_ok,
                duration_ms=elapsed_ms,
            )
        except Exception as ex:
            log.warning("va search trace finalize failed: %s", ex)
        try:
            async with mysql_connector.session_scope() as session:
                hist = VideoMaterialMatchHistory(
                    id=match_hist_id,
                    request_id=trace_rid,
                    source="video_analysis_search",
                    workspace=ws if ws and ws != "default" else None,
                    status="done" if api_ok else "failed",
                    va_context_history_id=history_id or None,
                    query_preview=(query_text or "")[:512] or None,
                    search_mode=search_mode_final,
                    hit_count=hit_n,
                    top1_obs_url=top1,
                    elapsed_ms=float(elapsed_ms),
                    error_message=err_msg if not api_ok else None,
                    strategy_snapshot=st_snap,
                )
                session.add(hist)
                await session.commit()
        except Exception as ex:
            log.warning("va search material history persist failed: %s", ex)

    IndexModel = CarInteriorAnalysisV2 if index_is_v2 else CarInteriorAnalysis

    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()

    vec_fields = get_vector_fields(IndexModel)
    text_fields = get_searchable_fields(IndexModel)
    weights = get_field_weights(IndexModel).copy()
    if req.text_weights:
        weights.update(req.text_weights)
    weighted_fields = [f"{f}^{weights.get(f, 1.0)}" for f in text_fields]
    must_not_mm = _must_not_clauses_from_texts(must_not_texts, weighted_fields)
    must_not_opt = must_not_mm if must_not_mm else None

    body: Optional[dict] = None
    pipeline_param: Optional[str] = None
    hits: Optional[List[dict]] = None
    search_mode = "precise"

    if not req.fuzzy:
        body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": weighted_fields,
                    "type": "best_fields",
                    "_name": "bm25_text_match",
                }
            },
            "_source": {"excludes": vec_fields},
        }
        search_mode = "precise"
    else:
        vec_weight_map = get_vector_weights(IndexModel).copy()
        if req.vector_weights:
            vec_weight_map.update(req.vector_weights)

        active_vecs = [f for f in vec_fields if float(vec_weight_map.get(f, 1.0) or 0) > 0]
        ordered_vecs = sorted(
            active_vecs, key=lambda f: float(vec_weight_map.get(f, 1.0)), reverse=True
        )

        q_vec = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(query_builder._generate_embedding, query_text),
        )

        use_rrf = bool(req.use_rrf) and len(ordered_vecs) > 0

        if use_rrf and len(ordered_vecs) > HYBRID_MAX_KNN:
            try:
                hits = await _video_analysis_client_rrf_hits(
                    client,
                    index_name=get_index_name(IndexModel),
                    IndexModel=IndexModel,
                    query_text=query_text,
                    q_vec=q_vec,
                    ordered_vec_fields=ordered_vecs,
                    vec_weight_map=vec_weight_map,
                    text_weights=req.text_weights,
                    size=size,
                    hist_prefix_opt=hist_prefix_opt,
                    extra_term_filters=term_filters,
                    must_not_multi_matches=must_not_opt,
                )
                search_mode = "fuzzy_rrf"
            except Exception as e:
                log.error(f"video-analysis client RRF search failed: {e}")
                await _audit_va_search_finish(
                    api_ok=False, search_mode_final="fuzzy_rrf", err_msg=str(e)
                )
                return {
                    "success": False,
                    "error": str(e),
                    "cards": [],
                    "match_history_id": match_hist_id,
                }
        else:
            top_vecs = ordered_vecs if use_rrf else ordered_vecs[:HYBRID_MAX_KNN]
            body = query_builder.build_dynamic_hybrid_search(
                IndexModel,
                query_text,
                size=size,
                bm25_factor=1.0 if use_rrf else req.bm25_weight,
                vector_factor=1.0 if use_rrf else req.vector_weight,
                vector_fields=top_vecs,
                query_vector=q_vec,
                field_weight_overrides=req.text_weights,
                vector_weight_overrides=req.vector_weights,
            )
            num_q = 1 + len(top_vecs)
            if use_rrf:
                raw_w = [1.0] + [float(vec_weight_map.get(f, 1.0)) for f in top_vecs]
                ssum = sum(raw_w) or 1.0
                rrf_w = [x / ssum for x in raw_w]
                pipeline_param = await ensure_rrf_pipeline(
                    client,
                    base_name="video-analysis-rrf",
                    num_queries=num_q,
                    weights=rrf_w,
                )
                search_mode = "fuzzy_rrf"
            else:
                pipeline_param = await ensure_hybrid_pipeline(
                    client, pipeline_name="nlp-search-pipeline", num_queries=num_q
                )
                search_mode = "fuzzy"

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

    if body is not None:
        inner_q = body.get("query")
        if inner_q is not None:
            if isinstance(inner_q, dict) and "hybrid" in inner_q:
                body["query"] = _wrap_hybrid_query_with_filters(
                    inner_q,
                    history_prefix=hist_prefix_opt,
                    term_filters=term_filters,
                    must_not=must_not_opt,
                    enable_road_run_fallback=req.enable_road_run_fallback,
                )
            else:
                body["query"] = _wrap_bool_query(
                    inner_q,
                    history_prefix=hist_prefix_opt,
                    term_filters=term_filters,
                    must_not=must_not_opt,
                    enable_road_run_fallback=req.enable_road_run_fallback,
                )
        body["highlight"] = {
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "require_field_match": False,
            "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in _HIGHLIGHT_FIELDS},
        }
        body["explain"] = search_mode != "fuzzy_rrf"

    try:
        index_name = get_index_name(IndexModel)
        if hits is None:
            assert body is not None
            search_params = {"search_pipeline": pipeline_param} if pipeline_param else None
            resp = await client.search(index=index_name, body=body, params=search_params)
            hits = ((resp.get("hits") or {}).get("hits") or [])
    except Exception as e:
        log.error(f"video-analysis search failed: {e}")
        await _audit_va_search_finish(api_ok=False, search_mode_final=search_mode, err_msg=str(e))
        return {
            "success": False,
            "error": str(e),
            "cards": [],
            "match_history_id": match_hist_id,
        }

    assert hits is not None
    top5 = [
        (h.get("_id"), round(float(h.get("_score") or 0), 4), h.get("matched_queries", []))
        for h in hits[:5]
    ]
    log.info(f"[search] hits={len(hits)}  mode={search_mode}  top5={top5}")

    keys_in_order: List[tuple[str, int]] = []
    meta_by_key: dict[tuple[str, int], dict] = {}
    for h in hits:
        doc_id = h.get("_id") or (h.get("_source") or {}).get("id")
        k = _parse_doc_id(str(doc_id)) if doc_id else None
        if k:
            keys_in_order.append(k)
            meta = {}
            raw_score = h.get("_score")
            if raw_score is not None:
                meta["_score"] = float(raw_score)
            raw_hl = h.get("highlight")
            if raw_hl:
                meta["_highlight"] = raw_hl
            matched_queries = h.get("matched_queries")
            if matched_queries:
                meta["_matched_queries"] = matched_queries
                if "road_run_fallback" in matched_queries and "strict_match" not in matched_queries:
                    meta["is_fallback"] = True
            explanation = h.get("_explanation")
            if explanation:
                meta["_explanation"] = explanation
            meta_by_key[k] = meta

    if not keys_in_order:
        await _audit_va_search_finish(api_ok=True, search_mode_final=search_mode, cards_result=[])
        return {
            "success": True,
            "cards": [],
            "search_mode": search_mode,
            "match_history_id": match_hist_id,
        }

    shot_ver = "v2" if index_is_v2 else "v1"
    cards = await video_analysis_db_service.get_cards_by_keys(keys_in_order, shot_cards_version=shot_ver)
    by_key = {(c.get("history_id"), int(c.get("scene_id") or 0)): c for c in (cards or [])}
    ordered = []
    for k in keys_in_order:
        if k not in by_key:
            continue
        card = dict(by_key[k])
        card.update(meta_by_key.get(k) or {})
        ordered.append(card)
    await _audit_va_search_finish(api_ok=True, search_mode_final=search_mode, cards_result=ordered)
    return {
        "success": True,
        "cards": ordered,
        "search_mode": search_mode,
        "match_history_id": match_hist_id,
    }

class VideoAnalysisReindexRequest(BaseModel):
    history_id: str
    scene_ids: List[int] = Field(default_factory=list)
    refresh: bool = False

@search_router.post("/reindex")
async def reindex_cards(req: VideoAnalysisReindexRequest):
    """
    Reindex selected cards into OpenSearch, then update MySQL os_index_status.
    """
    history_id = (req.history_id or "").strip()
    scene_ids = [int(x) for x in (req.scene_ids or [])]
    if not history_id or not scene_ids:
        return {"success": False, "error": "history_id / scene_ids required"}

    keys = [(history_id, sid) for sid in scene_ids]

    # mark as pending first (best effort)
    try:
        await video_analysis_db_service.update_cards_index_status(
            keys, status="PENDING", error=None, shot_cards_version="v2"
        )
    except Exception as e:
        log.warning(f"reindex: failed to mark PENDING: {e}")

    rows = await video_analysis_db_service.get_cards_by_keys(keys, shot_cards_version="v2")
    if not rows:
        await video_analysis_db_service.update_cards_index_status(
            keys, status="FAILED", error="cards not found in db", shot_cards_version="v2"
        )
        return {"success": False, "error": "cards not found", "updated": []}

    cards: List[PydShotCard] = []
    ok_keys: List[tuple[str, int]] = []
    skipped: List[tuple[str, int]] = []

    for r in rows:
        k = (r.get("history_id") or history_id, int(r.get("scene_id") or 0))
        if r.get("error"):
            skipped.append(k)
            continue
        try:
            cards.append(PydShotCard(**r))
            ok_keys.append(k)
        except Exception as e:
            skipped.append(k)
            log.warning(f"reindex: parse ShotCard failed for {k}: {e}")

    if not cards:
        await video_analysis_db_service.update_cards_index_status(
            keys, status="FAILED", error="no valid cards to reindex", shot_cards_version="v2"
        )
        return {"success": False, "error": "no valid cards", "updated": []}

    try:
        await index_shotcards_to_opensearch(cards, id_prefix=history_id, refresh=bool(req.refresh), workspace="v2")
        await video_analysis_db_service.update_cards_index_status(
            ok_keys, status="OK", error=None, shot_cards_version="v2"
        )
    except Exception as e:
        await video_analysis_db_service.update_cards_index_status(
            ok_keys, status="FAILED", error=str(e), shot_cards_version="v2"
        )
        return {"success": False, "error": str(e), "updated": []}

    updated_rows = await video_analysis_db_service.get_cards_by_keys(ok_keys, shot_cards_version="v2")
    return {
        "success": True,
        "updated": updated_rows,
        "skipped": [{"history_id": k[0], "scene_id": k[1]} for k in skipped],
    }


