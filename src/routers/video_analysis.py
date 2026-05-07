# routers/video_analysis.py — 视频分析路由
# 端点:
#   POST /video-analysis           — 上传视频并执行分析流水线, 返回分镜卡片列表
#   GET  /video-analysis/history   — 获取所有历史分析记录
#   POST /video-analysis/history   — 覆盖写入全部历史记录
#   POST /video-analysis/history/update — 追加/更新单条历史记录

import asyncio
import functools
import os
import sys
import uuid
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List

from models.pydantic.video_analysis_request import (
    HistorySaveRequest,
    HistoryUpdateRequest,
    VideoAnalysisHistoryItem,
    ShotCard as PydShotCard,
)
from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextIndexTagsEnvelope
from services.analysis_video import analyze_video, index_shotcards_to_opensearch
from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags
from services.video_analysis_db_service import video_analysis_db_service
from infra.logging.logger import logger as log
from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import query_builder
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import (
    get_index_name, get_vector_fields, get_searchable_fields, get_field_weights, get_vector_weights,
)
from services.script_match_recall import ensure_hybrid_pipeline
from core.workspace import list_workspaces, DEFAULT_WORKSPACE_KEY


video_analysis_router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])

UPLOAD_TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "video_analysis_uploads",
)


# ---------------- Workspace 接口 ----------------

@video_analysis_router.get("/workspaces")
async def get_workspaces():
    """返回当前支持的 workspace 列表（key / label / description）及默认值。"""
    return {
        "success": True,
        "workspaces": list_workspaces(),
        "default": DEFAULT_WORKSPACE_KEY,
    }


# ---------------- 搜索策略接口 ----------------

@video_analysis_router.get("/index-fields")
async def get_index_fields(workspace: str = Query("v2")):
    """获取指定 workspace 下索引的可用字段，用于前端动态生成权重调节滑块"""
    IndexModel = CarInteriorAnalysisV2 if workspace == "v2" else CarInteriorAnalysis
    text_fields = get_searchable_fields(IndexModel)
    vector_fields = get_vector_fields(IndexModel)
    return {
        "success": True,
        "text_fields": text_fields,
        "vector_fields": vector_fields
    }

class SearchStrategyCreate(BaseModel):
    name: str
    bm25_weight: float
    vector_weight: float
    text_weights: Optional[dict] = None
    vector_weights: Optional[dict] = None
    is_default: bool = False

@video_analysis_router.get("/search-strategies")
async def list_search_strategies():
    """获取所有搜索策略配置"""
    from infra.storage.mysql_connector import mysql_connector
    from sqlmodel import select
    from models.sqlmodel.video_analysis import VideoAnalysisSearchStrategy
    
    await mysql_connector.ensure_init()
    engine = await mysql_connector.get_engine()
    
    # Use async generator correctly
    async with mysql_connector.client() as conn:
        stmt = select(VideoAnalysisSearchStrategy).order_by(VideoAnalysisSearchStrategy.id)
        res = await conn.execute(stmt)
        strategies = res.scalars().all()
        
    return {"success": True, "strategies": [s.model_dump() for s in strategies]}

@video_analysis_router.post("/search-strategies")
async def save_search_strategy(req: SearchStrategyCreate):
    """保存或更新搜索策略配置"""
    from infra.storage.mysql_connector import mysql_connector
    from sqlmodel import select
    from models.sqlmodel.video_analysis import VideoAnalysisSearchStrategy
    from sqlalchemy import update
    
    await mysql_connector.ensure_init()
    
    async with mysql_connector.client() as conn:
        try:
            # 如果设为默认，先把其他的取消默认
            if req.is_default:
                await conn.execute(
                    update(VideoAnalysisSearchStrategy).values(is_default=False)
                )
                
            # 检查是否已存在同名策略
            stmt = select(VideoAnalysisSearchStrategy).where(VideoAnalysisSearchStrategy.name == req.name)
            res = await conn.execute(stmt)
            existing = res.scalars().first()
            
            if existing:
                await conn.execute(
                    update(VideoAnalysisSearchStrategy)
                    .where(VideoAnalysisSearchStrategy.id == existing.id)
                    .values(
                        bm25_weight=req.bm25_weight,
                        vector_weight=req.vector_weight,
                        text_weights=req.text_weights,
                        vector_weights=req.vector_weights,
                        is_default=req.is_default
                    )
                )
            else:
                new_strategy = VideoAnalysisSearchStrategy(
                    name=req.name,
                    bm25_weight=req.bm25_weight,
                    vector_weight=req.vector_weight,
                    text_weights=req.text_weights,
                    vector_weights=req.vector_weights,
                    is_default=req.is_default
                )
                conn.add(new_strategy)
                
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise e
            
    return {"success": True}

@video_analysis_router.delete("/search-strategies/{strategy_id}")
async def delete_search_strategy(strategy_id: int):
    """删除搜索策略"""
    from infra.storage.mysql_connector import mysql_connector
    from sqlalchemy import delete
    from models.sqlmodel.video_analysis import VideoAnalysisSearchStrategy
    
    await mysql_connector.ensure_init()
    async with mysql_connector.client() as conn:
        try:
            await conn.execute(
                delete(VideoAnalysisSearchStrategy).where(VideoAnalysisSearchStrategy.id == strategy_id)
            )
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise e
            
    return {"success": True}


# ---------------- 历史记录接口 ----------------

@video_analysis_router.get("/history")
async def get_history(workspace: Optional[str] = Query(None, description="工作区标识，如 v1 / v2")):
    history = await video_analysis_db_service.list_history(workspace=workspace)
    return {"success": True, "history": history}


@video_analysis_router.get("/history/{history_id}")
async def get_history_item(
    history_id: str,
    shot_cards_version: str = Query("v1", description="分镜表版本：v1 旧卡片 / v2 对齐 SceneAnalysisResultV2"),
):
    ver = shot_cards_version if shot_cards_version in ("v1", "v2") else "v1"
    item = await video_analysis_db_service.get_history_item(history_id, shot_cards_version=ver)
    if item is None:
        raise HTTPException(status_code=404, detail="history not found")
    return {"success": True, "item": item}


@video_analysis_router.get("/cards")
async def get_cards(
    history_id: Optional[str] = None,
    shot_cards_version: str = Query("v1", description="分镜表版本：v1 / v2"),
    workspace: Optional[str] = Query(None, description="工作区标识，如 v1 / v2"),
):
    """
    Get cards by history_id.
    - history_id=__all__ or missing => all cards across histories
    - else => cards of one history (from DB)
    """
    ver = shot_cards_version if shot_cards_version in ("v1", "v2") else "v1"
    if not history_id or history_id == "__all__":
        cards = await video_analysis_db_service.list_all_cards(shot_cards_version=ver, workspace=workspace)
        return {"success": True, "cards": cards, "shot_cards_version": ver}

    item = await video_analysis_db_service.get_history_item(history_id, shot_cards_version=ver)
    if item is None:
        raise HTTPException(status_code=404, detail="history not found")
    return {"success": True, "cards": item.get("cards", []), "shot_cards_version": ver}

class VideoAnalysisSearchToken(BaseModel):
    text: str
    join: Optional[str] = "AND"
    not_: bool = Field(False, alias="not")
    type: Optional[str] = "keyword"

class VideoAnalysisSearchRequest(BaseModel):
    tokens: List[VideoAnalysisSearchToken] = Field(default_factory=list)
    fuzzy: bool = False
    history_id: Optional[str] = None
    size: int = 50
    workspace: Optional[str] = None  # "v1" | "v2"
    bm25_weight: float = Field(default=0.3, description="BM25 搜索权重")
    vector_weight: float = Field(default=0.7, description="向量搜索权重")
    text_weights: Optional[dict] = Field(default=None, description="文本字段权重")
    vector_weights: Optional[dict] = Field(default=None, description="向量字段权重")

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

@video_analysis_router.post("/search")
async def search_cards(req: VideoAnalysisSearchRequest):
    """
    Search cards via OpenSearch (hybrid: keyword + vector).
    Returns full ShotCard payloads from DB (source of truth) ordered by OpenSearch score.
    精准匹配(fuzzy=False): BM25 only  /  模糊匹配(fuzzy=True): BM25 + KNN hybrid
    """
    tokens = [t.text.strip() for t in (req.tokens or []) if t.text and t.text.strip()]
    if not tokens:
        return {"success": True, "cards": []}

    query_text = " ".join(tokens)
    size = max(1, min(int(req.size or 50), 200))
    mode = "fuzzy(BM25+KNN)" if req.fuzzy else "precise(BM25)"
    ws = (req.workspace or "").strip() or "default"
    history_id = (req.history_id or "").strip()

    token_texts = [t.text for t in (req.tokens or [])[:10]]
    log.info(
        f"[search] query={query_text!r}  mode={mode}  size={size}"
        f"  workspace={ws}  history={history_id or '*'}  tokens={token_texts}"
    )

    # Workspace-aware index model: v2 uses CarInteriorAnalysisV2, else v1
    IndexModel = CarInteriorAnalysisV2 if ws == "v2" else CarInteriorAnalysis

    # ── Query body ────────────────────────────────────────────────────────────
    # Precise mode: plain multi_match BM25 — no hybrid query type, no pipeline needed.
    #   Scores are standard Lucene BM25 (always positive, comparable across docs).
    # Fuzzy mode: hybrid (BM25 + KNN) with a normalization pipeline.
    #   The `hybrid` query type REQUIRES search_pipeline; without it OpenSearch returns
    #   raw internal scores that can be huge negatives (known cluster bug).
    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()

    vec_fields = get_vector_fields(IndexModel)
    pipeline_param: Optional[str] = None

    if not req.fuzzy:
        # ── Precise: pure BM25 multi_match ────────────────────────────────
        text_fields = get_searchable_fields(IndexModel)
        weights = get_field_weights(IndexModel)
        if req.text_weights:
            weights.update(req.text_weights)
        weighted_fields = [f"{f}^{weights.get(f, 1.0)}" for f in text_fields]
        body: dict = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": weighted_fields,
                    "type": "best_fields",
                    "_name": "bm25_text_match"
                }
            },
            "_source": {"excludes": vec_fields},
        }
    else:
        # ── Fuzzy: hybrid BM25 + KNN with normalization pipeline ──────────
        # OpenSearch hybrid query has a hard sub-query cap (typically 5).
        # v2 has 7 vector fields → 1 BM25 + 7 KNN = 8, which exceeds the limit.
        # Sort by marker weight and keep only the top 2 KNN paths (total = 3).
        vec_weight_map = get_vector_weights(IndexModel)
        top_vecs = sorted(vec_fields, key=lambda f: vec_weight_map.get(f, 1.0), reverse=True)[:2]
        # Embedding 在默认线程池执行，避免阻塞 asyncio 事件循环（否则 /health 等接口卡顿）
        q_vec = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(query_builder._generate_embedding, query_text),
        )
        body = query_builder.build_dynamic_hybrid_search(
            IndexModel, query_text, size=size, 
            bm25_factor=req.bm25_weight, 
            vector_factor=req.vector_weight,
            vector_fields=top_vecs,
            query_vector=q_vec,
            field_weight_overrides=req.text_weights,
            vector_weight_overrides=req.vector_weights,
        )
        # num_queries = 1 (multi_match) + len(top_vecs)
        num_q = 1 + len(top_vecs)
        pipeline_param = await ensure_hybrid_pipeline(
            client, pipeline_name="nlp-search-pipeline", num_queries=num_q
        )

    # Optional history filter
    if history_id and history_id != "__all__":
        prefix = f"{history_id}_"
        q = body.get("query") or {}
        body["query"] = {"bool": {"must": [q], "filter": [{"prefix": {"id": prefix}}]}}

    # Highlight matched terms in text fields → "命中路径" drawer section
    _HIGHLIGHT_FIELDS = [
        "description", "subject", "object",
        "design_selling_points", "function_selling_points",
        "scenario_a", "scenario_b",
        "marketing_phrases", "appealing_audience", "scene_location",
    ]
    body["highlight"] = {
        "pre_tags": ["<em>"],
        "post_tags": ["</em>"],
        "require_field_match": False,
        "fields": {f: {"number_of_fragments": 2, "fragment_size": 80} for f in _HIGHLIGHT_FIELDS},
    }

    try:
        index_name = get_index_name(IndexModel)
        search_params = {"search_pipeline": pipeline_param} if pipeline_param else None
        resp = await client.search(index=index_name, body=body, params=search_params)
        hits = ((resp.get("hits") or {}).get("hits") or [])
    except Exception as e:
        log.error(f"video-analysis search failed: {e}")
        return {"success": False, "error": str(e), "cards": []}

    top5 = [(h.get("_id"), round(float(h.get("_score") or 0), 4), h.get("matched_queries", [])) for h in hits[:5]]
    log.info(f"[search] hits={len(hits)}  top5={top5}")

    # Map OpenSearch doc ids -> (history_id, scene_id) + capture per-doc score
    keys_in_order: List[tuple[str, int]] = []
    meta_by_key: dict[tuple[str, int], dict] = {}
    for h in hits:
        doc_id = h.get("_id") or (h.get("_source") or {}).get("id")
        k = _parse_doc_id(str(doc_id)) if doc_id else None
        if k:
            keys_in_order.append(k)
            meta: dict = {}
            raw_score = h.get("_score")
            if raw_score is not None:
                meta["_score"] = float(raw_score)
            raw_hl = h.get("highlight")
            if raw_hl:
                meta["_highlight"] = raw_hl
            matched_queries = h.get("matched_queries")
            if matched_queries:
                meta["_matched_queries"] = matched_queries
            meta_by_key[k] = meta

    if not keys_in_order:
        return {"success": True, "cards": []}

    cards = await video_analysis_db_service.get_cards_by_keys(keys_in_order, shot_cards_version="v2")
    by_key = {(c.get("history_id"), int(c.get("scene_id") or 0)): c for c in (cards or [])}
    ordered = []
    for k in keys_in_order:
        if k not in by_key:
            continue
        card = dict(by_key[k])
        card.update(meta_by_key.get(k) or {})
        ordered.append(card)
    return {"success": True, "cards": ordered, "search_mode": mode}

class VideoAnalysisReindexRequest(BaseModel):
    history_id: str
    scene_ids: List[int] = Field(default_factory=list)
    refresh: bool = False

@video_analysis_router.post("/reindex")
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


@video_analysis_router.post("/history")
async def overwrite_history(req: HistorySaveRequest):
    await video_analysis_db_service.overwrite_history([item.model_dump(exclude_none=True) for item in req.history])
    log.info(f"视频分析历史记录已覆盖, 共 {len(req.history)} 条")
    return {"success": True}


@video_analysis_router.post("/history/update")
async def update_single_history(req: HistoryUpdateRequest):
    """追加或替换某一条历史记录(按 id 匹配), 不存在则追加到最前"""
    new_item = req.item.model_dump(exclude_none=True)
    existed = await video_analysis_db_service.get_history_item(new_item.get("id", ""))
    await video_analysis_db_service.upsert_history_item(new_item)
    replaced = existed is not None
    log.info(f"视频分析历史记录更新: id={new_item.get('id')}, 操作={'替换' if replaced else '新增'}")
    return {"success": True, "replaced": replaced}


class RewriteScriptRequest(BaseModel):
    script: str = Field(..., description="需要提取的口播脚本或自然语言描述")
    topic: Optional[str] = None
    title: Optional[str] = None
    car_model: Optional[str] = None

@video_analysis_router.post("/rewrite-script")
async def rewrite_script_endpoint(req: RewriteScriptRequest):
    """
    调用大模型将自然语言脚本提取为结构化的检索标签。
    直接返回 SeedtextIndexTagsEnvelope 格式的字典。
    """
    log.info(f"[rewrite-script] received request: script={req.script!r} topic={req.topic!r} title={req.title!r} car_model={req.car_model!r}")
    if not req.script.strip():
        return {"success": False, "error": "script cannot be empty"}
    
    try:
        # 调用现有的 service 逻辑
        storyboard, tags = await rewrite_script_to_storyboard_and_tags(
            script=req.script,
            topic=req.topic,
            title=req.title,
            car_model=req.car_model,
            index=0
        )
        return {"success": True, "tags": tags.model_dump(exclude_none=True)}
    except Exception as e:
        log.error(f"rewrite_script failed: {e}")
        return {"success": False, "error": str(e)}


# ---------------- 视频分析主接口 ----------------

@video_analysis_router.post("")
async def analyze_video_endpoint(
    file: UploadFile = File(..., description="待分析的视频文件"),
    frame_interval: float = Form(2.0),
    threshold: float = Form(30.0),
    custom_prompt: Optional[str] = Form(None),
    split_scenes: bool = Form(True),
    workspace: str = Form("v1"),
    car_model: Optional[str] = Form(None),
):
    """接收上传视频并执行完整分析流水线, 直接返回分镜卡片列表"""
    log.info(f"[analyze_video_endpoint] received POST request: filename={file.filename}, workspace={workspace}, car_model={car_model}, frame_interval={frame_interval}, threshold={threshold}, split_scenes={split_scenes}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    project_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
    local_path = os.path.join(UPLOAD_TMP_DIR, f"{project_id}_{file.filename}")

    # 持久化上传文件
    try:
        with open(local_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        log.info(f"[{project_id}] 视频已落盘: {local_path}")
    except Exception as e:
        log.error(f"保存上传文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {e}")

    try:
        log.info(f"[{project_id}] 收到视频分析请求: workspace={workspace}, frame_interval={frame_interval}, threshold={threshold}, split_scenes={split_scenes}")
        
        # 尝试从缓存获取源视频 OBS URL，或上传
        from services.analysis_video import _get_or_upload_source_video
        obs_video_url = await _get_or_upload_source_video(local_path, project_id)
        
        cards = await analyze_video(
            local_video_path=local_path,
            project_id=project_id,
            frame_interval=frame_interval,
            threshold=threshold,
            custom_prompt=custom_prompt,
            split_scenes=split_scenes,
            cleanup_workspace=True,
            workspace=workspace,
            car_model=car_model,
        )

        # 打包成一条历史记录
        history_item = VideoAnalysisHistoryItem(
            id=project_id,
            name=file.filename,
            time=datetime.now().isoformat(timespec="seconds"),
            video_url=obs_video_url,
            workspace=workspace,
            cards=cards,
        )
        # 顺便写入历史(DB)
        # 根据 workspace 决定写入 v1 还是 v2 的卡片表
        await video_analysis_db_service.upsert_history_item(
            history_item.model_dump(exclude_none=True),
            shot_cards_version=workspace
        )

        # 同步：入库 OpenSearch 并写回状态，确保前端拿到的卡片就是最终状态
        keys_ok = [(project_id, c.scene_id) for c in cards if not c.error]
        keys_failed = [(project_id, c.scene_id) for c in cards if c.error]
        if keys_failed:
            # 分镜分析本身失败的，标记为 FAILED（和“入库失败”同一状态，便于前端统一展示）
            await video_analysis_db_service.update_cards_index_status(
                keys_failed,
                status="FAILED",
                error="analysis failed",
                shot_cards_version=workspace
            )

        if keys_ok:
            try:
                await index_shotcards_to_opensearch(cards, id_prefix=project_id, refresh=False, workspace=workspace)
                await video_analysis_db_service.update_cards_index_status(keys_ok, status="OK", error=None, shot_cards_version=workspace)
                log.info(f"[{project_id}] OpenSearch 入库完成并已写回状态")
            except Exception as _e:
                await video_analysis_db_service.update_cards_index_status(keys_ok, status="FAILED", error=str(_e), shot_cards_version=workspace)
                log.error(f"[{project_id}] OpenSearch 入库失败并已写回状态: {_e}")

        # 返回 DB 中最新的 item（包含 os_index_status / os_index_error）
        item = await video_analysis_db_service.get_history_item(project_id)
        return {"success": True, "item": item or history_item.model_dump(exclude_none=True)}
    except Exception as e:
        log.error(f"[{project_id}] 视频分析流程异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理上传的临时视频文件(保留帧抽取的 workspace, 方便排查)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception as cleanup_err:
            log.warning(f"清理临时文件失败: {cleanup_err}")
