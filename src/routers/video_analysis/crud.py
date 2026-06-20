# routers/video_analysis.py — 视频分析路由
# 端点:
#   POST /video-analysis           — 上传视频并执行分析流水线, 返回分镜卡片列表
#   GET  /video-analysis/history   — 获取所有历史分析记录
#   POST /video-analysis/history   — 覆盖写入全部历史记录
#   POST /video-analysis/history/update — 追加/更新单条历史记录

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from models.pydantic.video_analysis_request import (
    HistorySaveRequest,
    HistoryUpdateRequest,
)
from services.video_match_services.video_analysis_db_service import video_analysis_db_service
from services.taskboard_services.http_request_trace_service import http_request_trace_service
from infra.logging.logger import logger as log
from models.pydantic.opensearch_index.base_index import (
    get_vector_fields, get_searchable_fields, get_field_weights, get_vector_weights,
)
from services.video_match_services.token_join_template_service import (
    TOKEN_JOIN_TERM_FIELDS_V2,
    create_template,
    delete_template,
    get_default_and_fields,
    list_templates,
    set_default_template,
    update_template,
)
from core.workspace import list_workspaces, DEFAULT_WORKSPACE_KEY, get_workspace



crud_router = APIRouter()
# ---------------- Workspace 接口 ----------------

@crud_router.get("/workspaces")
async def get_workspaces():
    """返回当前支持的 workspace 列表（key / label / description）及默认值。"""
    return {
        "success": True,
        "workspaces": list_workspaces(),
        "default": DEFAULT_WORKSPACE_KEY,
    }


# ---------------- 搜索策略接口 ----------------

@crud_router.get("/index-fields")
async def get_index_fields(workspace: str = Query("v2")):
    """获取指定 workspace 下索引的可用字段，用于前端动态生成权重调节滑块"""
    IndexModel = get_workspace(workspace).index_class
    text_fields = get_searchable_fields(IndexModel)
    vector_fields = get_vector_fields(IndexModel)
    return {
        "success": True,
        "text_fields": text_fields,
        "vector_fields": vector_fields,
        "text_field_weights": get_field_weights(IndexModel),
        "vector_field_weights": get_vector_weights(IndexModel),
    }

class SearchStrategyCreate(BaseModel):
    name: str
    bm25_weight: float
    vector_weight: float
    text_weights: Optional[dict] = None
    vector_weights: Optional[dict] = None
    is_default: bool = False
    use_rrf: bool = False

@crud_router.get("/search-strategies")
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

@crud_router.post("/search-strategies")
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
                        is_default=req.is_default,
                        use_rrf=req.use_rrf,
                    )
                )
            else:
                new_strategy = VideoAnalysisSearchStrategy(
                    name=req.name,
                    bm25_weight=req.bm25_weight,
                    vector_weight=req.vector_weight,
                    text_weights=req.text_weights,
                    vector_weights=req.vector_weights,
                    is_default=req.is_default,
                    use_rrf=req.use_rrf,
                )
                conn.add(new_strategy)
                
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise e
            
    return {"success": True}

@crud_router.delete("/search-strategies/{strategy_id}")
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


# ---------------- Token AND 模板（转写 / 检索 term filter） ----------------


class TokenJoinTemplateCreate(BaseModel):
    name: str
    workspace: str = "v2"
    and_segment_fields: List[str] = Field(default_factory=list)
    is_default: bool = False


class TokenJoinTemplateUpdate(BaseModel):
    name: Optional[str] = None
    and_segment_fields: Optional[List[str]] = None
    is_default: Optional[bool] = None


@crud_router.get("/token-join-templates/allowed-fields")
async def token_join_allowed_fields():
    """可作 AND term filter 的 v2 索引 keyword 字段（与 segment 字段名一致）。"""
    return {"success": True, "fields": sorted(TOKEN_JOIN_TERM_FIELDS_V2)}


@crud_router.get("/token-join-templates/default-fields")
async def token_join_default_fields(workspace: str = Query("v2")):
    fields = await get_default_and_fields(workspace)
    return {"success": True, "workspace": (workspace or "v2").strip() or "v2", "and_segment_fields": fields}


@crud_router.get("/token-join-templates")
async def token_join_templates_list(workspace: Optional[str] = Query(None)):
    rows = await list_templates(workspace=workspace)
    return {"success": True, "templates": rows}


@crud_router.post("/token-join-templates")
async def token_join_templates_create(req: TokenJoinTemplateCreate):
    row = await create_template(
        name=req.name,
        workspace=req.workspace,
        and_segment_fields=req.and_segment_fields,
        is_default=req.is_default,
    )
    return {"success": True, "template": row.model_dump()}


@crud_router.put("/token-join-templates/{template_id}")
async def token_join_templates_put(template_id: int, req: TokenJoinTemplateUpdate):
    row = await update_template(
        template_id,
        name=req.name,
        and_segment_fields=req.and_segment_fields,
        is_default=req.is_default,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True, "template": row.model_dump()}


@crud_router.delete("/token-join-templates/{template_id}")
async def token_join_templates_delete(template_id: int):
    ok = await delete_template(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True}


@crud_router.post("/token-join-templates/{template_id}/set-default")
async def token_join_templates_set_default_route(template_id: int):
    row = await set_default_template(template_id)
    if row is None:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True, "template": row.model_dump()}
# ---------------- 历史记录接口 ----------------

@crud_router.get("/history")
async def get_history(
    workspace: Optional[str] = Query(None, description="用来区分工作区"),
    ids: Optional[str] = Query(None, description="逗号分隔的ID列表，用于局部轮询")
):
    history = await video_analysis_db_service.list_history(workspace=workspace, ids=ids)
    return {"success": True, "history": history}


@crud_router.get("/task-badges")
async def video_analysis_task_badges(workspace: Optional[str] = Query(None)):
    """侧边栏角标：PENDING + RUNNING 数量（可按 workspace 过滤，不传则全 workspace）。"""
    counts = await video_analysis_db_service.count_active_task_statuses(workspace=workspace)
    active_total = int(counts.get("PENDING", 0)) + int(counts.get("RUNNING", 0))
    return {"success": True, "counts": counts, "active_total": active_total}


@crud_router.get("/history/{history_id}/detail")
async def get_history_task_detail(history_id: str):
    """任务看板：视频分析历史条目的 HTTP 详情；有 request_id 时联表 http_request_traces。"""
    from services.video_match_services.task_detail_service import build_video_analysis_task_detail, merge_http_trace_into_detail

    row = await video_analysis_db_service.get_history_row(history_id)
    if row is None:
        raise HTTPException(status_code=404, detail="history not found")
    base = build_video_analysis_task_detail(row)
    trace_dict = None
    rid = row.get("request_id")
    if rid:
        trace_dict = await http_request_trace_service.get_dict(str(rid))
    return {"success": True, "detail": merge_http_trace_into_detail(base, trace_dict)}


@crud_router.get("/history/{history_id}")
async def get_history_item(
    history_id: str,
    shot_cards_version: str = Query("v1", description="分镜表版本：v1 旧卡片 / v2 对齐 SceneAnalysisResultV2"),
):
    ver = shot_cards_version if shot_cards_version in ("v1", "v2") else "v1"
    item = await video_analysis_db_service.get_history_item(history_id, shot_cards_version=ver)
    if item is None:
        raise HTTPException(status_code=404, detail="history not found")
    return {"success": True, "item": item}


@crud_router.get("/cards")
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

@crud_router.post("/history")
async def overwrite_history(req: HistorySaveRequest):
    await video_analysis_db_service.overwrite_history([item.model_dump(exclude_none=True) for item in req.history])
    log.info(f"视频分析历史记录已覆盖, 共 {len(req.history)} 条")
    return {"success": True}


@crud_router.post("/history/update")
async def update_single_history(req: HistoryUpdateRequest):
    """追加或替换某一条历史记录(按 id 匹配), 不存在则追加到最前"""
    new_item = req.item.model_dump(exclude_none=True)
    existed = await video_analysis_db_service.get_history_item(new_item.get("id", ""))
    await video_analysis_db_service.upsert_history_item(new_item)
    replaced = existed is not None
    log.info(f"视频分析历史记录更新: id={new_item.get('id')}, 操作={'替换' if replaced else '新增'}")
    return {"success": True, "replaced": replaced}


