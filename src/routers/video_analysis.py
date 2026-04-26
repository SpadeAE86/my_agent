# routers/video_analysis.py — 视频分析路由
# 端点:
#   POST /video-analysis           — 上传视频并执行分析流水线, 返回分镜卡片列表
#   GET  /video-analysis/history   — 获取所有历史分析记录
#   POST /video-analysis/history   — 覆盖写入全部历史记录
#   POST /video-analysis/history/update — 追加/更新单条历史记录

import os
import sys
import uuid
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List

from models.pydantic.video_analysis_request import (
    HistorySaveRequest,
    HistoryUpdateRequest,
    VideoAnalysisHistoryItem,
    ShotCard as PydShotCard,
)
from services.analysis_video import analyze_video, index_shotcards_to_opensearch
from services.video_analysis_db_service import video_analysis_db_service
from infra.logging.logger import logger as log
from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import query_builder
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from models.pydantic.opensearch_index.base_index import get_index_name


video_analysis_router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])

UPLOAD_TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "video_analysis_uploads",
)


# ---------------- 历史记录接口 ----------------

@video_analysis_router.get("/history")
async def get_history():
    history = await video_analysis_db_service.list_history()
    return {"success": True, "history": history}


@video_analysis_router.get("/history/{history_id}")
async def get_history_item(history_id: str):
    item = await video_analysis_db_service.get_history_item(history_id)
    if item is None:
        raise HTTPException(status_code=404, detail="history not found")
    return {"success": True, "item": item}


@video_analysis_router.get("/cards")
async def get_cards(history_id: Optional[str] = None):
    """
    Get cards by history_id.
    - history_id=__all__ or missing => all cards across histories
    - else => cards of one history (from DB)
    """
    if not history_id or history_id == "__all__":
        cards = await video_analysis_db_service.list_all_cards()
        return {"success": True, "cards": cards}

    item = await video_analysis_db_service.get_history_item(history_id)
    if item is None:
        raise HTTPException(status_code=404, detail="history not found")
    return {"success": True, "cards": item.get("cards", [])}

class VideoAnalysisSearchToken(BaseModel):
    text: str
    join: Optional[str] = "AND"
    not_: bool = Field(False, alias="not")

class VideoAnalysisSearchRequest(BaseModel):
    tokens: List[VideoAnalysisSearchToken] = Field(default_factory=list)
    fuzzy: bool = False
    history_id: Optional[str] = None
    size: int = 50

def _parse_doc_id(doc_id: str) -> Optional[tuple[str, int]]:
    """
    doc_id format: "{history_id}_scene_{scene_id:03d}"
    """
    try:
        if not doc_id:
            return None
        marker = "_scene_"
        if marker not in doc_id:
            return None
        hid, sid = doc_id.split(marker, 1)
        return hid, int(sid)
    except Exception:
        return None

@video_analysis_router.post("/search")
async def search_cards(req: VideoAnalysisSearchRequest):
    """
    Search cards via OpenSearch (hybrid: keyword + vector).
    Returns full ShotCard payloads from DB (source of truth) ordered by OpenSearch score.
    """
    tokens = [t.text.strip() for t in (req.tokens or []) if t.text and t.text.strip()]
    if not tokens:
        return {"success": True, "cards": []}

    query_text = " ".join(tokens)
    size = max(1, min(int(req.size or 50), 200))

    # Build base hybrid query (exclude vectors by default)
    body = query_builder.build_dynamic_hybrid_search(
        CarInteriorAnalysis,
        query_text,
        size=size,
        bm25_factor=0.5,
        vector_factor=0.5,
    )

    # Optional history filter (restrict to one analysis run)
    history_id = (req.history_id or "").strip()
    if history_id and history_id != "__all__":
        # Filter by doc id prefix: "{history_id}_scene_"
        prefix = f"{history_id}_scene_"
        q = body.get("query") or {}
        body["query"] = {
            "bool": {
                "must": [q],
                "filter": [{"prefix": {"id": prefix}}],
            }
        }

    try:
        await opensearch_connector.ensure_init()
        client = await opensearch_connector.get_client()
        index_name = get_index_name(CarInteriorAnalysis)
        resp = await client.search(index=index_name, body=body)
        hits = ((resp.get("hits") or {}).get("hits") or [])
    except Exception as e:
        log.error(f"video-analysis search failed: {e}")
        return {"success": False, "error": str(e), "cards": []}

    # Map OpenSearch doc ids -> (history_id, scene_id)
    keys_in_order: List[tuple[str, int]] = []
    for h in hits:
        doc_id = h.get("_id") or (h.get("_source") or {}).get("id")
        k = _parse_doc_id(str(doc_id)) if doc_id else None
        if k:
            keys_in_order.append(k)

    if not keys_in_order:
        return {"success": True, "cards": []}

    cards = await video_analysis_db_service.get_cards_by_keys(keys_in_order)
    by_key = {(c.get("history_id"), int(c.get("scene_id") or 0)): c for c in (cards or [])}
    ordered = [by_key[k] for k in keys_in_order if k in by_key]
    return {"success": True, "cards": ordered}

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
        await video_analysis_db_service.update_cards_index_status(keys, status="PENDING", error=None)
    except Exception as e:
        log.warning(f"reindex: failed to mark PENDING: {e}")

    rows = await video_analysis_db_service.get_cards_by_keys(keys)
    if not rows:
        await video_analysis_db_service.update_cards_index_status(keys, status="FAILED", error="cards not found in db")
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
        await video_analysis_db_service.update_cards_index_status(keys, status="FAILED", error="no valid cards to reindex")
        return {"success": False, "error": "no valid cards", "updated": []}

    try:
        await index_shotcards_to_opensearch(cards, id_prefix=history_id, refresh=bool(req.refresh))
        await video_analysis_db_service.update_cards_index_status(ok_keys, status="OK", error=None)
    except Exception as e:
        await video_analysis_db_service.update_cards_index_status(ok_keys, status="FAILED", error=str(e))
        return {"success": False, "error": str(e), "updated": []}

    updated_rows = await video_analysis_db_service.get_cards_by_keys(ok_keys)
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


# ---------------- 视频分析主接口 ----------------

@video_analysis_router.post("")
async def analyze_video_endpoint(
    file: UploadFile = File(..., description="待分析的视频文件"),
    frame_interval: float = Form(2.0),
    threshold: float = Form(30.0),
    custom_prompt: Optional[str] = Form(None),
    split_scenes: bool = Form(True),
):
    """接收上传视频并执行完整分析流水线, 直接返回分镜卡片列表"""
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
        cards = await analyze_video(
            local_video_path=local_path,
            project_id=project_id,
            frame_interval=frame_interval,
            threshold=threshold,
            custom_prompt=custom_prompt,
            split_scenes=split_scenes,
            cleanup_workspace=True,
        )

        # 打包成一条历史记录
        history_item = VideoAnalysisHistoryItem(
            id=project_id,
            name=file.filename,
            time=datetime.now().isoformat(timespec="seconds"),
            video_url=None,
            cards=cards,
        )
        # 顺便写入历史(DB)
        await video_analysis_db_service.upsert_history_item(history_item.model_dump(exclude_none=True))

        # 同步：入库 OpenSearch 并写回状态，确保前端拿到的卡片就是最终状态
        keys_ok = [(project_id, c.scene_id) for c in cards if not c.error]
        keys_failed = [(project_id, c.scene_id) for c in cards if c.error]
        if keys_failed:
            # 分镜分析本身失败的，标记为 FAILED（和“入库失败”同一状态，便于前端统一展示）
            await video_analysis_db_service.update_cards_index_status(
                keys_failed,
                status="FAILED",
                error="analysis failed",
            )

        if keys_ok:
            try:
                await index_shotcards_to_opensearch(cards, id_prefix=project_id, refresh=False)
                await video_analysis_db_service.update_cards_index_status(keys_ok, status="OK", error=None)
                log.info(f"[{project_id}] OpenSearch 入库完成并已写回状态")
            except Exception as _e:
                await video_analysis_db_service.update_cards_index_status(keys_ok, status="FAILED", error=str(_e))
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
