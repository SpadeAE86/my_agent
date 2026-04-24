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

from models.pydantic.video_analysis_request import HistorySaveRequest, HistoryUpdateRequest, VideoAnalysisHistoryItem
from services.analysis_video import analyze_video
from services.video_analysis_db_service import video_analysis_db_service
from infra.logging.logger import logger as log


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

        return {
            "success": True,
            "item": history_item.model_dump(exclude_none=True),
        }
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
