# routers/video.py — 视频生成路由
# 端点:
#   POST /video          — 提交 Seedance 模型生成视频任务
#   GET /video/status/{task_id} — 查询任务状态

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from models.pydantic.request import VideoGenerateRequest
from utils.call_model_utils import call_doubao_seedance, get_seedance_task_status
from infra.logging.logger import logger as log

video_router = APIRouter(prefix="/video", tags=["video"])

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "video_history.json")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"读取视频历史记录失败: {e}")
        return []

def save_history(history_list):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"保存视频历史记录失败: {e}")

class VideoHistoryItem(BaseModel):
    id: str
    prompt: str
    model: str
    resolution: Optional[str] = None
    size: Optional[str] = None
    ratio: str
    duration: Optional[int] = None
    generate_audio: Optional[bool] = None
    url: Optional[str] = None
    time: str
    type: str
    referenceMedia: Optional[List[dict]] = None
    error: Optional[str] = None
    taskId: Optional[str] = None
    status: Optional[str] = None

class HistorySaveRequest(BaseModel):
    history: List[VideoHistoryItem]

@video_router.get("/history")
async def get_video_history():
    return {"success": True, "history": load_history()}

@video_router.post("/history")
async def update_video_history(req: HistorySaveRequest):
    save_history([item.model_dump(exclude_none=True) for item in req.history])
    log.info(f"视频历史记录已更新，共{len(req.history)}条记录")
    return {"success": True}

class VideoGenerateResponse(BaseModel):
    """视频生成响应"""
    success: bool
    task_id: Optional[str] = None
    error: Optional[str] = None

@video_router.post("", response_model=VideoGenerateResponse)
async def generate_video(req: VideoGenerateRequest):
    """
    提交豆包 Seedance 模型生成视频任务
    """
    try:
        log.info(f"收到视频生成请求: model={req.model}, resolution={req.resolution}, ratio={req.ratio}")
        log.info(f"提示词: {req.prompt}")
        
        task_id = await call_doubao_seedance(
            prompt=req.prompt,
            model=req.model.value,
            resolution=req.resolution,
            ratio=req.ratio,
            duration=req.duration,
            generate_audio=req.generate_audio,
            reference_image_list=req.reference_image_list,
            reference_video_list=req.reference_video_list,
            reference_audio_list=req.reference_audio_list
        )
        
        if task_id:
            log.info(f"视频生成任务提交成功, task_id: {task_id}")
            return VideoGenerateResponse(success=True, task_id=task_id)
        else:
            log.error("视频生成任务提交失败")
            return VideoGenerateResponse(success=False, error="视频生成任务提交失败，请检查参数或重试")
            
    except Exception as e:
        log.error(f"视频生成请求异常: {e}")
        return VideoGenerateResponse(success=False, error=str(e))

@video_router.get("/status/{task_id}")
async def get_video_status(task_id: str):
    """
    查询视频生成任务状态
    """
    try:
        log.info(f"查询视频生成状态, task_id: {task_id}")
        status_info = await get_seedance_task_status(task_id)
        
        if status_info:
            return {"success": True, "data": status_info}
        else:
            return {"success": False, "error": "查询状态失败"}
            
    except Exception as e:
        log.error(f"查询视频生成状态异常: {e}")
        return {"success": False, "error": str(e)}
