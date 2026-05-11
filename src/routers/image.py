# routers/image.py — 图片和文本生成路由
# 端点:
#   POST /image          — 调用 Seedream 模型生成图片
#   POST /text           — 调用 Seed 文本模型生成文本

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import asyncio
import datetime

from models.pydantic.request import ImageGenerateRequest, TextGenerateRequest
from utils.call_model_utils import call_doubao_seedream, call_doubao_seedtext
from infra.logging.logger import logger as log
from services.image_history_db_service import image_history_db_service
from services.media_mirror_service import mirror_remote_url_to_obs, is_obs_url
from services.image_generate_service import generate_image as service_generate_image
import asyncio

image_router = APIRouter(prefix="", tags=["image", "text"])

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "image_history.json")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"读取历史记录失败: {e}")
        return []

def save_history(history_list):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"保存历史记录失败: {e}")

class ImageHistoryItem(BaseModel):
    id: str
    prompt: str
    model: str
    size: Optional[str] = None
    resolution: Optional[str] = None
    ratio: Optional[str] = None
    duration: Optional[int] = None
    url: Optional[str] = None
    time: str
    type: str
    referenceMedia: Optional[List[dict]] = None
    error: Optional[str] = None
    taskId: Optional[str] = None
    status: Optional[str] = None

class HistorySaveRequest(BaseModel):
    history: List[ImageHistoryItem]

@image_router.get("/image/history")
async def get_image_history():
    history = await image_history_db_service.list_all()
    return {"success": True, "history": history}

@image_router.post("/image/history")
async def update_image_history(req: HistorySaveRequest):
    payload = [item.model_dump(exclude_none=True) for item in req.history]
    await image_history_db_service.upsert_many(payload)
    log.info(f"历史记录已更新(DB)，共{len(req.history)}条记录")

    # 后台任务：将生成结果 url 镜像到 OBS（不阻塞前端）
    async def _mirror_and_update(items: list[dict]):
        for it in items:
            item_id = it.get("id")
            url = it.get("url")
            if not item_id or not url or is_obs_url(url):
                continue
            try:
                # type: t2i / i2i / t2v / i2v ...
                t = (it.get("type") or "").lower()
                prefix = "ai_picture/generated"
                if "v" in t:
                    prefix = "ai_picture/generated_video"
                else:
                    prefix = "ai_picture/generated_image"
                obs_url = await mirror_remote_url_to_obs(url, obs_prefix=prefix)
                await image_history_db_service.update_obs_url(item_id, obs_url)
                log.info(f"[mirror] updated {item_id} url -> {obs_url}")
            except Exception as e:
                log.warning(f"[mirror] failed for {item_id}: {e}")

    asyncio.create_task(_mirror_and_update(payload))
    return {"success": True}

@image_router.delete("/image/history/{item_id}")
async def delete_image_history_item(item_id: str):
    success = await image_history_db_service.delete_by_id(item_id)
    if success:
        log.info(f"历史记录已删除: {item_id}")
        return {"success": True}
    else:
        log.warning(f"尝试删除不存在的历史记录: {item_id}")
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Item not found")

class ImageGenerateResponse(BaseModel):
    """图片生成响应"""
    success: bool
    image_url: Optional[str] = None
    task_id: Optional[str] = None
    error: Optional[str] = None

@image_router.get("/image/status/{task_id}")
async def get_image_status(task_id: str):
    """查询异步生图任务状态"""
    item = await image_history_db_service.get_by_id(task_id)
    if not item:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = item.get("status") or "unknown"
    url = item.get("url")
    
    return {
        "success": True,
        "status": status,
        "url": url,
        "error": item.get("error")
    }


class TextGenerateResponse(BaseModel):
    """文本生成响应"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None


import asyncio

@image_router.post("/image", response_model=ImageGenerateResponse)
async def generate_image(req: ImageGenerateRequest, background_tasks: BackgroundTasks, async_mode: bool = True):
    """
    调用豆包 Seedream 模型生成图片
    """
    try:
        log.info(f"收到图片生成请求: model={req.model}, size={req.size}, async={async_mode}")
        log.info(f"提示词: {req.prompt}")
        
        if not async_mode:
            # 原有的同步模式（保留用于备选）
            image_url = await service_generate_image(
                prompt=req.prompt,
                model=req.model.value,
                size=req.size,
                reference_image_list=req.reference_image_list
            )
            if image_url:
                log.info(f"图片生成成功(同步): {image_url}")
                return ImageGenerateResponse(success=True, image_url=image_url)
            else:
                return ImageGenerateResponse(success=False, error="图片生成失败")
        
        # 异步模式：先占位
        import uuid
        task_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        time_str = now.strftime("%m-%d %H:%M")
        
        # 准备入库基础数据
        payload = {
            "id": task_id,
            "prompt": req.prompt,
            "model": req.model.value,
            "size": req.size,
            "ratio": req.ratio,
            "time": time_str,
            "type": "i2i" if req.reference_image_list else "t2i",
            "status": "running",
            "referenceMedia": [{"url": m, "type": "image"} for m in req.reference_image_list] if req.reference_image_list else None
        }
        await image_history_db_service.upsert_many([payload])
        
        # 定义后台处理逻辑
        async def _do_generate(tid: str, r: ImageGenerateRequest):
            try:
                img_url = await service_generate_image(
                    prompt=r.prompt,
                    model=r.model.value,
                    size=r.size,
                    reference_image_list=r.reference_image_list
                )
                if img_url:
                    # 自动镜像到 OBS 并更新状态
                    prefix = "ai_picture/generated_image"
                    try:
                        from services.media_mirror_service import mirror_remote_url_to_obs
                        obs_url = await mirror_remote_url_to_obs(img_url, obs_prefix=prefix)
                        await image_history_db_service.upsert_many([{
                            "id": tid,
                            "obs_url": obs_url,
                            "doubao_url": img_url,
                            "status": "success"
                        }])
                    except Exception as e:
                        log.warning(f"Mirror failed in background for {tid}: {e}")
                        await image_history_db_service.upsert_many([{
                            "id": tid,
                            "doubao_url": img_url,
                            "status": "success"
                        }])
                else:
                    await image_history_db_service.upsert_many([{
                        "id": tid,
                        "status": "failed",
                        "error": "生成失败，未获取到 URL"
                    }])
            except Exception as e:
                log.error(f"Background generation error for {tid}: {e}")
                await image_history_db_service.upsert_many([{
                    "id": tid,
                    "status": "failed",
                    "error": str(e)
                }])

        background_tasks.add_task(_do_generate, task_id, req)
        return ImageGenerateResponse(success=True, task_id=task_id)
            
    except Exception as e:
        log.error(f"图片生成异常: {e}")
        return ImageGenerateResponse(success=False, error=str(e))


@image_router.post("/text", response_model=TextGenerateResponse)
async def generate_text(req: TextGenerateRequest):
    """
    调用豆包 Seed 文本模型生成文本
    
    支持模型:
    - Seed 2.0 Pro (默认)
    - Seed 2.0 Lite
    - Seed 2.0 Mini
    """
    try:
        log.info(f"收到文本生成请求: model={req.model}")
        if req.system_prompt:
            log.info(f"系统提示词已提供")
        log.info(f"提示词: {req.prompt}")
        
        text = await call_doubao_seedtext(
            prompt=req.prompt,
            model=req.model.value,
            system_prompt=req.system_prompt,
            video_duration=req.video_duration,
        )
        
        if text:
            log.info(f"文本生成成功")
            return TextGenerateResponse(success=True, text=text)
        else:
            log.error("文本生成失败")
            return TextGenerateResponse(success=False, error="文本生成失败，请检查提示词或重试")
            
    except Exception as e:
        log.error(f"文本生成异常: {e}")
        return TextGenerateResponse(success=False, error=str(e))
