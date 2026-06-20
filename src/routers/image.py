# routers/image.py — 图片和文本生成路由
# 端点:
#   POST /image          — 调用 Seedream 模型生成图片
#   POST /text           — 调用 Seed 文本模型生成文本

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import datetime
import time

from models.pydantic.request import ImageGenerateRequest, TextGenerateRequest, SeedreamModel
from utils.call_model_utils import call_doubao_seedtext
from infra.logging.logger import logger as log
from services.media_generate_services.image_history_db_service import image_history_db_service
from services.media_mirror_service import mirror_remote_url_to_obs, is_obs_url
from services.media_generate_services.image_generate_service import generate_image as service_generate_image
from services.taskboard_services.http_request_trace_service import http_request_trace_service

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
    request_id: Optional[str] = None

class HistorySaveRequest(BaseModel):
    history: List[ImageHistoryItem]

@image_router.get("/image/history")
async def get_image_history(ids: Optional[str] = None):
    history = await image_history_db_service.list_all(ids=ids)
    return {"success": True, "history": history}


@image_router.get("/image/history/{item_id}/detail")
async def get_image_history_task_detail(item_id: str):
    """任务看板：合成「HTTP 调用详情」；有 request_id 时联表 http_request_traces。"""
    from services.video_match_services.task_detail_service import build_image_task_detail, merge_http_trace_into_detail

    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=404, detail="Item not found")
    row = await image_history_db_service.get_by_id_or_task_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    base = build_image_task_detail(row)
    trace_dict = None
    rid = row.get("request_id")
    if rid:
        trace_dict = await http_request_trace_service.get_dict(str(rid))
    return {"success": True, "detail": merge_http_trace_into_detail(base, trace_dict)}

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


def _reference_urls_from_history_row(row: dict) -> Optional[list[str]]:
    rm = row.get("referenceMedia")
    if not rm or not isinstance(rm, list):
        return None
    urls: list[str] = []
    for x in rm:
        if isinstance(x, dict):
            u = x.get("url")
            if u:
                t = str(u).strip()
                if t:
                    urls.append(t)
        elif isinstance(x, str) and x.strip():
            urls.append(x.strip())
    return urls or None


def _seedream_model_from_stored(model_str: Optional[str]) -> SeedreamModel:
    v = (model_str or "").strip()
    for m in SeedreamModel:
        if m.value == v:
            return m
    log.warning("unknown stored model %r, fallback to Seedream 5.0", model_str)
    return SeedreamModel.V5_0


async def run_image_generation_job(tid: str, r: ImageGenerateRequest, trace_rid: str) -> None:
    """异步生图后台任务（POST /image 与重试共用）。"""
    t0 = time.monotonic()
    duration_ms = 0

    async def _finalize_ok(resp: dict):
        nonlocal duration_ms
        duration_ms = int((time.monotonic() - t0) * 1000)
        await http_request_trace_service.finalize(
            trace_rid,
            status_code=200,
            response_body=resp,
            duration_ms=duration_ms,
            business_success=True,
        )

    async def _finalize_fail(msg: str, resp: Optional[dict] = None):
        nonlocal duration_ms
        duration_ms = int((time.monotonic() - t0) * 1000)
        await http_request_trace_service.finalize(
            trace_rid,
            status_code=500,
            error_message=msg,
            response_body=resp or {"success": False, "error": msg},
            duration_ms=duration_ms,
            business_success=False,
        )

    try:
        img_url = await service_generate_image(
            prompt=r.prompt,
            model=r.model.value,
            size=r.size,
            reference_image_list=r.reference_image_list,
        )
        if img_url:
            prefix = "ai_picture/generated_image"
            try:
                obs_url = await mirror_remote_url_to_obs(img_url, obs_prefix=prefix)
                await image_history_db_service.upsert_many(
                    [
                        {
                            "id": tid,
                            "obs_url": obs_url,
                            "doubao_url": img_url,
                            "status": "success",
                        }
                    ]
                )
                await _finalize_ok(
                    {
                        "success": True,
                        "image_url": img_url,
                        "obs_url": obs_url,
                        "task_id": tid,
                    }
                )
            except Exception as e:
                log.warning(f"Mirror failed in background for {tid}: {e}")
                await image_history_db_service.upsert_many(
                    [
                        {
                            "id": tid,
                            "doubao_url": img_url,
                            "status": "success",
                        }
                    ]
                )
                await _finalize_ok(
                    {
                        "success": True,
                        "image_url": img_url,
                        "task_id": tid,
                        "mirror_warning": str(e),
                    }
                )
        else:
            await image_history_db_service.upsert_many(
                [
                    {
                        "id": tid,
                        "status": "failed",
                        "error": "生成失败，未获取到 URL",
                    }
                ]
            )
            await _finalize_fail("生成失败，未获取到 URL")
    except Exception as e:
        log.error(f"Background generation error for {tid}: {e}")
        await image_history_db_service.upsert_many(
            [{"id": tid, "status": "failed", "error": str(e)}]
        )
        await _finalize_fail(str(e))


@image_router.post("/image/history/{item_id}/retry")
async def retry_failed_image_generation(item_id: str, background_tasks: BackgroundTasks):
    """仅失败任务：用同一行 id 重新排队生图（参数来自历史行）。"""
    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="invalid id")
    row = await image_history_db_service.get_by_id_or_task_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    st = (row.get("status") or "").lower()
    if st != "failed":
        raise HTTPException(status_code=400, detail="仅失败任务可重试")
    prompt = (row.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="记录缺少提示词，无法重试")

    model = _seedream_model_from_stored(row.get("model"))
    ref = _reference_urls_from_history_row(row)
    try:
        req = ImageGenerateRequest(
            prompt=prompt,
            model=model,
            size=str(row.get("size") or "720x1280"),
            ratio=row.get("ratio"),
            reference_image_list=ref,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法构造生图请求: {e}") from e

    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d %H:%M")
    rid = str(row.get("id") or item_id)
    run_start = datetime.datetime.now(datetime.timezone.utc)

    trace_id = await http_request_trace_service.create_initial(
        request_url=f"/image/history/{rid}/retry",
        http_method="POST",
        request_body={"retry_of": rid, **req.model_dump(mode="json")},
        business_type="IMAGE_GEN",
        method_name="POST /image/history/retry",
        upstream_task_id=rid,
    )

    await image_history_db_service.upsert_many(
        [
            {
                "id": rid,
                "status": "running",
                "error": None,
                "request_id": trace_id,
                "doubao_url": None,
                "obs_url": None,
                "time": time_str,
                "current_run_started_at": run_start,
            }
        ]
    )
    background_tasks.add_task(run_image_generation_job, rid, req, trace_id)
    return {"success": True, "task_id": rid}


@image_router.post("/image", response_model=ImageGenerateResponse)
async def generate_image(req: ImageGenerateRequest, background_tasks: BackgroundTasks, async_mode: bool = True):
    """
    调用豆包 Seedream 模型生成图片
    """
    try:
        log.info(f"收到图片生成请求: model={req.model}, size={req.size}, async={async_mode}")
        log.info(f"提示词: {req.prompt}")
        if req.reference_image_list:
            log.info(f"参考图列表:\n " + "\n".join(req.reference_image_list))

        import uuid
        task_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        time_str = now.strftime("%m-%d %H:%M")
        run_start = datetime.datetime.now(datetime.timezone.utc)

        # 异步模式下使用 /image，同步下使用 /image?async_mode=false 作为 trace_url
        trace_url = "/image" if async_mode else "/image?async_mode=false"
        trace_id = await http_request_trace_service.create_initial(
            request_url=trace_url,
            http_method="POST",
            request_body=req.model_dump(mode="json"),
            business_type="IMAGE_GEN",
            method_name=f"POST {trace_url}",
            upstream_task_id=task_id,
        )

        # 统一写入初始的基础运行状态，使对比页面和历史看板能及时显示记录
        payload = {
            "id": task_id,
            "prompt": req.prompt,
            "model": req.model.value,
            "size": req.size,
            "ratio": req.ratio,
            "time": time_str,
            "type": "i2i" if req.reference_image_list else "t2i",
            "status": "running",
            "request_id": trace_id,
            "referenceMedia": [{"url": m, "type": "image"} for m in req.reference_image_list] if req.reference_image_list else None,
            "current_run_started_at": run_start,
        }
        await image_history_db_service.upsert_many([payload])

        if not async_mode:
            # 同步模式下的生图执行
            t0 = time.monotonic()
            try:
                image_url = await service_generate_image(
                    prompt=req.prompt,
                    model=req.model.value,
                    size=req.size,
                    reference_image_list=req.reference_image_list
                )
                duration_ms = int((time.monotonic() - t0) * 1000)

                if image_url:
                    prefix = "ai_picture/generated_image"
                    obs_url = image_url
                    try:
                        obs_url = await mirror_remote_url_to_obs(image_url, obs_prefix=prefix)
                        log.info(f"[mirror] updated {task_id} url -> {obs_url}")
                    except Exception as e:
                        log.warning(f"[mirror] failed for {task_id}: {e}")

                    # 更新为成功状态并填充最终的 obs_url/doubao_url
                    await image_history_db_service.upsert_many([
                        {
                            "id": task_id,
                            "obs_url": obs_url,
                            "doubao_url": image_url,
                            "status": "success",
                        }
                    ])

                    await http_request_trace_service.finalize(
                        trace_id,
                        status_code=200,
                        response_body={
                            "success": True,
                            "image_url": image_url,
                            "obs_url": obs_url,
                            "task_id": task_id,
                        },
                        duration_ms=duration_ms,
                        business_success=True,
                    )
                    log.info(f"图片生成成功(同步): {image_url}")
                    return ImageGenerateResponse(success=True, image_url=obs_url, task_id=task_id)
                else:
                    await image_history_db_service.upsert_many([
                        {
                            "id": task_id,
                            "status": "failed",
                            "error": "生成失败，未获取到 URL",
                        }
                    ])
                    await http_request_trace_service.finalize(
                        trace_id,
                        status_code=500,
                        error_message="生成失败，未获取到 URL",
                        response_body={"success": False, "error": "生成失败，未获取到 URL"},
                        duration_ms=duration_ms,
                        business_success=False,
                    )
                    return ImageGenerateResponse(success=False, error="图片生成失败")
            except Exception as e:
                duration_ms = int((time.monotonic() - t0) * 1000)
                log.error(f"图片生成异常(同步): {e}")
                await image_history_db_service.upsert_many([
                    {
                        "id": task_id,
                        "status": "failed",
                        "error": str(e),
                    }
                ])
                await http_request_trace_service.finalize(
                    trace_id,
                    status_code=500,
                    error_message=str(e),
                    response_body={"success": False, "error": str(e)},
                    duration_ms=duration_ms,
                    business_success=False,
                )
                return ImageGenerateResponse(success=False, error=str(e))
        
        # 异步模式：提交至后台进程任务直接返回
        background_tasks.add_task(run_image_generation_job, task_id, req, trace_id)
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
        if req.reference_image_list:
            log.info(f"参考图列表:\n " + "\n".join(req.reference_image_list))
        
        text = await call_doubao_seedtext(
            prompt=req.prompt,
            model=req.model.value,
            system_prompt=req.system_prompt,
            video_duration=req.video_duration,
            reference_image_list=req.reference_image_list,
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
