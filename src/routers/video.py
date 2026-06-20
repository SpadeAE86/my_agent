# routers/video.py — 视频生成路由（已对齐图像生成看板）
# 端点:
#   POST /video              — 提交 Seedance 任务，后台协程轮询豆包，完成后 OBS 镜像
#   GET  /video/history      — 从 DB 读取视频历史（type=t2v/i2v）
#   GET  /video/status/{id}  — 查询任务状态（从 DB，不直透豆包）
#   DELETE /video/history/{id}         — 删除单条记录
#   POST /video/history/{id}/retry     — 失败任务重试
#   GET  /video/history/{id}/detail    — 任务看板 HTTP 明细

import asyncio
import datetime
import time
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional, List

from models.pydantic.request import VideoGenerateRequest, SeedanceModel
from utils.call_model_utils import call_doubao_seedance, get_seedance_task_status
from services.video_match_services.video_history_db_service import video_history_db_service
from services.media_mirror_service import mirror_remote_url_to_obs
from services.taskboard_services.http_request_trace_service import http_request_trace_service
from infra.logging.logger import logger as log

video_router = APIRouter(prefix="/video", tags=["video"])


# ---------------------------------------------------------------------------
# History CRUD
# ---------------------------------------------------------------------------

@video_router.get("/history")
async def get_video_history(ids: Optional[str] = None):
    history = await video_history_db_service.list_all(ids=ids)
    return {"success": True, "history": history}


@video_router.delete("/history/{item_id}")
async def delete_video_history_item(item_id: str):
    success = await video_history_db_service.delete_by_id(item_id)
    if success:
        log.info(f"视频历史记录已删除: {item_id}")
        return {"success": True}
    raise HTTPException(status_code=404, detail="Item not found")


@video_router.get("/history/{item_id}/detail")
async def get_video_history_detail(item_id: str):
    """任务看板：合成视频生成 HTTP 明细。"""
    from services.video_match_services.task_detail_service import build_video_gen_task_detail, merge_http_trace_into_detail

    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=404, detail="Item not found")
    row = await video_history_db_service.get_by_id_or_task_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    base = build_video_gen_task_detail(row)
    trace_dict = None
    rid = row.get("request_id")
    if rid:
        trace_dict = await http_request_trace_service.get_dict(str(rid))
    return {"success": True, "detail": merge_http_trace_into_detail(base, trace_dict)}


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

@video_router.get("/status/{task_id}")
async def get_video_status(task_id: str):
    """
    查询视频生成任务状态（从 DB，不直透豆包）。
    task_id 可以是我们给前端的 legacy_id（UUID），也可以是豆包下发的 taskId。
    """
    row = await video_history_db_service.get_by_id_or_task_id(task_id)
    if not row:
        return {"success": False, "error": "Task not found"}

    status = (row.get("status") or "running").lower()
    # 对齐豆包 status 字符串给前端使用
    if status in ("success", "succeed", "succeeded"):
        doubao_status = "succeeded"
    elif status == "failed":
        doubao_status = "failed"
    else:
        doubao_status = "running"

    return {
        "success": True,
        "data": {
            "status": doubao_status,
            "video_url": row.get("url"),
            "error": row.get("error"),
        },
    }


# ---------------------------------------------------------------------------
# 核心后台生成任务
# ---------------------------------------------------------------------------

def _seedance_model_from_stored(model_str: Optional[str]) -> SeedanceModel:
    v = (model_str or "").strip()
    for m in SeedanceModel:
        if m.value == v:
            return m
    log.warning("unknown stored video model %r, fallback to Seedance 2.0", model_str)
    return SeedanceModel.V2_0


def _reference_urls_from_history_row(row: dict, media_type: str) -> Optional[List[str]]:
    """从历史行的 referenceMedia 里按 type 过滤出 URL 列表。"""
    rm = row.get("referenceMedia")
    if not rm or not isinstance(rm, list):
        return None
    urls = []
    for x in rm:
        if isinstance(x, dict) and x.get("type") == media_type:
            u = x.get("url")
            if u:
                urls.append(str(u).strip())
    return urls or None


async def poll_and_finalize_video_task(
    legacy_id: str,
    doubao_task_id: str,
    trace_rid: str,
    start_time_monotonic: float,
) -> None:
    """轮询豆包视频生成状态，完成后镜像到 OBS，并更新 DB 状态与 HTTP Trace。"""
    t0 = start_time_monotonic

    async def _finalize_ok(resp: dict) -> None:
        duration_ms = int((time.monotonic() - t0) * 1000)
        await http_request_trace_service.finalize(
            trace_rid,
            status_code=200,
            response_body=resp,
            duration_ms=duration_ms,
            business_success=True,
        )

    async def _finalize_fail(msg: str, resp: Optional[dict] = None) -> None:
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
        # 轮询豆包任务直到完成（最长 15 分钟）
        max_wait_s = 15 * 60
        poll_interval_s = 10
        elapsed = 0
        video_url: Optional[str] = None
        err_msg: Optional[str] = None

        while elapsed < max_wait_s:
            await asyncio.sleep(poll_interval_s)
            elapsed += poll_interval_s

            status_info = await get_seedance_task_status(doubao_task_id)
            st = status_info.get("status", "")

            if st in ("succeed", "succeeded"):
                video_url = status_info.get("video_url")
                break
            elif st == "failed":
                err_msg = status_info.get("error") or "豆包视频生成失败"
                break

        if err_msg or not video_url:
            msg = err_msg or "超时：视频生成未完成"
            await video_history_db_service.upsert_many(
                [{"id": legacy_id, "status": "failed", "error": msg}]
            )
            await _finalize_fail(msg)
            return

        # OBS 镜像
        try:
            obs_url = await mirror_remote_url_to_obs(
                video_url, obs_prefix="ai_picture/generated_video"
            )
            await video_history_db_service.upsert_many(
                [
                    {
                        "id": legacy_id,
                        "doubao_url": video_url,
                        "obs_url": obs_url,
                        "status": "success",
                    }
                ]
            )
            await _finalize_ok(
                {
                    "success": True,
                    "video_url": obs_url,
                    "doubao_url": video_url,
                    "task_id": legacy_id,
                }
            )
            log.info(f"[video_gen] 完成 legacy={legacy_id} obs_url={obs_url}")
        except Exception as e:
            log.warning(f"[video_gen] OBS mirror 失败 legacy={legacy_id}: {e}")
            await video_history_db_service.upsert_many(
                [
                    {
                        "id": legacy_id,
                        "doubao_url": video_url,
                        "status": "success",
                    }
                ]
            )
            await _finalize_ok(
                {
                    "success": True,
                    "video_url": video_url,
                    "task_id": legacy_id,
                    "mirror_warning": str(e),
                }
            )

    except Exception as e:
        log.error(f"[video_gen] 轮询任务异常 legacy={legacy_id}: {e}")
        await video_history_db_service.upsert_many(
            [{"id": legacy_id, "status": "failed", "error": str(e)}]
        )
        await _finalize_fail(str(e))


async def run_video_generation_job(
    legacy_id: str,
    req: VideoGenerateRequest,
    trace_rid: str,
) -> None:
    """后台协程：提交豆包任务 → 轮询 → OBS 镜像 → 更新 DB。"""
    t0 = time.monotonic()

    async def _finalize_fail(msg: str, resp: Optional[dict] = None) -> None:
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
        # 1. 提交到豆包，得到豆包 task_id
        doubao_task_id = await call_doubao_seedance(
            prompt=req.prompt,
            model=req.model.value,
            resolution=req.resolution,
            ratio=req.ratio,
            duration=req.duration,
            generate_audio=req.generate_audio,
            reference_image_list=req.reference_image_list,
            reference_video_list=req.reference_video_list,
            reference_audio_list=req.reference_audio_list,
        )

        if not doubao_task_id:
            await video_history_db_service.upsert_many(
                [{"id": legacy_id, "status": "failed", "error": "豆包任务提交失败"}]
            )
            await _finalize_fail("豆包任务提交失败")
            return

        # 2. 更新 DB：记录豆包 task_id，状态改为 polling
        await video_history_db_service.upsert_many(
            [{"id": legacy_id, "taskId": doubao_task_id, "status": "running"}]
        )
        log.info(f"[video_gen] legacy={legacy_id} doubao_task_id={doubao_task_id}")

        # 3. 轮询并完成
        await poll_and_finalize_video_task(legacy_id, doubao_task_id, trace_rid, t0)

    except Exception as e:
        log.error(f"[video_gen] 后台任务异常 legacy={legacy_id}: {e}")
        await video_history_db_service.upsert_many(
            [{"id": legacy_id, "status": "failed", "error": str(e)}]
        )
        await _finalize_fail(str(e))


# ---------------------------------------------------------------------------
# 生成入口
# ---------------------------------------------------------------------------

@video_router.post("")
async def generate_video(req: VideoGenerateRequest, background_tasks: BackgroundTasks):
    """
    提交豆包 Seedance 视频生成任务。
    立即返回我们分配的 task_id（UUID），后台协程负责轮询豆包并更新 DB。
    """
    try:
        log.info(
            f"收到视频生成请求: model={req.model}, resolution={req.resolution}, "
            f"ratio={req.ratio}, duration={req.duration}"
        )
        log.info(f"提示词: {req.prompt}")

        is_i2v = bool(req.reference_image_list or req.reference_video_list)
        legacy_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        time_str = now.strftime("%m-%d %H:%M")
        run_start = datetime.datetime.now(datetime.timezone.utc)

        trace_id = await http_request_trace_service.create_initial(
            request_url="/video",
            http_method="POST",
            request_body=req.model_dump(mode="json"),
            business_type="VIDEO_GEN",
            method_name="POST /video",
            upstream_task_id=legacy_id,
        )

        ref_media = None
        if req.reference_image_list:
            ref_media = (ref_media or []) + [
                {"url": u, "type": "image"} for u in req.reference_image_list
            ]
        if req.reference_video_list:
            ref_media = (ref_media or []) + [
                {"url": u, "type": "video"} for u in req.reference_video_list
            ]
        if req.reference_audio_list:
            ref_media = (ref_media or []) + [
                {"url": u, "type": "audio"} for u in req.reference_audio_list
            ]

        await video_history_db_service.upsert_many(
            [
                {
                    "id": legacy_id,
                    "prompt": req.prompt,
                    "model": req.model.value,
                    "resolution": req.resolution,
                    "ratio": req.ratio,
                    "duration": req.duration,
                    "time": time_str,
                    "type": "i2v" if is_i2v else "t2v",
                    "status": "running",
                    "request_id": trace_id,
                    "referenceMedia": ref_media,
                    "current_run_started_at": run_start,
                }
            ]
        )

        background_tasks.add_task(run_video_generation_job, legacy_id, req, trace_id)
        return {"success": True, "task_id": legacy_id}

    except Exception as e:
        log.error(f"视频生成请求异常: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 重试
# ---------------------------------------------------------------------------

@video_router.post("/history/{item_id}/retry")
async def retry_failed_video_generation(item_id: str, background_tasks: BackgroundTasks):
    """仅失败任务：用同一行 legacy_id 重新排队生成视频。"""
    item_id = (item_id or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="invalid id")

    row = await video_history_db_service.get_by_id_or_task_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")

    st = (row.get("status") or "").lower()
    if st != "failed":
        raise HTTPException(status_code=400, detail="仅失败任务可重试")

    prompt = (row.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="记录缺少提示词，无法重试")

    model = _seedance_model_from_stored(row.get("model"))
    try:
        req = VideoGenerateRequest(
            prompt=prompt,
            model=model,
            resolution=row.get("resolution") or "720p",
            ratio=row.get("ratio") or "adaptive",
            duration=row.get("duration") or 5,
            generate_audio=True,
            reference_image_list=_reference_urls_from_history_row(row, "image"),
            reference_video_list=_reference_urls_from_history_row(row, "video"),
            reference_audio_list=_reference_urls_from_history_row(row, "audio"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法构造视频生成请求: {e}") from e

    rid_str = str(row.get("id") or item_id)
    run_start = datetime.datetime.now(datetime.timezone.utc)
    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d %H:%M")

    trace_id = await http_request_trace_service.create_initial(
        request_url=f"/video/history/{rid_str}/retry",
        http_method="POST",
        request_body={"retry_of": rid_str, **req.model_dump(mode="json")},
        business_type="VIDEO_GEN",
        method_name="POST /video/history/retry",
        upstream_task_id=rid_str,
    )

    await video_history_db_service.upsert_many(
        [
            {
                "id": rid_str,
                "status": "running",
                "error": None,
                "doubao_url": None,
                "obs_url": None,
                "taskId": None,
                "request_id": trace_id,
                "time": time_str,
                "current_run_started_at": run_start,
            }
        ]
    )

    background_tasks.add_task(run_video_generation_job, rid_str, req, trace_id)
    return {"success": True, "task_id": rid_str}
