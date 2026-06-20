"""进程启动时统一将「进行中」任务标为失败，对于已提交的视频任务则尝试恢复状态（包含之前因中断被误标为失败的任务）。"""

from __future__ import annotations

from typing import Dict
import time
import asyncio

from infra.logging.logger import logger as log
from sqlmodel import select
from sqlalchemy import or_, and_
from models.sqlmodel.image_history import ImageHistoryCard
from infra.storage.mysql_connector import mysql_connector
from services.video_match_services.video_history_db_service import video_history_db_service
from utils.call_model_utils import get_seedance_task_status
from routers.video import poll_and_finalize_video_task


async def recover_running_video_tasks() -> int:
    """
    启动恢复：检查所有进行中/排队中的视频生成任务，以及之前由于进程中断被误标为失败的任务。
    根据其 doubao_task_id 向豆包查询：
    - 若已成功：拉取结果 URL，镜像并更新 DB 为 success
    - 若已失败：更新 DB 为 failed 并记录错误
    - 若仍在运行：继续启动后台协程轮询该任务
    - 若无 taskId：直接标为 failed
    """
    _VIDEO_TYPES = ("t2v", "i2v")

    async with mysql_connector.session_scope() as session:
        stmt = select(ImageHistoryCard).where(
            ImageHistoryCard.type.in_(list(_VIDEO_TYPES)),
            or_(
                ImageHistoryCard.status.in_(["running", "pending", "processing"]),
                and_(
                    ImageHistoryCard.status == "failed",
                    ImageHistoryCard.error.in_([
                        "服务重启或进程中断，任务未完成",
                        "豆包视频生成失败",
                    ])
                )
            )
        )
        res = await session.execute(stmt)
        running_cards = res.scalars().all()

    recovered_count = 0
    for card in running_cards:
        legacy_id = card.legacy_id
        doubao_task_id = card.taskId
        trace_rid = card.request_id or ""

        if not doubao_task_id:
            # 无 taskId，直接标记为失败
            await video_history_db_service.upsert_many([
                {"id": legacy_id, "status": "failed", "error": "服务中断，且未获取到豆包任务 ID"}
            ])
            if trace_rid:
                try:
                    from services.taskboard_services.http_request_trace_service import http_request_trace_service
                    await http_request_trace_service.finalize(
                        trace_id=trace_rid,
                        status_code=500,
                        error_message="服务中断，且未获取到豆包任务 ID",
                        response_body={"success": False, "error": "服务中断，且未获取到豆包任务 ID"},
                        duration_ms=None,
                        business_success=False,
                    )
                except Exception as _tr_err:
                    log.warning("启动恢复：更新 http_request_traces 失败: %s", _tr_err)
            recovered_count += 1
            continue

        try:
            log.info("启动恢复：发现未完成的视频任务 %s，豆包 taskId 为 %s，正在查询状态...", legacy_id, doubao_task_id)
            status_info = await get_seedance_task_status(doubao_task_id)
            st = status_info.get("status", "")

            if st in ("succeed", "succeeded"):
                video_url = status_info.get("video_url")
                if video_url:
                    log.info("启动恢复：视频任务 %s 已生成成功，开始镜像至 OBS...", legacy_id)
                    from services.media_mirror_service import mirror_remote_url_to_obs
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
                                    "error": None,
                                }
                            ]
                        )
                        if trace_rid:
                            try:
                                from services.taskboard_services.http_request_trace_service import http_request_trace_service
                                await http_request_trace_service.finalize(
                                    trace_id=trace_rid,
                                    status_code=200,
                                    response_body={
                                        "success": True,
                                        "video_url": obs_url,
                                        "doubao_url": video_url,
                                        "task_id": legacy_id,
                                    },
                                    duration_ms=None,
                                    business_success=True,
                                )
                            except Exception as _tr_err:
                                log.warning("启动恢复：更新 http_request_traces 成功状态失败: %s", _tr_err)
                    except Exception as e:
                        log.warning("启动恢复：OBS 镜像失败 %s: %s", legacy_id, e)
                        await video_history_db_service.upsert_many(
                            [
                                {
                                    "id": legacy_id,
                                    "doubao_url": video_url,
                                    "status": "success",
                                    "error": None,
                                }
                            ]
                        )
                        if trace_rid:
                            try:
                                from services.taskboard_services.http_request_trace_service import http_request_trace_service
                                await http_request_trace_service.finalize(
                                    trace_id=trace_rid,
                                    status_code=200,
                                    response_body={
                                        "success": True,
                                        "video_url": video_url,
                                        "task_id": legacy_id,
                                        "mirror_warning": str(e),
                                    },
                                    duration_ms=None,
                                    business_success=True,
                                )
                            except Exception as _tr_err:
                                log.warning("启动恢复：更新 http_request_traces 状态失败: %s", _tr_err)
                else:
                    msg = "豆包任务已成功但未返回视频 URL"
                    await video_history_db_service.upsert_many([
                        {"id": legacy_id, "status": "failed", "error": msg}
                    ])
                    if trace_rid:
                        try:
                            from services.taskboard_services.http_request_trace_service import http_request_trace_service
                            await http_request_trace_service.finalize(
                                trace_id=trace_rid,
                                status_code=500,
                                error_message=msg,
                                response_body={"success": False, "error": msg},
                                duration_ms=None,
                                business_success=False,
                            )
                        except Exception as _tr_err:
                            log.warning("启动恢复：更新 http_request_traces 状态失败: %s", _tr_err)
                recovered_count += 1
            elif st == "failed":
                err_msg = status_info.get("error") or "豆包视频生成失败"
                log.info("启动恢复：视频任务 %s 在云端已失败：%s", legacy_id, err_msg)
                await video_history_db_service.upsert_many([
                    {"id": legacy_id, "status": "failed", "error": err_msg}
                ])
                if trace_rid:
                    try:
                        from services.taskboard_services.http_request_trace_service import http_request_trace_service
                        await http_request_trace_service.finalize(
                            trace_id=trace_rid,
                            status_code=500,
                            error_message=err_msg,
                            response_body={"success": False, "error": err_msg},
                            duration_ms=None,
                            business_success=False,
                        )
                    except Exception as _tr_err:
                        log.warning("启动恢复：更新 http_request_traces 状态失败: %s", _tr_err)
                recovered_count += 1
            else:
                # 仍在运行：在后台重新拉起轮询协程
                log.info("启动恢复：视频任务 %s 仍在豆包云端运行，重新拉起后台轮询...", legacy_id)
                t0 = time.monotonic()
                asyncio.create_task(poll_and_finalize_video_task(legacy_id, doubao_task_id, trace_rid, t0))
        except Exception as e:
            log.error("恢复视频任务 %s 失败: %s", legacy_id, e)

    return recovered_count


async def mark_interrupted_tasks_on_startup(reason: str) -> Dict[str, int]:
    """
    各表独立实现具体 SQL；此处聚合调用并返回影响行数（便于日志）。
    """
    counts: Dict[str, int] = {"image": 0, "video": 0, "video_analysis": 0, "video_match_services": 0}
    from services.media_generate_services.image_history_db_service import image_history_db_service
    from services.video_match_services.video_analysis_db_service import video_analysis_db_service
    from services.video_match_services.video_match_service import mark_interrupted_video_match_jobs_failed

    # 1. 各表 running → failed 恢复
    counts["image"] = await image_history_db_service.mark_interrupted_running_as_failed(reason)
    counts["video"] = await recover_running_video_tasks()
    counts["video_analysis"] = await video_analysis_db_service.mark_interrupted_running_histories_failed(reason)
    counts["video_match_services"] = await mark_interrupted_video_match_jobs_failed(reason)

    total = sum(counts.values())
    if total:
        log.warning(
            "启动恢复: 生图 %s 条已标为失败，视频生成处理 %s 条，视频分析 %s 条标为失败，视频匹配 job %s 条标为失败（原因: %s）",
            counts["image"],
            counts["video"],
            counts["video_analysis"],
            counts["video_match_services"],
            reason[:120],
        )
    return counts
