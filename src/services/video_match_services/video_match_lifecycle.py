
import logging
from typing import Optional, Dict, Any

from sqlalchemy import func, update, delete

from infra.logging.logger import logger as log  # noqa: F401 — used via log.exception in query layer
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow
from services.video_match_services.script_rewrite_service import synthesize_text_to_obs_wav
from services.video_match_services.video_match_query import shot_row_to_api_dict

logger = logging.getLogger(__name__)

async def synthesize_shot_obs_audio(job_id: str, shot_row_id: int) -> Dict[str, Any]:
    """为单条分镜生成阿里云 TTS、上传 OBS，并写入 ``obs_audio_url``（可选更新 ``duration_sec``）。"""
    jid = (job_id or "").strip()
    if not jid:
        return {"success": False, "error": "invalid job_id"}
    if shot_row_id <= 0:
        return {"success": False, "error": "invalid shot_row_id"}

    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMatchShotRow, shot_row_id)
        if row is None or str(row.job_id) != jid:
            return {"success": False, "error": "shot not found"}
        segment = (row.segment_text or "").strip()
        if not segment:
            return {"success": False, "error": "segment_text is empty"}
        order_hint = int(row.shot_order)

    try:
        url, dur = await synthesize_text_to_obs_wav(
            segment,
            obs_project_id=jid,
            tts_tid_suffix=f"s{shot_row_id}_{order_hint}",
        )
    except Exception as e:
        log.exception("synthesize_shot_obs_audio: TTS failed job={} shot={}", jid, shot_row_id)
        return {"success": False, "error": str(e)}

    if not url:
        return {"success": False, "error": "TTS or OBS upload returned empty URL"}

    async with mysql_connector.session_scope() as session:
        row2 = await session.get(VideoMatchShotRow, shot_row_id)
        if row2 is None or str(row2.job_id) != jid:
            return {"success": False, "error": "shot not found after synthesis"}
        row2.obs_audio_url = url
        if dur is not None:
            row2.duration_sec = round(float(dur) + 0.4, 2)
        session.add(row2)
        await session.commit()
        payload = shot_row_to_api_dict(row2)

    return {"success": True, "shot": payload}



async def mark_interrupted_video_match_jobs_failed(reason: str) -> int:
    """进程重启后：口播解析 / 素材检索仍为进行中的 job 标为失败（search 的 pending 表示未发起匹配，不处理）。"""
    msg = (reason or "").strip() or "interrupted"
    n = 0
    async with mysql_connector.session_scope() as session:
        res_p = await session.execute(
            update(VideoMatchJob)
            .where(
                func.lower(func.coalesce(VideoMatchJob.parse_status, "")).in_(
                    ["running", "pending", "processing"]
                )
            )
            .values(parse_status="failed", parse_error=msg)
        )
        n += int(res_p.rowcount or 0)
        res_s = await session.execute(
            update(VideoMatchJob)
            .where(
                func.lower(func.coalesce(VideoMatchJob.search_status, "")).in_(["running", "processing"])
            )
            .values(search_status="failed", search_error=msg)
        )
        n += int(res_s.rowcount or 0)
        await session.commit()
    return n



async def schedule_video_match_job_retry(job_id: str) -> Dict[str, Any]:
    """
    校验失败任务并占用状态（解析重试会清空分镜行）。
    返回 { success, kind: 'parse'|'search', strategy_name? }。
    """
    jid = (job_id or "").strip()
    if not jid:
        return {"success": False, "error": "invalid job_id"}
    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, jid)
        if job is None:
            return {"success": False, "error": "job not found"}
        ps = (job.parse_status or "").lower()
        ss = (job.search_status or "").lower()
        if ps == "running" or ss == "running" or ps == "processing" or ss == "processing":
            return {"success": False, "error": "任务仍在执行中，请稍后再试"}
        if ps == "failed":
            await session.execute(delete(VideoMatchShotRow).where(VideoMatchShotRow.job_id == jid))
            job.parse_status = "running"
            job.parse_error = None
            job.search_status = "pending"
            job.search_error = None
            job.search_total_ms = None
            job.search_strategy_snapshot = None
            session.add(job)
            await session.commit()
            return {"success": True, "kind": "parse"}
        if ps == "done" and ss == "failed":
            snap = job.search_strategy_snapshot
            name = ""
            if isinstance(snap, dict):
                raw = snap.get("name")
                name = str(raw).strip() if raw else ""
            if not name:
                return {
                    "success": False,
                    "error": "缺少历史检索策略，请先在视频匹配页成功发起过一次检索后再重试",
                }
            job.search_status = "running"
            job.search_error = None
            session.add(job)
            await session.commit()
            return {"success": True, "kind": "search", "strategy_name": name}
        return {"success": False, "error": "仅口播转写失败或素材检索失败的任务可重试"}



async def run_video_match_retry_background(
    job_id: str, kind: str, strategy_name: Optional[str] = None
) -> None:
    jid = (job_id or "").strip()
    k = (kind or "").strip()
    try:
        if k == "parse":
            from services.video_match_services.video_match_service import _reparse_video_match_job_core
            await _reparse_video_match_job_core(jid)
        elif k == "search" and (strategy_name or "").strip():
            from services.video_match_services.video_match_service import run_job_search
            await run_job_search(
                jid,
                strategy_name=str(strategy_name).strip(),
                mode="field_aligned_hybrid",
                top_k=5,
            )
        else:
            logger.error("video_match_services retry worker: bad args job=%s kind=%s", jid, k)
    except Exception:
        logger.exception("video_match_services retry background failed job=%s", jid)
