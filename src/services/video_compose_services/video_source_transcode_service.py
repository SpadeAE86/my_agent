from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import select, text

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_upload_cache import VideoSourceUploadCache
from utils.huawei.mpc_async_client import mpc_client_from_env
from utils.huawei.obs_url import parse_obs_https_url, transcode_output_prefix


def _template_low_high() -> Tuple[int, int]:
    low_raw = (os.getenv("HUAWEI_MPC_TEMPLATE_LOW_ID") or "").strip()
    high_raw = (os.getenv("HUAWEI_MPC_TEMPLATE_HIGH_ID") or "").strip()
    if low_raw and high_raw:
        return int(low_raw), int(high_raw)
    combined = (os.getenv("HUAWEI_MPC_TEMPLATE_IDS") or "").strip()
    if not combined:
        raise ValueError(
            "Set HUAWEI_MPC_TEMPLATE_LOW_ID and HUAWEI_MPC_TEMPLATE_HIGH_ID, "
            "or HUAWEI_MPC_TEMPLATE_IDS with two comma-separated ids (low,high)"
        )
    parts = [p.strip() for p in combined.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("HUAWEI_MPC_TEMPLATE_IDS must contain at least two template ids for low/high")
    return int(parts[0]), int(parts[1])


def _lock_stale_seconds() -> int:
    return max(60, int(os.getenv("HUAWEI_MPC_TRANSCODE_LOCK_STALE_SEC", "3600")))


def _transcode_wait_budget_seconds() -> int:
    return max(60, int(os.getenv("HUAWEI_MPC_POLL_TIMEOUT", "900")))


def _poll_interval_seconds() -> int:
    return max(1, int(os.getenv("HUAWEI_MPC_TRANSCODE_COOPERATE_POLL_SEC", "3")))


def _urls_both_set(row: VideoSourceUploadCache) -> bool:
    return bool((row.low_res_url or "").strip() and (row.high_res_url or "").strip())


async def _run_one_template(
    *,
    bucket: str,
    object_key: str,
    template_id: int,
    output_bucket: str,
) -> str:
    client = mpc_client_from_env()
    if client is None:
        raise RuntimeError("MPC client unavailable (check HUAWEI_CLOUD_AK/SK, HUAWEI_PROJECT_ID)")
    prefix = f"{transcode_output_prefix()}/{template_id}"
    tid = await client.create_transcoding_task(
        input_bucket=bucket,
        input_object=object_key,
        output_bucket=output_bucket,
        output_object_prefix=prefix,
        template_ids=[template_id],
        priority=6,
    )
    if not tid:
        raise RuntimeError(f"MPC create task failed for template {template_id}")
    outs = await client.wait_transcoding_success(
        int(tid),
        obs_bucket_for_url=output_bucket,
        poll_interval=5,
        timeout=int(os.getenv("HUAWEI_MPC_POLL_TIMEOUT", "900")),
    )
    if not outs or not outs[0].url:
        raise RuntimeError(f"MPC no output url for template {template_id}")
    return str(outs[0].url)


async def _mark_transcode_failed(row_id: int, message: str) -> None:
    msg = (message or "").strip()[:2000]
    async with mysql_connector.session_scope() as session:
        r = await session.get(VideoSourceUploadCache, row_id)
        if r is None:
            return
        r.transcode_status = "failed"
        r.transcode_error = msg
        session.add(r)
        await session.commit()


async def _execute_transcode_for_row(row_id: int, obs: str, *, force: bool) -> None:
    """当前进程已持有 processing 锁：补齐 low/high MPC，成功或失败时更新行。"""
    bucket, object_key = parse_obs_https_url(obs)
    out_bucket = (os.getenv("HUAWEI_MPC_OUTPUT_BUCKET") or bucket).strip()
    low_id, high_id = _template_low_high()

    async with mysql_connector.session_scope() as session:
        db_row = await session.get(VideoSourceUploadCache, row_id)
        if db_row is None:
            raise RuntimeError("cache row missing mid-transcode")
        low_u = "" if force else (db_row.low_res_url or "").strip()
        high_u = "" if force else (db_row.high_res_url or "").strip()

    low_out = low_u or None
    high_out = high_u or None
    try:
        if not low_out:
            log.info("transcode low template=%s for %s", low_id, obs[:120])
            low_out = await _run_one_template(
                bucket=bucket,
                object_key=object_key,
                template_id=low_id,
                output_bucket=out_bucket,
            )
        if not high_out:
            log.info("transcode high template=%s for %s", high_id, obs[:120])
            high_out = await _run_one_template(
                bucket=bucket,
                object_key=object_key,
                template_id=high_id,
                output_bucket=out_bucket,
            )
    except Exception as e:
        await _mark_transcode_failed(row_id, str(e))
        raise

    low_f = (low_out or "").strip()
    high_f = (high_out or "").strip()
    if not low_f or not high_f:
        await _mark_transcode_failed(row_id, "transcode finished but missing low/high url")
        raise RuntimeError("transcode did not produce low/high urls")

    async with mysql_connector.session_scope() as session:
        db_row = await session.get(VideoSourceUploadCache, row_id)
        if db_row is None:
            raise RuntimeError("cache row missing after transcode")
        db_row.low_res_url = low_f
        db_row.high_res_url = high_f
        db_row.transcode_status = "ready"
        db_row.transcode_error = None
        db_row.transcode_started_at = None
        session.add(db_row)
        await session.commit()


async def ensure_low_high_for_cache_row(
    row: VideoSourceUploadCache,
    *,
    force: bool = False,
) -> VideoSourceUploadCache:
    """
    若 low_res_url / high_res_url 缺失则走 MPC（低/高各至多一次）。
    使用 ``transcode_status=processing`` 协作锁：同源多请求时仅一路真正转码，其它轮询等待写回结果。
    """
    obs = (row.obs_url or "").strip()
    if not obs:
        raise ValueError("VideoSourceUploadCache.obs_url is empty")
    if row.id is None:
        raise ValueError("VideoSourceUploadCache.id is required (persist row before transcode)")

    row_id = int(row.id)
    if force:
        async with mysql_connector.session_scope() as session:
            db_row = await session.get(VideoSourceUploadCache, row_id)
            if db_row is None:
                raise ValueError("cache row not found")
            db_row.transcode_status = "processing"
            db_row.transcode_error = None
            db_row.transcode_started_at = datetime.now(timezone.utc)
            db_row.low_res_url = None
            db_row.high_res_url = None
            session.add(db_row)
            await session.commit()
        await _execute_transcode_for_row(row_id, obs, force=True)
        async with mysql_connector.session_scope() as session:
            out = await session.get(VideoSourceUploadCache, row_id)
            if out is None:
                raise RuntimeError("cache row missing after force transcode")
            await session.refresh(out)
            return out

    if _urls_both_set(row):
        async with mysql_connector.session_scope() as session:
            db_row = await session.get(VideoSourceUploadCache, row_id)
            if db_row is None:
                return row
            if (db_row.transcode_status or "") != "ready":
                db_row.transcode_status = "ready"
                db_row.transcode_error = None
                session.add(db_row)
                await session.commit()
            await session.refresh(db_row)
            return db_row

    budget = _transcode_wait_budget_seconds()
    stale = _lock_stale_seconds()
    poll = _poll_interval_seconds()
    waited = 0

    claim_sql = text(
        f"""
        UPDATE video_source_upload_cache
        SET transcode_status = 'processing',
            transcode_error = NULL,
            transcode_started_at = UTC_TIMESTAMP(6)
        WHERE id = :rid
          AND (
            transcode_status IS NULL
            OR transcode_status = 'idle'
            OR transcode_status = 'failed'
            OR (
              transcode_status = 'processing'
              AND transcode_started_at IS NOT NULL
              AND transcode_started_at < (UTC_TIMESTAMP(6) - INTERVAL {stale} SECOND)
            )
          )
          AND (
            low_res_url IS NULL OR low_res_url = ''
            OR high_res_url IS NULL OR high_res_url = ''
          )
        """
    )

    while waited <= budget:
        st = "idle"
        rc = 0
        async with mysql_connector.session_scope() as session:
            db_row = await session.get(VideoSourceUploadCache, row_id)
            if db_row is None:
                raise RuntimeError("cache row not found")

            if _urls_both_set(db_row):
                if (db_row.transcode_status or "") != "ready":
                    db_row.transcode_status = "ready"
                    db_row.transcode_error = None
                    session.add(db_row)
                    await session.commit()
                await session.refresh(db_row)
                return db_row

            st = (db_row.transcode_status or "idle").lower()
            if st == "failed":
                raise RuntimeError(db_row.transcode_error or "transcode failed (see transcode_error)")

            if st != "processing":
                res = await session.execute(claim_sql, {"rid": row_id})
                await session.commit()
                rc = int(getattr(res, "rowcount", 0) or 0)

        if st == "processing":
            await asyncio.sleep(poll)
            waited += poll
            continue

        if rc > 0:
            try:
                await _execute_transcode_for_row(row_id, obs, force=False)
            except Exception:
                raise
            async with mysql_connector.session_scope() as session:
                out = await session.get(VideoSourceUploadCache, row_id)
                if out is None:
                    raise RuntimeError("cache row missing after transcode")
                await session.refresh(out)
                return out

        await asyncio.sleep(poll)
        waited += poll

    raise TimeoutError(
        f"waited {waited}s for transcode on video_source_upload_cache.id={row_id} "
        f"(another worker may still be processing; check transcode_status)"
    )


async def get_cache_row_by_obs_url(obs_url: str) -> Optional[VideoSourceUploadCache]:
    u = (obs_url or "").strip()
    if not u:
        return None
    async with mysql_connector.session_scope() as session:
        res = await session.execute(
            select(VideoSourceUploadCache).where(VideoSourceUploadCache.obs_url == u)
        )
        row = res.scalar_one_or_none()
        if row:
            return row
        from os.path import basename

        base = basename(u.split("?", 1)[0])
        if not base:
            return None
        res2 = await session.execute(
            select(VideoSourceUploadCache).where(VideoSourceUploadCache.file_name == base)
        )
        return res2.scalar_one_or_none()


async def ensure_low_high_for_obs_url(obs_url: str, *, force: bool = False) -> Tuple[str, str]:
    """
    保证返回 (low_res_url, high_res_url)。必要时转码并更新或插入缓存行。
    """
    row = await get_cache_row_by_obs_url(obs_url)
    if row is None:
        from os.path import basename

        bn = basename(obs_url.split("?", 1)[0].rstrip("/"))
        row = VideoSourceUploadCache(file_name=bn or "unknown.mp4", obs_url=obs_url.strip())
        async with mysql_connector.session_scope() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
    updated = await ensure_low_high_for_cache_row(row, force=force)
    low = (updated.low_res_url or "").strip()
    high = (updated.high_res_url or "").strip()
    if not low or not high:
        raise RuntimeError("transcode did not produce low/high urls")
    return low, high
