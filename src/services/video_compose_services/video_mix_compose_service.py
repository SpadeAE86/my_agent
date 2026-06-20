from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from typing import Any, Dict, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from config.config import MY_CONFIG
from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from infra.storage.mix_overall_time_mysql import get_mix_overall_time_engine
from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow
from models.sqlmodel.video_mix_compose import VideoMixComposeJob
from services.video_compose_services.video_mix_timeline_builder import (
    build_mixed_video_request_from_shots,
    build_srt_from_match_shots,
    collect_unique_source_obs_urls,
)
from services.video_compose_services.video_source_transcode_service import ensure_low_high_for_obs_url
from utils.huawei.obs_url import mix_obs_object_path


def _truthy_env(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def mix_compose_settings() -> Dict[str, Any]:
    cfg = dict(MY_CONFIG.get("mix_compose") or {})
    env_mock = _truthy_env("MIX_COMPOSE_MOCK")
    if env_mock is not None:
        cfg["mock"] = env_mock
    base = (os.getenv("MIX_COMPOSE_API_BASE") or cfg.get("api_base") or "").rstrip("/")
    if base:
        cfg["api_base"] = base
    cfg.setdefault("mock", True)
    cfg.setdefault("api_base", "http://127.0.0.1:8000")
    cfg.setdefault("edit_path", "/api/v1/video/edit")
    cfg.setdefault("poll_interval_sec", 5)
    cfg.setdefault("poll_timeout_sec", 900)
    cfg.setdefault(
        "mock_result_obs_url",
        "aigc/dev/mock_mixed_output_placeholder.mp4",
    )
    cfg.setdefault("dump_request_json", "")
    cfg.setdefault("overall_time_mysql_env", "test")
    return cfg


async def _allocate_three_digit_biz_id(session: AsyncSession) -> str:
    """与日常测试环境六位 biz_id 区分：本链路使用 100–999，且在 ``video_mix_compose_job`` 内唯一。"""
    for _ in range(400):
        cand = str(random.randint(100, 999))
        res = await session.execute(select(VideoMixComposeJob.id).where(VideoMixComposeJob.biz_id == cand))
        if res.first() is None:
            return cand
    raise RuntimeError("could not allocate unique 3-digit biz_id for video_mix_compose_job")


def normalize_mix_worker_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """混剪 Worker：路径无域名；当前不传转场。"""
    out = dict(d)
    out.pop("transition_config", None)
    for k in ("obs_video_path_list", "obs_audio_path_list", "obs_bgm_path_list"):
        if k in out and isinstance(out[k], list):
            out[k] = [mix_obs_object_path(str(x)) if x is not None else "" for x in out[k]]
    osp = out.get("obs_sticker_path_list")
    if isinstance(osp, list):
        out["obs_sticker_path_list"] = [mix_obs_object_path(str(x)) for x in osp if x]
    return out


def _resolve_dump_request_path(template: str, compose_id: str, biz_id: str) -> str:
    s = (template or "").strip()
    if not s:
        return ""
    expanded = s.replace("{compose_id}", compose_id).replace("{biz_id}", biz_id)
    if expanded.endswith(("/", "\\")):
        return os.path.join(expanded.rstrip("/\\"), f"mix_compose_request_{compose_id}.json")
    return expanded


def _write_mix_request_json_file(path: str, payload: Dict[str, Any]) -> None:
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _maybe_dump_mix_request_json(
    cfg: Dict[str, Any],
    *,
    compose_id: str,
    biz_id: str,
    req_payload: Dict[str, Any],
) -> None:
    raw = (os.getenv("MIX_COMPOSE_DUMP_REQUEST_JSON") or "").strip() or str(
        cfg.get("dump_request_json") or ""
    ).strip()
    if not raw:
        return
    out_path = _resolve_dump_request_path(raw, compose_id, biz_id)
    if not out_path:
        return
    try:
        _write_mix_request_json_file(out_path, req_payload)
        log.info("wrote mix request JSON for manual POST (absolute path): {}", os.path.abspath(out_path))
    except OSError as e:
        log.warning("failed to write mix request JSON to {}: {}", out_path, e)


async def _patch_compose_job(compose_id: str, **fields: Any) -> None:
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMixComposeJob, compose_id)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        session.add(row)
        await session.commit()


async def fetch_mix_result_obs_url(biz_id: str) -> Optional[str]:
    q = text("SELECT output_url FROM mix_video_overall_time WHERE biz_id = :b LIMIT 1")
    engine = await get_mix_overall_time_engine()
    async with engine.connect() as conn:
        res = await conn.execute(q, {"b": biz_id})
        first = res.first()
        if not first:
            return None
        val = first[0]
        if val is None:
            return None
        s = str(val).strip()
        return s or None


async def mock_write_mix_overall_time(biz_id: str, obs_url: str) -> None:
    path_or_url = (obs_url or "").strip()
    stored = mix_obs_object_path(path_or_url) if path_or_url else ""
    if not stored:
        stored = path_or_url
    upd = text("UPDATE mix_video_overall_time SET output_url = :u WHERE biz_id = :b")
    ins = text("INSERT INTO mix_video_overall_time (biz_id, output_url) VALUES (:b, :u)")
    engine = await get_mix_overall_time_engine()
    async with engine.begin() as conn:
        res = await conn.execute(upd, {"u": stored, "b": biz_id})
        rc = getattr(res, "rowcount", None)
        if rc:
            return
        await conn.execute(ins, {"b": biz_id, "u": stored})


async def _run_compose_pipeline(compose_id: str, mix_mock_override: Optional[bool] = None) -> None:
    try:
        cfg = mix_compose_settings()
        if mix_mock_override is not None:
            cfg = {**cfg, "mock": bool(mix_mock_override)}
        job_id: Optional[str] = None
        biz_id: Optional[str] = None

        prefer_srt_flag = False
        async with mysql_connector.session_scope() as session:
            job_row = await session.get(VideoMixComposeJob, compose_id)
            if job_row is None:
                log.error("compose pipeline: missing job {}", compose_id)
                return
            prefer_srt_flag = bool(job_row.prefer_srt)
            job_row.status = "transcoding"
            session.add(job_row)
            await session.commit()
            job_id = job_row.video_match_job_id
            biz_id = job_row.biz_id

        if not job_id or not biz_id:
            await _patch_compose_job(
                compose_id,
                status="failed",
                error_message="missing video_match_job_id or biz_id",
            )
            return

        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoMatchShotRow)
                .where(VideoMatchShotRow.job_id == job_id)
                .order_by(VideoMatchShotRow.shot_order)
            )
            shots = list(res.scalars().all())

        if not shots:
            await _patch_compose_job(compose_id, status="failed", error_message="no shot rows")
            return

        src_urls = collect_unique_source_obs_urls(shots)
        if not src_urls:
            await _patch_compose_job(compose_id, status="failed", error_message="no source video urls in shots")
            return

        mapping: Dict[str, str] = {}
        for u in src_urls:
            try:
                _, high = await ensure_low_high_for_obs_url(u, force=False)
                mapping[u] = high
            except Exception as e:
                log.warning("transcode ensure failed for {}: {}", u[:120], e)
                await _patch_compose_job(
                    compose_id,
                    status="failed",
                    error_message=f"transcode failed for source: {e!s}"[:2000],
                )
                return

        await _patch_compose_job(compose_id, status="building")

        try:
            biz_int = int(str(biz_id).strip())
        except ValueError:
            await _patch_compose_job(compose_id, status="failed", error_message="invalid biz_id on compose job")
            return
        if not (100 <= biz_int <= 999):
            await _patch_compose_job(
                compose_id,
                status="failed",
                error_message=f"biz_id must be 3-digit (100–999), got {biz_int}",
            )
            return

        try:
            mix_req = build_mixed_video_request_from_shots(
                shots,
                mapping,
                biz_id=biz_int,
                include_cap_config=not prefer_srt_flag,
            )
        except Exception as e:
            await _patch_compose_job(compose_id, status="failed", error_message=f"timeline build: {e!s}"[:2000])
            return

        req_payload = mix_req.model_dump(exclude_none=True, mode="json")
        req_payload = normalize_mix_worker_payload(req_payload)
        _maybe_dump_mix_request_json(
            cfg,
            compose_id=compose_id,
            biz_id=biz_id,
            req_payload=req_payload,
        )
        await _patch_compose_job(
            compose_id,
            status="submitting",
            request_json=req_payload,
        )

        mock_mode = bool(cfg.get("mock"))
        if mock_mode:
            mock_url = str(cfg.get("mock_result_obs_url") or "").strip()
            try:
                await mock_write_mix_overall_time(biz_id, mock_url)
            except Exception as e:
                log.exception("mock mix_video_overall_time write failed")
                await _patch_compose_job(
                    compose_id,
                    status="failed",
                    error_message=f"mock mix table write failed: {e!s}"[:2000],
                )
                return
        else:
            api_base = str(cfg.get("api_base") or "").rstrip("/")
            path = str(cfg.get("edit_path") or "/api/v1/video/edit")
            if not api_base:
                await _patch_compose_job(compose_id, status="failed", error_message="mix_compose.api_base empty")
                return
            post_url = f"{api_base}{path if path.startswith('/') else '/' + path}"
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(post_url, json=req_payload)
                    if r.status_code >= 400:
                        body_snip = (r.text or "")[:800]
                        log.warning(
                            "mix upstream rejected POST {} -> HTTP {} (first bytes of body): {}",
                            post_url,
                            r.status_code,
                            body_snip,
                        )
                        await _patch_compose_job(
                            compose_id,
                            status="failed",
                            error_message=f"mix API HTTP {r.status_code}: {body_snip}"[:2000],
                        )
                        return
            except Exception as e:
                log.warning("mix upstream POST {} failed: {}", post_url, e)
                await _patch_compose_job(
                    compose_id,
                    status="failed",
                    error_message=f"mix API request failed: {e!s}"[:2000],
                )
                return

        await _patch_compose_job(compose_id, status="processing")

        interval = float(cfg.get("poll_interval_sec") or 5)
        timeout = float(cfg.get("poll_timeout_sec") or 900)
        waited = 0.0
        while waited <= timeout:
            url = await fetch_mix_result_obs_url(biz_id)
            if url:
                srt_out: Optional[str] = None
                if prefer_srt_flag:
                    try:
                        srt_out = build_srt_from_match_shots(shots)
                    except Exception as e:
                        log.warning("build_srt_from_match_shots failed: {}", e)
                        srt_out = ""
                await _patch_compose_job(
                    compose_id,
                    status="done",
                    result_obs_url=url,
                    error_message=None,
                    result_srt_text=srt_out,
                )
                return
            await asyncio.sleep(interval)
            waited += interval

        await _patch_compose_job(
            compose_id,
            status="failed",
            error_message=f"poll timeout after {timeout}s",
        )
    except Exception as e:
        log.exception("compose pipeline failed compose_id={}", compose_id)
        await _patch_compose_job(
            compose_id,
            status="failed",
            error_message=str(e)[:2000],
        )


async def start_mix_compose_for_job(
    job_id: str,
    mix_mock: Optional[bool] = None,
    *,
    prefer_srt: bool = False,
) -> Dict[str, Any]:
    compose_id = str(uuid.uuid4())

    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        if job is None:
            raise ValueError("job not found")
        biz_id = await _allocate_three_digit_biz_id(session)
        row = VideoMixComposeJob(
            id=compose_id,
            biz_id=biz_id,
            video_match_job_id=job_id,
            status="pending",
            prefer_srt=bool(prefer_srt),
        )
        session.add(row)
        await session.commit()

    asyncio.create_task(_run_compose_pipeline(compose_id, mix_mock_override=mix_mock))
    return {"compose_id": compose_id, "biz_id": int(biz_id), "status": "pending", "prefer_srt": bool(prefer_srt)}


async def get_mix_compose_job(compose_id: str) -> Optional[Dict[str, Any]]:
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMixComposeJob, compose_id)
        if row is None:
            return None
        bid = row.biz_id or ""
        biz_out: Any = int(bid) if bid.isdigit() else bid
        return {
            "compose_id": row.id,
            "biz_id": biz_out,
            "video_match_job_id": row.video_match_job_id,
            "status": row.status,
            "error_message": row.error_message,
            "result_obs_url": row.result_obs_url,
            "prefer_srt": bool(row.prefer_srt),
            "result_srt_text": row.result_srt_text,
            "request_json": row.request_json,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
