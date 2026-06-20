from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sqlmodel import select

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextIndexTagsEnvelope
from models.sqlmodel.video_analysis import VideoAnalysisSearchStrategy
from models.sqlmodel.video_material_match import VideoMaterialMatchHistory
from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow
from services.taskboard_services.http_request_trace_service import http_request_trace_service
from services.video_match_services.script_match_query_builder import INDEX_NAME
from services.video_match_services.script_match_service import match_script_tags_segments
from services.video_match_services.video_match_http_trace import (
    hits_for_db_with_truncated_explain,
    opensearch_body_for_debug_log,
    trace_request_body_for_shot_search,
    trace_response_top_hits_with_explain,
    truncate_for_trace,
)
from services.video_match_services.script_rewrite_service import (
    rewrite_script_to_storyboard_and_tags,
    rewrite_storyboard_to_tags,
)
from utils.frame_orientation import infer_frame_orientation

from services.video_match_services.video_match_query import (
    get_job_payload, get_shot_match_detail, get_material_match_board_detail, _enrich_hits_with_resolved_urls, _top5_video_urls_from_hits, _mock_response_payload
)
from services.video_match_services.video_match_lifecycle import (
    run_video_match_retry_background, schedule_video_match_job_retry, synthesize_shot_obs_audio
)

# 后续「每分镜 OpenSearch 匹配」时在此使用 asyncio.Semaphore 限制并发
MATCH_CONCURRENCY = 4


def _tags_by_seg_id(tags: SeedtextIndexTagsEnvelope) -> Dict[int, Any]:
    return {seg.id: seg for seg in tags.segment_result}


def _resolve_tag_segment(
    tags: SeedtextIndexTagsEnvelope,
      storyboard_seg_id: int,
      order: int,
):
    by_id = _tags_by_seg_id(tags)
    if storyboard_seg_id in by_id:
        return by_id[storyboard_seg_id]
    if order < len(tags.segment_result):
        return tags.segment_result[order]
    return None


def _norm_job_frame_orientation(val: Optional[str]) -> Optional[str]:
    s = (val or "").strip()
    if s in ("横屏", "竖屏"):
        return s
    return None


def _merge_job_constraints_into_segment_tags(
    base: Dict[str, Any],
    *,
    car_model: Optional[str],
    frame_size: Optional[str],
    frame_orientation: Optional[str] = None,
) -> Dict[str, Any]:
    """任务表单约束：写入每镜 tags_json，检索时 frame_size / frame_orientation / car_model 参与 bool.filter（AND）。"""
    out = dict(base) if base else {}
    cm = (car_model or "").strip()
    if cm:
        out["car_model"] = cm
    fs = (frame_size or "").strip()
    if fs and fs != "未知":
        out["frame_size"] = fs
    fo = _norm_job_frame_orientation(frame_orientation)
    if fo:
        out["frame_orientation"] = fo
    elif fs and fs != "未知":
        inf = infer_frame_orientation(fs)
        if inf and inf != "未知":
            out["frame_orientation"] = inf
    return out


async def rematch_video_match_shot(job_id: str, shot_row_id: int) -> Dict[str, Any]:
    """
    单条分镜重新跑 OpenSearch 匹配（需 job 已有 search_strategy_snapshot，通常需先成功跑过整 job 匹配）。
    """
    jid = str(job_id or "").strip()
    try:
        sid = int(shot_row_id)
    except (TypeError, ValueError):
        sid = 0
    if not jid or sid <= 0:
        return {"success": False, "error": "invalid id"}

    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, jid)
        if job is None:
            return {"success": False, "error": "job not found"}
        if job.parse_status != "done":
            return {"success": False, "error": "parse not completed"}
        snap = job.search_strategy_snapshot
        if not isinstance(snap, dict) or not str(snap.get("name") or "").strip():
            return {
                "success": False,
                "error": "缺少检索策略快照，请先在视频匹配工作台完成一次整 job 素材匹配",
            }
        strategy_name = str(snap.get("name")).strip()
        row = await session.get(VideoMatchShotRow, sid)
        if row is None or str(row.job_id) != jid:
            return {"success": False, "error": "shot not found"}
        if not (row.tags_json or {}):
            return {"success": False, "error": "分镜缺少标签，无法检索"}
        row.search_status = "running"
        row.search_request_id = None
        row.match_id = None
        row.top1_obs_url = None
        row.match_top_hits_json = None
        row.match_elapsed_ms = None
        session.add(row)
        await session.commit()

        seg = dict(row.tags_json or {})
        if getattr(row, "search_tokens_json", None):
            seg["_search_tokens_json"] = row.search_tokens_json
        ws = (job.workspace or "v1").strip()
        shot_ver = "v2" if ws == "v2" else "v1"

    strategy = await _load_strategy_by_name(strategy_name)
    if strategy is None:
        async with mysql_connector.session_scope() as session:
            row2 = await session.get(VideoMatchShotRow, sid)
            if row2:
                row2.search_status = "failed"
                session.add(row2)
                await session.commit()
        return {"success": False, "error": f"strategy not found: {strategy_name!r}"}

    bw = float(strategy.bm25_weight or 0.3)
    vw = float(strategy.vector_weight or 0.7)
    den = bw + vw or 1.0
    bm25_f = bw / den
    vec_f = vw / den
    mode = "field_aligned_hybrid"
    top_k = 5

    async def persist_one(_idx: int, m: Dict[str, Any]) -> None:
        top_hits_raw = m.get("top_hits") or []
        enriched = await _enrich_hits_with_resolved_urls(top_hits_raw, shot_ver)
        to_store = hits_for_db_with_truncated_explain(enriched)
        urls5 = _top5_video_urls_from_hits(enriched)
        match_ok = bool(urls5)
        top1 = urls5[0] if urls5 else None
        elapsed = float(m.get("elapsed_ms") or 0)
        body_for_trace = trace_request_body_for_shot_search(m, rematch_single_shot=True)
        trace_rid = await http_request_trace_service.create_initial(
            request_url=f"/opensearch/{INDEX_NAME}/_search",
            http_method="POST",
            method_name="POST /opensearch/_search",
            business_type="VIDEO_MATCH_SHOT_SEARCH",
            business_id=str(sid),
            upstream_task_id=jid,
            request_body=body_for_trace,
        )
        try:
            async with mysql_connector.session_scope() as session:
                db_row = await session.get(VideoMatchShotRow, sid)
                if db_row is None:
                    await http_request_trace_service.finalize(
                        trace_rid,
                        status_code=500,
                        error_message="shot row missing after search",
                        business_success=False,
                    )
                    return
                seg_preview = (db_row.segment_text or "").strip()[:512] or None
                hist = VideoMaterialMatchHistory(
                    request_id=trace_rid,
                    source="video_match_shot",
                    workspace=ws or None,
                    status="running",
                    video_match_job_id=jid,
                    video_match_shot_row_id=sid,
                    query_preview=seg_preview,
                    strategy_snapshot=snap if isinstance(snap, dict) else None,
                )
                session.add(hist)
                await session.flush()
                await session.refresh(hist)
                match_hist_id = hist.id

                db_row.match_top_hits_json = to_store
                db_row.match_elapsed_ms = elapsed
                db_row.search_status = "done" if match_ok else "failed"
                db_row.top1_obs_url = top1
                db_row.search_request_id = trace_rid
                db_row.match_id = match_hist_id
                session.add(db_row)
                await session.flush()

                # Query all shot rows under this job to calculate overall job search status
                res_sr = await session.execute(
                    select(VideoMatchShotRow).where(VideoMatchShotRow.job_id == jid)
                )
                shot_rows = list(res_sr.scalars().all())
                n_fail = sum(1 for sr in shot_rows if (sr.search_status or "").lower() == "failed")
                n_running = sum(1 for sr in shot_rows if (sr.search_status or "").lower() in ("running", "pending"))
                
                if job:
                    if n_running > 0:
                        job.search_status = "running"
                        job.search_error = None
                    elif n_fail > 0:
                        job.search_status = "failed"
                        job.search_error = (
                            f"{n_fail} 条分镜素材匹配失败（无 OpenSearch 命中或无法解析出有效视频地址 / Top5 为空）"
                        )
                    else:
                        job.search_status = "done"
                        job.search_error = None
                    session.add(job)

                await session.commit()
        except Exception:
            log.exception("video_match_services rematch persist DB failed job={} row={}", jid, sid)
            await http_request_trace_service.finalize(
                trace_rid,
                status_code=500,
                error_message="persist rematch database error",
                business_success=False,
            )
            raise

        shot_ord = 0
        async with mysql_connector.session_scope() as session:
            rord = await session.get(VideoMatchShotRow, sid)
            if rord is not None:
                shot_ord = int(rord.shot_order)
        resp_summary = {
            "hit_count": len(top_hits_raw),
            "top_history_ids": [h.get("history_id") for h in top_hits_raw[:5]],
            "elapsed_ms": elapsed,
            "shot_order": shot_ord,
            "rematch": True,
            "match_ok": match_ok,
            "top5_nonempty": match_ok,
            "top_hits_explain": trace_response_top_hits_with_explain(enriched),
        }
        await http_request_trace_service.finalize(
            trace_rid,
            status_code=200,
            response_body=truncate_for_trace(resp_summary, max_bytes=200_000),
            business_success=match_ok,
            duration_ms=int(elapsed) if elapsed else None,
        )
        try:
            async with mysql_connector.session_scope() as session:
                h = await session.get(VideoMaterialMatchHistory, match_hist_id)
                if h:
                    h.status = "done" if match_ok else "failed"
                    h.hit_count = len(top_hits_raw)
                    h.top1_obs_url = top1
                    h.elapsed_ms = elapsed
                    h.error_message = None if match_ok else "无有效命中或无法解析视频地址"
                    session.add(h)
                    await session.commit()
        except Exception:
            log.warning("video_match_services rematch finalize material_match_history failed id={}", match_hist_id)
        log.debug(
            "video_match_services rematch opensearch_body job={} shot={} body={}",
            jid, shot_ord, opensearch_body_for_debug_log(m),
        )
        _r_top1_name = top1.split("/")[-1] if top1 else "—"
        _r_top5_lines = "\n".join(
            "  [{}] {}".format(i + 1, u.split("/")[-1])
            for i, u in enumerate(urls5)
        ) or "  (空)"
        log.info(
            "video_match_services rematch job={} shot={} row={} hits={} ok={} elapsed={:.0f}ms\n"
            "  top1: {}\n"
            "  top5:\n{}",
            jid, shot_ord, sid, len(top_hits_raw), match_ok, elapsed, _r_top1_name, _r_top5_lines,
        )

    try:
        await match_script_tags_segments(
            [seg],
            top_k=int(top_k),
            mode=mode,
            shot_cards_version=shot_ver,
            concurrency=1,
            bm25_factor=bm25_f,
            vector_factor=vec_f,
            use_rrf=bool(strategy.use_rrf),
            with_timings=True,
            on_segment_done=persist_one,
        )
    except Exception as e:
        log.exception("video_match_services rematch shot failed: {}", e)
        async with mysql_connector.session_scope() as session:
            row3 = await session.get(VideoMatchShotRow, sid)
            if row3:
                row3.search_status = "failed"
                session.add(row3)
                await session.commit()
        return {"success": False, "error": str(e)}

    out = await get_shot_match_detail(jid, sid)
    if out:
        shot = out.get("shot")
        if isinstance(shot, dict) and (shot.get("search_status") or "").lower() == "failed":
            out["success"] = False
            out["error"] = "本分镜无有效素材命中（Top5 为空或无法解析视频地址）"
        return out
    return {"success": False, "error": "rematch ok but failed to load shot detail"}


async def _load_strategy_by_name(name: str) -> Optional[VideoAnalysisSearchStrategy]:
    n = (name or "").strip()
    if not n:
        return None
    async with mysql_connector.session_scope() as session:
        res = await session.execute(
            select(VideoAnalysisSearchStrategy).where(VideoAnalysisSearchStrategy.name == n)
        )
        return res.scalars().first()


async def run_job_search(
    job_id: str,
    strategy_name: str,
    mode: str = "field_aligned_hybrid",
    top_k: int = 5,
    enable_road_run_fallback: bool = True,
) -> Dict[str, Any]:
    strategy_name = (strategy_name or "").strip()
    if not strategy_name:
        return {"success": False, "error": "strategy_name is required"}

    strategy = await _load_strategy_by_name(strategy_name)
    if strategy is None:
        return {"success": False, "error": f"strategy not found: {strategy_name!r}"}

    bw = float(strategy.bm25_weight or 0.3)
    vw = float(strategy.vector_weight or 0.7)
    den = bw + vw or 1.0
    bm25_f = bw / den
    vec_f = vw / den

    snapshot: Dict[str, Any] = {
        "name": strategy.name,
        "use_rrf": bool(strategy.use_rrf),
        "bm25_weight": strategy.bm25_weight,
        "vector_weight": strategy.vector_weight,
        "text_weights": strategy.text_weights,
        "vector_weights": strategy.vector_weights,
    }
    if strategy.text_weights or strategy.vector_weights:
        log.info(
            "video_match_services search: strategy {} has field-level weights; script_match uses macro bm25/vector only for now",
            strategy.name,
        )

    row_ids: List[int] = []
    shot_ver = "v1"
    rows: List[VideoMatchShotRow] = []

    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        if job is None:
            return {"success": False, "error": "job not found"}
        if job.parse_status != "done":
            return {"success": False, "error": "parse not completed"}
        res = await session.execute(
            select(VideoMatchShotRow)
            .where(VideoMatchShotRow.job_id == job_id)
            .order_by(VideoMatchShotRow.shot_order)
        )
        rows = list(res.scalars().all())
        if not rows:
            return {"success": False, "error": "no shot rows"}
        for r in rows:
            if not (r.tags_json or {}):
                return {"success": False, "error": f"shot_order={r.shot_order} missing tags_json"}
        missing_ids = [r for r in rows if r.id is None]
        if missing_ids:
            return {"success": False, "error": "shot rows missing ids (database error)"}

        row_ids = [int(r.id) for r in rows]
        ws = (job.workspace or "v1").strip()
        shot_ver = "v2" if ws == "v2" else "v1"
        job_workspace_for_hist = ws

        job.search_status = "running"
        job.search_error = None
        job.search_strategy_snapshot = snapshot
        # 整 job 重跑匹配：所有分镜进入「匹配中」，并清空上轮结果，避免前端仍显示旧 Top5/成功态
        for r in rows:
            r.search_status = "running"
            r.top1_obs_url = None
            r.match_top_hits_json = None
            r.match_elapsed_ms = None
            r.search_request_id = None
            r.match_id = None
            session.add(r)
        session.add(job)
        await session.commit()

    log.info(
        "video_match_services search start job={} shot_count={} strategy={} top_k={} shot_cards_version={}",
        job_id,
        len(rows),
        strategy_name,
        top_k,
        shot_ver,
    )

    segments = []
    for r in rows:
        seg = dict(r.tags_json or {})
        if getattr(r, "search_tokens_json", None):
            seg["_search_tokens_json"] = r.search_tokens_json
        segments.append(seg)

    async def persist_shot(idx: int, m: Dict[str, Any]) -> None:
        if idx < 0 or idx >= len(row_ids):
            return
        row_id = row_ids[idx]
        shot_ord = rows[idx].shot_order
        top_hits_raw = m.get("top_hits") or []
        enriched = await _enrich_hits_with_resolved_urls(top_hits_raw, shot_ver)
        to_store = hits_for_db_with_truncated_explain(enriched)
        urls5 = _top5_video_urls_from_hits(enriched)
        match_ok = bool(urls5)
        top1 = urls5[0] if urls5 else None
        elapsed = float(m.get("elapsed_ms") or 0)

        body_for_trace = trace_request_body_for_shot_search(m)
        trace_rid = await http_request_trace_service.create_initial(
            request_url=f"/opensearch/{INDEX_NAME}/_search",
            http_method="POST",
            method_name="POST /opensearch/_search",
            business_type="VIDEO_MATCH_SHOT_SEARCH",
            business_id=str(row_id),
            upstream_task_id=job_id,
            request_body=body_for_trace,
        )
        seg_preview = (rows[idx].segment_text or "").strip()[:512] or None
        try:
            async with mysql_connector.session_scope() as session:
                row = await session.get(VideoMatchShotRow, row_id)
                if row is None:
                    await http_request_trace_service.finalize(
                        trace_rid,
                        status_code=500,
                        error_message="shot row missing after search",
                        business_success=False,
                    )
                    return
                hist = VideoMaterialMatchHistory(
                    request_id=trace_rid,
                    source="video_match_shot",
                    workspace=job_workspace_for_hist or None,
                    status="running",
                    video_match_job_id=job_id,
                    video_match_shot_row_id=row_id,
                    query_preview=seg_preview,
                    strategy_snapshot=snapshot,
                    enable_road_run_fallback=enable_road_run_fallback,
                )
                session.add(hist)
                await session.flush()
                await session.refresh(hist)
                match_hist_id = hist.id

                row.match_top_hits_json = to_store
                row.match_elapsed_ms = elapsed
                row.search_status = "done" if match_ok else "failed"
                row.top1_obs_url = top1
                row.search_request_id = trace_rid
                row.match_id = match_hist_id
                session.add(row)
                await session.commit()
        except Exception:
            log.exception("video_match_services persist_shot DB failed job={} row={}", job_id, row_id)
            await http_request_trace_service.finalize(
                trace_rid,
                status_code=500,
                error_message="persist_shot database error",
                business_success=False,
            )
            raise

        resp_summary = {
            "hit_count": len(top_hits_raw),
            "top_history_ids": [h.get("history_id") for h in top_hits_raw[:5]],
            "elapsed_ms": elapsed,
            "shot_order": shot_ord,
            "match_ok": match_ok,
            "top5_nonempty": match_ok,
            "top_hits_explain": trace_response_top_hits_with_explain(enriched),
        }
        await http_request_trace_service.finalize(
            trace_rid,
            status_code=200,
            response_body=truncate_for_trace(resp_summary, max_bytes=200_000),
            business_success=match_ok,
            duration_ms=int(elapsed) if elapsed else None,
        )
        try:
            async with mysql_connector.session_scope() as session:
                h = await session.get(VideoMaterialMatchHistory, match_hist_id)
                if h:
                    h.status = "done" if match_ok else "failed"
                    h.hit_count = len(top_hits_raw)
                    h.top1_obs_url = top1
                    h.elapsed_ms = elapsed
                    h.error_message = None if match_ok else "无有效命中或无法解析视频地址"
                    session.add(h)
                    await session.commit()
        except Exception:
            log.warning("video_match_services finalize material_match_history failed id={}", match_hist_id)
        log.debug(
            "video_match_services shot_search opensearch_body job={} shot={} body={}",
            job_id, shot_ord, opensearch_body_for_debug_log(m),
        )
        top1_name = top1.split("/")[-1] if top1 else "—"
        top5_lines = "\n".join(
            "  [{}] {}".format(i + 1, u.split("/")[-1])
            for i, u in enumerate(urls5)
        ) or "  (空)"
        log.info(
            "video_match_services shot_search job={} shot={} row={} hits={} ok={} elapsed={:.0f}ms\n"
            "  top1: {}\n"
            "  top5:\n{}",
            job_id, shot_ord, row_id, len(top_hits_raw), match_ok, elapsed, top1_name, top5_lines,
        )

    t_wall0 = time.perf_counter()
    try:
        matches = await match_script_tags_segments(
            segments,
            top_k=int(top_k),
            mode=mode,
            shot_cards_version=shot_ver,
            concurrency=MATCH_CONCURRENCY,
            bm25_factor=bm25_f,
            vector_factor=vec_f,
            use_rrf=bool(strategy.use_rrf),
            with_timings=True,
            enable_road_run_fallback=enable_road_run_fallback,
            on_segment_done=persist_shot,
        )
    except Exception as e:
        log.exception("video_match_services search failed: {}", e)
        async with mysql_connector.session_scope() as session:
            job = await session.get(VideoMatchJob, job_id)
            if job:
                job.search_status = "failed"
                job.search_error = str(e)
                session.add(job)
                await session.commit()
        return {"success": False, "job_id": job_id, "error": str(e)}

    total_ms = round((time.perf_counter() - t_wall0) * 1000, 3)

    if len(matches) != len(rows):
        log.warning("video_match_services: match count {} != rows {}", len(matches), len(rows))

    # job 级汇总：Top1 去重率（帮助判断检索策略多样性）
    _all_top1s = []
    for _m in matches:
        if isinstance(_m, dict):
            _hits = _m.get("top_hits") or []
            _t1 = _hits[0].get("video_path", "") if _hits else ""
            if _t1:
                _all_top1s.append(_t1.split("/")[-1])
    from collections import Counter as _Counter
    _dup = _Counter(_all_top1s)
    _dup_lines = "\n".join(
        "  x{} -> {}".format(cnt, name)
        for name, cnt in _dup.most_common()
        if cnt > 1
    ) or "  (无重复)"
    log.info(
        "video_match_services search finished job={} wall_ms={} segments={} top1_unique={}/{}\n"
        "  重复 Top1:\n{}",
        job_id, total_ms, len(rows), len(_dup), len(_all_top1s), _dup_lines,
    )

    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        res_sr = await session.execute(
            select(VideoMatchShotRow).where(VideoMatchShotRow.job_id == job_id)
        )
        shot_rows = list(res_sr.scalars().all())
        n_fail = sum(1 for sr in shot_rows if (sr.search_status or "").lower() == "failed")
        if job:
            if n_fail:
                job.search_status = "failed"
                job.search_error = (
                    f"{n_fail} 条分镜素材匹配失败（无 OpenSearch 命中或无法解析出有效视频地址 / Top5 为空）"
                )
            else:
                job.search_status = "done"
                job.search_error = None
            job.search_total_ms = total_ms
            session.add(job)
        await session.commit()

    out = await get_job_payload(job_id)
    if out is None:
        return {"success": False, "error": "job not found after search"}
    job_failed = (out.get("search_status") or "").lower() == "failed"
    out["success"] = not job_failed
    if job_failed:
        out["error"] = out.get("search_error") or "部分或全部分镜匹配失败"
    out["search_total_ms"] = total_ms
    return out


async def create_job_and_parse(
    *,
    script: str,
    topic: Optional[str] = None,
    title: Optional[str] = None,
    car_model: Optional[str] = None,
    frame_size: Optional[str] = None,
    frame_orientation: Optional[str] = None,
    workspace: Optional[str] = "v1",
    mock: bool = False,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Dict[str, Any]:
    if mock:
        return _mock_response_payload()

    script = (script or "").strip()
    if not script:
        return {"success": False, "error": "script cannot be empty"}

    ws = (workspace or "v1").strip() or "v1"

    fs_norm = (frame_size or "").strip() or None
    fo_norm = _norm_job_frame_orientation(frame_orientation)

    async with mysql_connector.session_scope() as session:
        job = VideoMatchJob(
            workspace=ws,
            script=script,
            topic=topic.strip() if topic else None,
            title=title.strip() if title else None,
            car_model=car_model.strip() if car_model else None,
            frame_size=fs_norm,
            frame_orientation=fo_norm,
            parse_status="running",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id

    parse_rid = await http_request_trace_service.create_initial(
        request_url="/internal/video-match/parse",
        http_method="POST",
        method_name="POST /video-match/jobs",
        business_type="VIDEO_MATCH_PARSE",
        business_id=str(job_id),
        upstream_task_id=str(job_id),
        request_body=truncate_for_trace(
            {
                "job_id": job_id,
                "workspace": ws,
                "script_preview": script[:8000],
                "topic": topic,
                "title": title,
                "car_model": car_model,
                "frame_size": fs_norm,
                "frame_orientation": fo_norm,
            },
            max_bytes=32000,
        ),
    )
    async with mysql_connector.session_scope() as session:
        job_link = await session.get(VideoMatchJob, job_id)
        if job_link:
            job_link.request_id = parse_rid
            session.add(job_link)
            await session.commit()

    async def parse_task():
        t_parse0 = time.perf_counter()
        try:
            tts_audio_urls: List[Optional[str]] = []
            storyboard, tags = await rewrite_script_to_storyboard_and_tags(
                script,
                topic=topic,
                title=title,
                car_model=car_model,
                frame_size=fs_norm,
                frame_orientation=fo_norm,
                index=0,
                tts_obs_project_id=str(job_id),
                out_obs_audio_urls=tts_audio_urls,
            )

            async with mysql_connector.session_scope() as session:
                for order, seg in enumerate(storyboard.storyboard):
                    tag_seg = _resolve_tag_segment(tags, seg.id, order)
                    tj: Optional[Dict[str, Any]]
                    if tag_seg is not None:
                        tj = tag_seg.model_dump(exclude_none=True)
                    else:
                        tj = {}
                    tj = _merge_job_constraints_into_segment_tags(
                        tj,
                        car_model=car_model,
                        frame_size=fs_norm,
                        frame_orientation=fo_norm,
                    )
                    obs_url = tts_audio_urls[order] if order < len(tts_audio_urls) else None
                    row = VideoMatchShotRow(
                        job_id=job_id,
                        shot_order=order,
                        storyboard_id=int(seg.id),
                        segment_text=seg.segment_text,
                        duration_sec=float(seg.duration),
                        description=seg.description,
                        tags_json=tj if tj else None,
                        extract_status="done" if tj else "pending",
                        search_status="pending",
                        obs_audio_url=obs_url,
                    )
                    session.add(row)
                job_obj = await session.get(VideoMatchJob, job_id)
                if job_obj:
                    job_obj.parse_status = "done"
                    job_obj.parse_error = None
                    job_obj.extract_status = "done"
                    job_obj.extract_error = None
                    session.add(job_obj)
                await session.commit()

            await http_request_trace_service.finalize(
                parse_rid,
                status_code=200,
                response_body={
                    "parse_status": "done",
                    "shot_count": len(storyboard.storyboard),
                },
                business_success=True,
                duration_ms=int((time.perf_counter() - t_parse0) * 1000),
            )
            log.info("Background parsing completed successfully for job_id={}", job_id)
        except Exception as e:
            log.exception("Background parsing failed for job_id={}: {}", job_id, e)
            await http_request_trace_service.finalize(
                parse_rid,
                status_code=500,
                error_message=str(e)[:2000],
                response_body={"parse_status": "failed"},
                business_success=False,
                duration_ms=int((time.perf_counter() - t_parse0) * 1000),
            )
            async with mysql_connector.session_scope() as session:
                job_obj = await session.get(VideoMatchJob, job_id)
                if job_obj:
                    job_obj.parse_status = "failed"
                    job_obj.parse_error = str(e)
                    session.add(job_obj)
                    await session.commit()

    if background_tasks:
        background_tasks.add_task(parse_task)
        return {
            "success": True,
            "mock": False,
            "job_id": job_id,
            "serial_no": None,
            "request_id": parse_rid,
            "script": script,
            "topic": topic,
            "title": title,
            "car_model": car_model,
            "frame_size": fs_norm,
            "frame_orientation": fo_norm,
            "workspace": ws,
            "parse_status": "running",
            "parse_error": None,
            "extract_status": "pending",
            "search_status": "pending",
            "search_total_ms": None,
            "search_error": None,
            "search_strategy_snapshot": None,
            "shots": [],
        }
    else:
        await parse_task()
        loaded = await get_job_payload(job_id)
        if loaded is None:
            return {"success": False, "job_id": job_id, "error": "job not found after parse"}
        return loaded


async def list_video_match_jobs(
    *,
    parse_status: Optional[str] = None,
    workspace: Optional[str] = None,
    ids: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Lightweight job list for debugging / history picker（如「已转写」会话）。
    """
    lim = max(1, min(int(limit or 50), 100))
    async with mysql_connector.session_scope() as session:
        stmt = select(VideoMatchJob).order_by(VideoMatchJob.created_at.desc()).limit(lim)
        ps = (parse_status or "").strip()
        if ps:
            stmt = stmt.where(VideoMatchJob.parse_status == ps)
        ws = (workspace or "").strip()
        if ws:
            stmt = stmt.where(VideoMatchJob.workspace == ws)
        ids_str = (ids or "").strip()
        if ids_str:
            id_list = [i.strip() for i in ids_str.split(",") if i.strip()]
            if id_list:
                from sqlalchemy import or_
                # Check if any id could be a serial_no
                sn_list = [int(i) for i in id_list if i.isdigit()]
                if sn_list:
                    stmt = stmt.where(or_(VideoMatchJob.id.in_(id_list), VideoMatchJob.serial_no.in_(sn_list)))
                else:
                    stmt = stmt.where(VideoMatchJob.id.in_(id_list))
        res = await session.execute(stmt)
        jobs = list(res.scalars().all())
    items: List[Dict[str, Any]] = []
    for j in jobs:
        items.append(
            {
                "id": j.id,
                "serial_no": getattr(j, "serial_no", None),
                "workspace": j.workspace,
                "parse_status": j.parse_status,
                "extract_status": getattr(j, "extract_status", "pending"),
                "search_status": j.search_status,
                "title": j.title,
                "topic": j.topic,
                "car_model": j.car_model,
                "frame_size": j.frame_size,
                "frame_orientation": j.frame_orientation,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "updated_at": j.updated_at.isoformat() if j.updated_at else None,
                "request_id": j.request_id,
            }
        )
    return {"success": True, "jobs": items}


async def list_material_match_histories(
    *,
    workspace: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    ids: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """素材匹配看板列表（视频匹配分镜检索 + 视频分析搜索栏）。"""
    lim = max(1, min(int(limit or 100), 200))
    async with mysql_connector.session_scope() as session:
        stmt = select(VideoMaterialMatchHistory).order_by(
            VideoMaterialMatchHistory.created_at.desc()
        ).limit(lim)
        ws = (workspace or "").strip()
        if ws:
            stmt = stmt.where(VideoMaterialMatchHistory.workspace == ws)
        src = (source or "").strip()
        if src:
            stmt = stmt.where(VideoMaterialMatchHistory.source == src)
        ids_str = (ids or "").strip()
        if ids_str:
            id_list = [i.strip() for i in ids_str.split(",") if i.strip()]
            if id_list:
                stmt = stmt.where(VideoMaterialMatchHistory.id.in_(id_list))
        stf = (status or "").strip()
        if stf:
            stmt = stmt.where(VideoMaterialMatchHistory.status == stf)
        res = await session.execute(stmt)
        rows = list(res.scalars().all())
    items: List[Dict[str, Any]] = []
    for h in rows:
        items.append(
            {
                "id": h.id,
                "request_id": h.request_id,
                "source": h.source,
                "workspace": h.workspace,
                "status": h.status,
                "error_message": h.error_message,
                "video_match_job_id": h.video_match_job_id,
                "video_match_shot_row_id": h.video_match_shot_row_id,
                "va_context_history_id": h.va_context_history_id,
                "hit_count": h.hit_count,
                "top1_obs_url": h.top1_obs_url,
                "elapsed_ms": h.elapsed_ms,
                "query_preview": h.query_preview,
                "search_mode": h.search_mode,
                "strategy_snapshot": h.strategy_snapshot,
                "enable_road_run_fallback": h.enable_road_run_fallback,
                "created_at": h.created_at.isoformat() if h.created_at else None,
                "updated_at": h.updated_at.isoformat() if h.updated_at else None,
            }
        )
    return {"success": True, "matches": items}


async def _reparse_video_match_job_core(job_id: str) -> None:
    """假定 job 已 parse_status=running 且分镜行已清空；执行 LLM 转写并落库。"""
    jid = (job_id or "").strip()
    if not jid:
        return
    async with mysql_connector.session_scope() as session:
        job0 = await session.get(VideoMatchJob, jid)
        if job0 is None:
            return
        script = (job0.script or "").strip()
        topic = job0.topic
        title = job0.title
        car_model = job0.car_model
        frame_size_job = (job0.frame_size or "").strip() or None
        frame_orientation_job = _norm_job_frame_orientation(job0.frame_orientation)
        ws = (job0.workspace or "v1").strip() or "v1"
    if not script:
        async with mysql_connector.session_scope() as session:
            jbad = await session.get(VideoMatchJob, jid)
            if jbad:
                jbad.parse_status = "failed"
                jbad.parse_error = "script is empty"
                session.add(jbad)
                await session.commit()
        return

    parse_rid = await http_request_trace_service.create_initial(
        request_url="/internal/video-match/parse-retry",
        http_method="POST",
        method_name="POST /video-match/jobs/{id}/retry",
        business_type="VIDEO_MATCH_PARSE",
        business_id=jid,
        upstream_task_id=jid,
        request_body=truncate_for_trace(
            {
                "job_id": jid,
                "workspace": ws,
                "retry": True,
                "script_preview": script[:8000],
                "car_model": car_model,
                "frame_size": frame_size_job,
                "frame_orientation": frame_orientation_job,
            },
            max_bytes=32000,
        ),
    )
    async with mysql_connector.session_scope() as session:
        job_link = await session.get(VideoMatchJob, jid)
        if job_link:
            job_link.request_id = parse_rid
            session.add(job_link)
            await session.commit()

    t_parse0 = time.perf_counter()
    try:
        tts_audio_urls: List[Optional[str]] = []
        storyboard, tags = await rewrite_script_to_storyboard_and_tags(
            script,
            topic=topic,
            title=title,
            car_model=car_model,
            frame_size=frame_size_job,
            frame_orientation=frame_orientation_job,
            index=0,
            tts_obs_project_id=str(jid),
            out_obs_audio_urls=tts_audio_urls,
        )
    except Exception as e:
        log.exception("video_match_services parse retry failed: {}", e)
        await http_request_trace_service.finalize(
            parse_rid,
            status_code=500,
            error_message=str(e)[:2000],
            response_body={"parse_status": "failed"},
            business_success=False,
            duration_ms=int((time.perf_counter() - t_parse0) * 1000),
        )
        async with mysql_connector.session_scope() as session:
            job = await session.get(VideoMatchJob, jid)
            if job:
                job.parse_status = "failed"
                job.parse_error = str(e)
                session.add(job)
                await session.commit()
        return

    async with mysql_connector.session_scope() as session:
        for order, seg in enumerate(storyboard.storyboard):
            tag_seg = _resolve_tag_segment(tags, seg.id, order)
            tj: Optional[Dict[str, Any]]
            if tag_seg is not None:
                tj = tag_seg.model_dump(exclude_none=True)
            else:
                tj = {}
            tj = _merge_job_constraints_into_segment_tags(
                tj,
                car_model=car_model,
                frame_size=frame_size_job,
                frame_orientation=frame_orientation_job,
            )
            obs_url = tts_audio_urls[order] if order < len(tts_audio_urls) else None
            row = VideoMatchShotRow(
                job_id=jid,
                shot_order=order,
                storyboard_id=int(seg.id),
                segment_text=seg.segment_text,
                duration_sec=float(seg.duration),
                description=seg.description,
                tags_json=tj if tj else None,
                extract_status="done" if tj else "pending",
                search_status="pending",
                obs_audio_url=obs_url,
            )
            session.add(row)
        job = await session.get(VideoMatchJob, jid)
        if job:
            job.parse_status = "done"
            job.parse_error = None
            job.extract_status = "done"
            job.extract_error = None
            session.add(job)
        await session.commit()

    await http_request_trace_service.finalize(
        parse_rid,
        status_code=200,
        response_body={
            "parse_status": "done",
            "shot_count": len(storyboard.storyboard),
            "retry": True,
        },
        business_success=True,
        duration_ms=int((time.perf_counter() - t_parse0) * 1000),
    )


async def run_job_extract_tags(job_id: str) -> Dict[str, Any]:
    from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextStoryboardEnvelope, SeedtextStoryboardSegment
    import time
    
    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        if job is None:
            return {"success": False, "error": "job not found"}
        if job.parse_status != "done":
            return {"success": False, "error": "parse not completed"}
            
        from sqlalchemy import select
        res = await session.execute(
            select(VideoMatchShotRow)
            .where(VideoMatchShotRow.job_id == job_id)
            .order_by(VideoMatchShotRow.shot_order)
        )
        rows = list(res.scalars().all())
        if not rows:
            return {"success": False, "error": "no shot rows"}
            
        job.extract_status = "running"
        job.extract_error = None
        for r in rows:
            r.extract_status = "running"
            session.add(r)
        session.add(job)
        await session.commit()
        
    t0 = time.perf_counter()
    try:
        segs = []
        for r in rows:
            tj = r.tags_json or {}
            
            # Map from tags_json if present, else fallback
            video_usage_list = tj.get("video_usage", [])
            video_usage = video_usage_list[0] if video_usage_list else "未知"
            shot_style = tj.get("shot_style", "未知")
            shot_type = tj.get("shot_type", "未知")
            subject = tj.get("subject", "未知")
            obj_list = tj.get("object", [])
            if not obj_list:
                obj_list = ["未知"]
                
            sp_list = tj.get("design_selling_points", []) + tj.get("function_selling_points", [])
            if not sp_list:
                sp_list = ["未知"]

            segs.append(SeedtextStoryboardSegment(
                id=str(r.storyboard_id),
                index=0,
                segment_text=r.segment_text,
                duration=str(r.duration_sec),
                description=r.description,
                video_usage=video_usage,
                shot_style=shot_style,
                shot_type=shot_type,
                subject=subject,
                object=obj_list,
                selling_point=sp_list
            ))
        storyboard = SeedtextStoryboardEnvelope(storyboard=segs)
        
        tags = await rewrite_storyboard_to_tags(
            storyboard,
            frame_size=job.frame_size,
            frame_orientation=job.frame_orientation,
            index=0
        )
    except Exception as e:
        log.exception("extract tags failed: {}", e)
        async with mysql_connector.session_scope() as session:
            job = await session.get(VideoMatchJob, job_id)
            if job:
                job.extract_status = "failed"
                job.extract_error = str(e)
                session.add(job)
            await session.commit()
        return {"success": False, "error": str(e)}
        
    async with mysql_connector.session_scope() as session:
        for order, r in enumerate(rows):
            tag_seg = _resolve_tag_segment(tags, str(r.storyboard_id), order)
            tj = {}
            if tag_seg is not None:
                tj = tag_seg.model_dump(exclude_none=True)
            tj = _merge_job_constraints_into_segment_tags(
                tj,
                car_model=job.car_model,
                frame_size=job.frame_size,
                frame_orientation=job.frame_orientation,
            )
            
            # Re-fetch row
            r_db = await session.get(VideoMatchShotRow, r.id)
            if r_db:
                r_db.tags_json = tj
                r_db.extract_status = "done"
                session.add(r_db)
                
        job_db = await session.get(VideoMatchJob, job_id)
        if job_db:
            job_db.extract_status = "done"
            job_db.extract_error = None
            session.add(job_db)
        await session.commit()
        
    return {"success": True, "duration_ms": int((time.perf_counter() - t0) * 1000)}


async def run_shot_extract_tags(job_id: str, shot_row_id: int) -> Dict[str, Any]:
    from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextStoryboardEnvelope, SeedtextScriptSegment
    import time
    
    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        if job is None:
            return {"success": False, "error": "job not found"}
            
        r = await session.get(VideoMatchShotRow, shot_row_id)
        if r is None or r.job_id != job_id:
            return {"success": False, "error": "shot not found"}
            
        r.extract_status = "running"
        session.add(r)
        await session.commit()
        
    t0 = time.perf_counter()
    try:
        seg = SeedtextScriptSegment(
            id=str(r.storyboard_id),
            index=0,
            segment_text=r.segment_text,
            duration=str(r.duration_sec),
            description=r.description
        )
        storyboard = SeedtextStoryboardEnvelope(storyboard=[seg])
        
        tags = await rewrite_storyboard_to_tags(
            storyboard,
            frame_size=job.frame_size,
            frame_orientation=job.frame_orientation,
            index=0
        )
    except Exception as e:
        log.exception("extract shot tags failed: {}", e)
        async with mysql_connector.session_scope() as session:
            r_db = await session.get(VideoMatchShotRow, shot_row_id)
            if r_db:
                r_db.extract_status = "failed"
                session.add(r_db)
            await session.commit()
        return {"success": False, "error": str(e)}
        
    async with mysql_connector.session_scope() as session:
        tag_seg = _resolve_tag_segment(tags, str(r.storyboard_id), 0)
        tj = {}
        if tag_seg is not None:
            tj = tag_seg.model_dump(exclude_none=True)
        tj = _merge_job_constraints_into_segment_tags(
            tj,
            car_model=job.car_model,
            frame_size=job.frame_size,
            frame_orientation=job.frame_orientation,
        )
        
        r_db = await session.get(VideoMatchShotRow, shot_row_id)
        if r_db:
            r_db.tags_json = tj
            r_db.extract_status = "done"
            session.add(r_db)
        await session.commit()
        
    return {"success": True, "duration_ms": int((time.perf_counter() - t0) * 1000)}

async def update_shot_tokens(job_id: str, shot_row_id: int, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    保存用户手动编辑的 AND/OR/NOT 标签到 search_tokens_json。
    """
    jid = (job_id or "").strip()
    try:
        sid = int(shot_row_id)
    except (TypeError, ValueError):
        sid = 0
    if not jid or sid <= 0:
        return {"success": False, "error": "invalid id"}

    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMatchShotRow, sid)
        if row is None or str(row.job_id) != jid:
            return {"success": False, "error": "shot not found"}
        
        row.search_tokens_json = tokens
        session.add(row)
        await session.commit()
        return {"success": True}


async def update_shot_top1_url(job_id: str, shot_row_id: int, top1_obs_url: str) -> Dict[str, Any]:
    """
    手动切换分镜的 Top1 视频。
    """
    jid = str(job_id or "").strip()
    try:
        sid = int(shot_row_id)
    except (TypeError, ValueError):
        sid = 0
    if not jid or sid <= 0:
        return {"success": False, "error": "invalid id"}

    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMatchShotRow, sid)
        if row is None or str(row.job_id) != jid:
            return {"success": False, "error": "shot not found"}
        
        row.top1_obs_url = top1_obs_url
        # 如果当前分镜是失败状态，但被用户手动指定了有效的视频地址，则更新为 done 成功状态
        if top1_obs_url and row.search_status == "failed":
            row.search_status = "done"
            
        session.add(row)
        await session.flush()

        # 同时更新 parent job 的 search_status，防止因为单个分镜手动选择后，父 Job 仍处于 failed 状态
        res_sr = await session.execute(
            select(VideoMatchShotRow).where(VideoMatchShotRow.job_id == jid)
        )
        shot_rows = list(res_sr.scalars().all())
        n_fail = sum(1 for sr in shot_rows if (sr.search_status or "").lower() == "failed")
        n_running = sum(1 for sr in shot_rows if (sr.search_status or "").lower() in ("running", "pending"))
        
        job = await session.get(VideoMatchJob, jid)
        if job:
            if n_running > 0:
                job.search_status = "running"
                job.search_error = None
            elif n_fail > 0:
                job.search_status = "failed"
                job.search_error = (
                    f"{n_fail} 条分镜素材匹配失败（无 OpenSearch 命中或无法解析出有效视频地址 / Top5 为空）"
                )
            else:
                job.search_status = "done"
                job.search_error = None
            session.add(job)
            
        await session.commit()
        return {"success": True, "top1_obs_url": top1_obs_url}

