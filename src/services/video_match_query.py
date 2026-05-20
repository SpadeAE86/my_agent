
import json
from typing import Optional, List, Dict, Any

from sqlmodel import select

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow
from models.sqlmodel.video_material_match import VideoMaterialMatchHistory
from services.http_request_trace_service import http_request_trace_service
from services.video_analysis_db_service import video_analysis_db_service
from services.script_match_query_builder import INDEX_NAME

def shot_row_to_api_dict(row: VideoMatchShotRow) -> Dict[str, Any]:
    tj = row.tags_json or {}
    summary = _tags_summary_from_json(tj)
    hits = row.match_top_hits_json
    stored_top1 = str(row.top1_obs_url or "").strip() or None
    fallback_top1 = _best_video_path_from_hits(hits)
    top1_effective = fallback_top1 or stored_top1
    return {
        "id": row.id,
        "shot_order": row.shot_order,
        "storyboard_id": row.storyboard_id,
        "segment_text": row.segment_text,
        "duration_sec": row.duration_sec,
        "description": row.description,
        "tags_summary": summary,
        "tags_json": tj,
        "extract_status": getattr(row, "extract_status", "pending"),
        "search_status": row.search_status,
        "top1_obs_url": top1_effective,
        "top5_video_urls": _top5_video_urls_from_hits(hits),
        "obs_audio_url": row.obs_audio_url,
        "match_top_hits_json": row.match_top_hits_json,
        "match_elapsed_ms": row.match_elapsed_ms,
        "match_hit_count": len(hits) if isinstance(hits, list) else 0,
        "search_request_id": row.search_request_id,
        "match_id": row.match_id,
    }



async def get_job_payload(job_id: str) -> Optional[Dict[str, Any]]:
    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, job_id)
        if job is None:
            return None
        res = await session.execute(
            select(VideoMatchShotRow)
            .where(VideoMatchShotRow.job_id == job_id)
            .order_by(VideoMatchShotRow.shot_order)
        )
        rows = list(res.scalars().all())
    ws_ver = "v2" if str(job.workspace or "v1").strip() == "v2" else "v1"
    shots = [shot_row_to_api_dict(r) for r in rows]
    for s in shots:
        await _hydrate_shot_match_urls_for_response(s, shot_cards_version=ws_ver)
    return {
        "success": True,
        "mock": False,
        "job_id": job.id,
        "serial_no": getattr(job, "serial_no", None),
        "request_id": job.request_id,
        "script": job.script,
        "topic": job.topic,
        "title": job.title,
        "car_model": job.car_model,
        "frame_size": job.frame_size,
        "frame_orientation": job.frame_orientation,
        "workspace": job.workspace,
        "parse_status": job.parse_status,
        "parse_error": job.parse_error,
        "extract_status": getattr(job, "extract_status", "pending"),
        "search_status": job.search_status,
        "search_total_ms": job.search_total_ms,
        "search_error": job.search_error,
        "search_strategy_snapshot": job.search_strategy_snapshot,
        "shots": shots,
    }



async def get_shot_match_detail(job_id: str, shot_row_id: int) -> Optional[Dict[str, Any]]:
    """分镜「素材匹配」阶段详情：合并 http_request_traces（search_request_id）。"""
    jid = (job_id or "").strip()
    if not jid or shot_row_id <= 0:
        return None
    from services.task_detail_service import (
        build_video_match_shot_search_task_detail,
        merge_http_trace_into_detail,
    )

    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMatchShotRow, shot_row_id)
        if row is None or str(row.job_id) != jid:
            return None
        job_row = await session.get(VideoMatchJob, jid)
        ws_ver = (
            "v2"
            if job_row is not None and str(job_row.workspace or "v1").strip() == "v2"
            else "v1"
        )
        rid = (row.search_request_id or "").strip()
        shot_order = int(row.shot_order)
        seg_text = row.segment_text or ""
        shot_api = shot_row_to_api_dict(row)

    await _hydrate_shot_match_urls_for_response(shot_api, shot_cards_version=ws_ver)

    base = build_video_match_shot_search_task_detail(
        job_id=jid,
        shot_row_id=shot_row_id,
        shot_order=shot_order,
        segment_text_preview=seg_text,
        opensearch_index=INDEX_NAME,
    )
    trace = await http_request_trace_service.get_dict(rid) if rid else None
    detail = merge_http_trace_into_detail(base, trace)
    return {"success": True, "detail": detail, "shot": shot_api}


async def get_material_match_board_detail(match_id: str) -> Optional[Dict[str, Any]]:
    """任务看板：单条素材匹配履历 HTTP 详情。"""
    from services.task_detail_service import (
        build_video_material_match_task_detail,
        merge_http_trace_into_detail,
    )

    mid = (match_id or "").strip()
    if not mid:
        return None
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoMaterialMatchHistory, mid)
    if row is None:
        return None
    d: Dict[str, Any] = {
        "id": row.id,
        "status": row.status,
        "source": row.source,
        "workspace": row.workspace,
        "error_message": row.error_message,
        "video_match_job_id": row.video_match_job_id,
        "video_match_shot_row_id": row.video_match_shot_row_id,
        "va_context_history_id": row.va_context_history_id,
        "hit_count": row.hit_count,
        "top1_obs_url": row.top1_obs_url,
        "elapsed_ms": row.elapsed_ms,
        "query_preview": row.query_preview,
        "search_mode": row.search_mode,
        "strategy_snapshot": row.strategy_snapshot,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    base = build_video_material_match_task_detail(d)
    rid = (row.request_id or "").strip()
    trace_dict = await http_request_trace_service.get_dict(str(rid)) if rid else None
    return merge_http_trace_into_detail(base, trace_dict)



async def _hydrate_shot_match_urls_for_response(
    shot: Dict[str, Any],
    *,
    shot_cards_version: str,
) -> None:
    """
    读取任务时补全展示字段：库内 ``top1_obs_url`` 可能因历史 bug 为空，但 ``match_top_hits_json``
    里仍有 ``history_id``。用 ``resolve_source_video_url_for_index_key`` 再解析一次。
    """
    if str(shot.get("top1_obs_url") or "").strip():
        return
    if str(shot.get("search_status") or "").lower() != "done":
        return
    hits = shot.get("match_top_hits_json")
    if not isinstance(hits, list) or not hits:
        return
    ver: Any = "v2" if (shot_cards_version or "v1").strip() == "v2" else "v1"
    new_hits: List[Any] = []
    for h in hits:
        if not isinstance(h, dict):
            new_hits.append(h)
            continue
        nh = dict(h)
        if not str(
            nh.get("video_path") or nh.get("video_url") or nh.get("url") or nh.get("obs_video_url") or ""
        ).strip():
            hid = str(nh.get("history_id") or "").strip()
            if hid:
                url = await video_analysis_db_service.resolve_source_video_url_for_index_key(
                    hid, shot_cards_version=ver
                )
                if url:
                    nh["video_path"] = url
        new_hits.append(nh)
    t1 = _best_video_path_from_hits(new_hits)
    if not t1:
        log.debug(
            "video_match hydrate: shot_order={} still no top1 (hits={})",
            shot.get("shot_order"),
            len(new_hits),
        )
        return
    shot["match_top_hits_json"] = new_hits
    shot["top1_obs_url"] = t1
    shot["top5_video_urls"] = _top5_video_urls_from_hits(new_hits)
    shot["match_hit_count"] = len(new_hits)


async def _enrich_hits_with_resolved_urls(top_hits: Any, shot_cards_version: str) -> List[Dict[str, Any]]:
    """
    将 OpenSearch 命中里的 history_id 解析为可播放地址并写回各 hit 的 video_path，
    便于落库与 Top5 判定（与 get_job_payload 中的 hydrate 同源逻辑）。
    """
    if not isinstance(top_hits, list) or not top_hits:
        return []
    ver: Any = "v2" if (shot_cards_version or "v1").strip() == "v2" else "v1"
    out: List[Dict[str, Any]] = []
    for h in top_hits:
        if not isinstance(h, dict):
            continue
        nh = dict(h)
        if not str(
            nh.get("video_path", "") or nh.get("video_url", "") or nh.get("url", "") or nh.get("obs_video_url", "") or ""
        ).strip():
            hid = str(nh.get("history_id") or "").strip()
            if hid:
                url = await video_analysis_db_service.resolve_source_video_url_for_index_key(
                    hid, shot_cards_version=ver
                )
                if url:
                    nh["video_path"] = url
        out.append(nh)
    return out



def _best_video_path_from_hits(hits: Any) -> Optional[str]:
    """从检索命中取可播放地址（兼容 video_path / video_url 等列）。"""
    if not isinstance(hits, list):
        return None
    for h in hits:
        if not isinstance(h, dict):
            continue
        for key in ("video_path", "video_url", "url", "obs_video_url"):
            u = str(h.get(key) or "").strip()
            if u:
                return u
    return None



def _top5_video_urls_from_hits(hits: Any) -> List[str]:
    if not isinstance(hits, list):
        return []
    urls: List[str] = []
    seen: set[str] = set()
    for h in hits:
        if len(urls) >= 5:
            break
        if not isinstance(h, dict):
            continue
        u = ""
        for key in ("video_path", "video_url", "url", "obs_video_url"):
            u = str(h.get(key) or "").strip()
            if u:
                break
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    return urls



def _tags_summary_from_json(tj: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "car_model",
        "frame_size",
        "frame_orientation",
        "subject",
        "footage_type",
        "movement",
        "product_status_scene",
    ):
        v = tj.get(key)
        if v and isinstance(v, str) and v.strip() and v != "未知":
            parts.append(v.strip())
    objs = tj.get("object")
    if isinstance(objs, list):
        parts.extend(str(x) for x in objs[:3] if x)
    return " · ".join(parts[:8]) if parts else "—"



def _mock_response_payload() -> Dict[str, Any]:
    """与真实 parse 成功时相同 schema，便于前端联调。"""
    shots = [
        {
            "id": None,
            "shot_order": 0,
            "storyboard_id": 1,
            "segment_text": "同级唯一，全系标配大厂底盘。",
            "duration_sec": 2.4,
            "description": "城市道路跟拍，车身平稳，强调底盘质感。",
            "tags_summary": "",
            "tags_json": {
                "id": 1,
                "segment_text": "同级唯一，全系标配大厂底盘。",
                "duration": 2.4,
                "description": "跟拍路跑，侧向航拍交代环境",
                "movement": "行驶",
                "subject": "智己LS6",
                "footage_type": "生活实拍",
                "product_status_scene": "动态路跑",
                "object": ["四轮", "城市道路"],
                "scene_location": ["城市道路"],
            },
            "search_status": "pending",
            "top1_obs_url": None,
            "top5_video_urls": [],
            "obs_audio_url": None,
            "match_top_hits_json": None,
            "match_elapsed_ms": None,
            "match_hit_count": 0,
            "search_request_id": None,
            "match_id": None,
        },
        {
            "id": None,
            "shot_order": 1,
            "storyboard_id": 2,
            "segment_text": "一键 AI 泊车，地库自己找车位。",
            "duration_sec": 3.0,
            "description": "车内 POV，中控显示泊车界面，地库环境。",
            "tags_summary": "",
            "tags_json": {
                "id": 2,
                "segment_text": "一键 AI 泊车，地库自己找车位。",
                "duration": 3.0,
                "description": "中控大屏与方向盘入画，泊车 UI",
                "movement": "静止",
                "subject": "中控大屏",
                "footage_type": "生活实拍",
                "product_status_scene": "功能演示",
                "object": ["方向盘", "地库"],
                "scene_location": ["地库"],
            },
            "search_status": "pending",
            "top1_obs_url": None,
            "top5_video_urls": [],
            "obs_audio_url": None,
            "match_top_hits_json": None,
            "match_elapsed_ms": None,
            "match_hit_count": 0,
            "search_request_id": None,
            "match_id": None,
        },
        {
            "id": None,
            "shot_order": 2,
            "storyboard_id": 3,
            "segment_text": "静谧座舱，长途也不累。",
            "duration_sec": 2.0,
            "description": "后排乘坐空间与氛围光，安静体感。",
            "tags_summary": "",
            "tags_json": {
                "id": 3,
                "segment_text": "静谧座舱，长途也不累。",
                "duration": 2.0,
                "description": "后排座椅与车窗取景",
                "movement": "静止",
                "subject": "后排座椅",
                "footage_type": "TVC切片",
                "product_status_scene": "静态内饰",
                "object": ["氛围灯"],
                "scene_location": ["车内"],
            },
            "search_status": "pending",
            "top1_obs_url": None,
            "top5_video_urls": [],
            "obs_audio_url": None,
            "match_top_hits_json": None,
            "match_elapsed_ms": None,
            "match_hit_count": 0,
            "search_request_id": None,
            "match_id": None,
        },
    ]
    for s in shots:
        s["tags_summary"] = _tags_summary_from_json(s["tags_json"])
    return {
        "success": True,
        "mock": True,
        "job_id": "00000000-0000-0000-0000-00000000mock",
        "request_id": None,
        "script": "",
        "topic": None,
        "title": None,
        "car_model": None,
        "frame_size": None,
        "frame_orientation": None,
        "parse_status": "done",
        "search_status": "pending",
        "search_total_ms": None,
        "search_error": None,
        "search_strategy_snapshot": None,
        "shots": shots,
    }



