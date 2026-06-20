# services/task_detail_service.py — 任务看板「HTTP 调用详情」合成（便于后续接真实网关 trace JSON）
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _iso(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat(timespec="seconds")
    if hasattr(dt, "isoformat"):
        try:
            return dt.isoformat()  # type: ignore[no-any-return]
        except Exception:
            return str(dt)
    return str(dt)


def _duration_ms_created_updated(row: Dict[str, Any]) -> Optional[int]:
    ca, ua = row.get("created_at"), row.get("updated_at")
    if not ca or not ua:
        return None
    try:
        if isinstance(ca, str):
            ca = datetime.fromisoformat(ca.replace("Z", "+00:00"))
        if isinstance(ua, str):
            ua = datetime.fromisoformat(ua.replace("Z", "+00:00"))
        if isinstance(ca, datetime) and isinstance(ua, datetime):
            delta = ua - ca
            return int(delta.total_seconds() * 1000)
    except Exception:
        pass
    return None


def _image_reference_urls(row: Dict[str, Any]) -> List[str]:
    raw = row.get("referenceMedia")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for m in raw:
        if isinstance(m, dict):
            u = m.get("url")
            if u:
                out.append(str(u))
    return out


def _norm_image_success(row: Dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    s = (row.get("status") or "").lower()
    if s in ("failed", "error"):
        return False
    if s in ("running", "pending"):
        return False
    if row.get("url") or row.get("obs_url") or row.get("doubao_url"):
        return True
    if s in ("success", "succeed", "succeeded"):
        return True
    return False


def build_image_task_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    """由 image_history_cards 行数据合成管理后台风格的 HTTP 明细。"""
    ok = _norm_image_success(row)
    duration_ms = _duration_ms_created_updated(row)
    req_body: Dict[str, Any] = {
        "prompt": row.get("prompt"),
        "model": row.get("model"),
        "size": row.get("size"),
        "ratio": row.get("ratio"),
        "resolution": row.get("resolution"),
        "type": row.get("type"),
        "reference_image_list": _image_reference_urls(row) or None,
    }
    resp_body: Dict[str, Any] = {
        "success": ok,
        "image_url": row.get("url"),
        "task_id": row.get("taskId"),
        "error": row.get("error"),
        "doubao_url": row.get("doubao_url"),
        "obs_url": row.get("obs_url"),
    }
    return {
        "id": row.get("id"),
        "taskId": row.get("taskId"),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": "POST /image",
        "httpMethod": "POST",
        "businessType": "IMAGE_GEN",
        "prompt_full": row.get("prompt"),
        "result_image_url": row.get("url") or row.get("obs_url") or row.get("doubao_url"),
        "requestUrl": "/image",
        "statusCode": 200 if ok else 500,
        "durationMs": duration_ms,
        "businessSuccess": ok,
        "businessStatusLabel": "成功" if ok else ("失败" if row.get("error") or (row.get("status") or "").lower() == "failed" else "进行中"),
        "errorMessage": row.get("error"),
        "timeDisplay": row.get("time"),
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
        "requestHeaders": {},
        "requestBody": req_body,
        "responseHeaders": {},
        "responseBody": resp_body,
        "note": "当前为根据历史表字段推断的合成详情；接入网关后可写入真实 request/response。",
    }


def build_video_gen_task_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    """由 image_history_cards 行数据（type=t2v/i2v）合成视频生成 HTTP 明细。"""
    ok = _norm_image_success(row)
    duration_ms = _duration_ms_created_updated(row)
    req_body: Dict[str, Any] = {
        "prompt": row.get("prompt"),
        "model": row.get("model"),
        "resolution": row.get("resolution"),
        "ratio": row.get("ratio"),
        "duration": row.get("duration"),
        "type": row.get("type"),
    }
    resp_body: Dict[str, Any] = {
        "success": ok,
        "video_url": row.get("url"),
        "doubao_url": row.get("doubao_url"),
        "obs_url": row.get("obs_url"),
        "task_id": row.get("taskId"),
        "error": row.get("error"),
    }
    st = (row.get("status") or "").lower()
    if ok:
        label = "成功"
    elif row.get("error") or st == "failed":
        label = "失败"
    else:
        label = "进行中"
    return {
        "id": row.get("id"),
        "taskId": row.get("taskId"),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": "POST /video",
        "httpMethod": "POST",
        "businessType": "VIDEO_GEN",
        "prompt_full": row.get("prompt"),
        "result_image_url": row.get("url") or row.get("obs_url") or row.get("doubao_url"),
        "requestUrl": "/video",
        "statusCode": 200 if ok else 500,
        "durationMs": duration_ms,
        "businessSuccess": ok,
        "businessStatusLabel": label,
        "errorMessage": row.get("error"),
        "timeDisplay": row.get("time"),
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
        "requestHeaders": {},
        "requestBody": req_body,
        "responseHeaders": {},
        "responseBody": resp_body,
        "note": "当前为根据历史表字段推断的合成详情（视频生成）；接入网关后可写入真实 request/response。",
    }




def _norm_va_success(row: Dict[str, Any]) -> bool:
    return (row.get("status") or "").strip().upper() == "SUCCESS"


def build_video_analysis_task_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    """由 video_analysis_history 行数据合成 HTTP 明细。"""
    ok = _norm_va_success(row)
    duration_ms = _duration_ms_created_updated(row)
    req_body: Dict[str, Any] = {
        "history_id": row.get("id"),
        "name": row.get("name"),
        "workspace": row.get("workspace"),
        "video_url": row.get("video_url"),
    }
    resp_body: Dict[str, Any] = {
        "status": row.get("status"),
        "error_msg": row.get("error_msg"),
    }
    st = (row.get("status") or "").strip().upper()
    if st == "SUCCESS":
        label = "成功"
    elif st == "FAILED":
        label = "失败"
    elif st in ("RUNNING", "PENDING"):
        label = "进行中"
    else:
        label = row.get("status") or "—"
    return {
        "id": row.get("id"),
        "taskId": row.get("id"),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": "POST /video-analysis",
        "httpMethod": "POST",
        "businessType": "VIDEO_ANALYSIS",
        "requestUrl": "/video-analysis",
        "statusCode": 200 if ok else 500,
        "durationMs": duration_ms,
        "businessSuccess": ok,
        "businessStatusLabel": label,
        "errorMessage": row.get("error_msg"),
        "timeDisplay": row.get("time"),
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
        "requestHeaders": {},
        "requestBody": req_body,
        "responseHeaders": {},
        "responseBody": resp_body,
        "note": "当前为根据历史表字段推断的合成详情；完整分镜卡片请使用视频分析页或 GET /video-analysis/history/{id}。",
    }


def build_video_match_job_task_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    """由 video_match_job 行合成「口播转写 / 解析」阶段的任务看板 HTTP 明细。"""
    ps = (row.get("parse_status") or "").strip().lower()
    duration_ms = _duration_ms_created_updated(row)
    script = row.get("script") or ""
    preview = str(script)[:8000] if script else ""
    req_body: Dict[str, Any] = {
        "job_id": row.get("id"),
        "script_preview": preview or None,
        "topic": row.get("topic"),
        "title": row.get("title"),
        "car_model": row.get("car_model"),
        "frame_size": row.get("frame_size"),
        "frame_orientation": row.get("frame_orientation"),
        "workspace": row.get("workspace"),
    }
    resp_body: Dict[str, Any] = {
        "parse_status": row.get("parse_status"),
        "parse_error": row.get("parse_error"),
        "search_status": row.get("search_status"),
        "search_error": row.get("search_error"),
    }
    if ps == "done":
        label = "成功"
        ok = True
    elif ps == "failed":
        label = "失败"
        ok = False
    elif ps in ("running", "pending"):
        label = "进行中"
        ok = False
    else:
        label = row.get("parse_status") or "—"
        ok = False
    return {
        "id": row.get("id"),
        "taskId": row.get("id"),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": "POST /video-match/jobs",
        "httpMethod": "POST",
        "businessType": "VIDEO_MATCH_PARSE",
        "requestUrl": "/video-match/jobs",
        "statusCode": 200 if ok else 500,
        "durationMs": duration_ms,
        "businessSuccess": ok,
        "businessStatusLabel": label,
        "errorMessage": row.get("parse_error"),
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
        "requestHeaders": {},
        "requestBody": req_body,
        "responseHeaders": {},
        "responseBody": resp_body,
        "note": "当前为根据 video_match_job 字段推断的合成详情；有 request_id 时合并 http_request_traces。",
    }


def build_video_match_shot_search_task_detail(
    *,
    job_id: str,
    shot_row_id: int,
    shot_order: int,
    segment_text_preview: str,
    opensearch_index: str = "car_interior_analysis_v2",
) -> Dict[str, Any]:
    """无 http_request_traces 行时的占位详情；有 trace 时由 merge_http_trace_into_detail 覆盖。"""
    preview = (segment_text_preview or "").strip()[:240]
    return {
        "id": f"{job_id}:{shot_row_id}",
        "taskId": str(shot_row_id),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": f"POST /internal/opensearch/{opensearch_index}/_search",
        "httpMethod": "POST",
        "businessType": "VIDEO_MATCH_SHOT_SEARCH",
        "requestUrl": f"/opensearch/{opensearch_index}/_search",
        "statusCode": None,
        "durationMs": None,
        "businessSuccess": None,
        "businessStatusLabel": "—",
        "errorMessage": None,
        "createdAt": None,
        "updatedAt": None,
        "requestHeaders": {},
        "requestBody": {
            "job_id": job_id,
            "shot_row_id": shot_row_id,
            "shot_order": shot_order,
            "segment_preview": preview or None,
        },
        "responseHeaders": {},
        "responseBody": {},
        "note": "执行「素材匹配」后写入 http_request_traces，并通过分镜行的 search_request_id 关联；无 rid 时仅展示占位字段。",
    }


def build_video_material_match_task_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    """由 video_material_match_history 行合成占位详情；有 request_id 时合并 http_request_traces。"""
    st = (row.get("status") or "").strip().lower()
    src = (row.get("source") or "").strip()
    duration_ms = _duration_ms_created_updated(row)
    if st == "done":
        label = "成功"
        ok = True
    elif st == "failed":
        label = "失败"
        ok = False
    elif st in ("running", "pending"):
        label = "进行中"
        ok = False
    else:
        label = row.get("status") or "—"
        ok = False
    bt = (
        "VIDEO_ANALYSIS_CARD_SEARCH"
        if src == "video_analysis_search"
        else "VIDEO_MATCH_SHOT_SEARCH"
    )
    req_body: Dict[str, Any] = {
        "material_match_id": row.get("id"),
        "source": src,
        "workspace": row.get("workspace"),
        "video_match_job_id": row.get("video_match_job_id"),
        "video_match_shot_row_id": row.get("video_match_shot_row_id"),
        "va_context_history_id": row.get("va_context_history_id"),
        "query_preview": row.get("query_preview"),
        "hit_count": row.get("hit_count"),
        "search_mode": row.get("search_mode"),
        "strategy_snapshot": row.get("strategy_snapshot"),
    }
    resp_body: Dict[str, Any] = {
        "status": row.get("status"),
        "hit_count": row.get("hit_count"),
        "top1_obs_url": row.get("top1_obs_url"),
        "elapsed_ms": row.get("elapsed_ms"),
        "error_message": row.get("error_message"),
    }
    return {
        "id": row.get("id"),
        "taskId": row.get("id"),
        "traceId": None,
        "parentTraceId": None,
        "serviceName": "my_bot_advance",
        "methodName": "POST /video-analysis/search"
        if src == "video_analysis_search"
        else "POST /internal/opensearch/_search",
        "httpMethod": "POST",
        "businessType": bt,
        "requestUrl": "/video-analysis/search"
        if src == "video_analysis_search"
        else "/opensearch/car_interior_analysis_v2/_search",
        "statusCode": 200 if ok else 500,
        "durationMs": duration_ms,
        "businessSuccess": ok,
        "businessStatusLabel": label,
        "errorMessage": row.get("error_message"),
        "createdAt": _iso(row.get("created_at")),
        "updatedAt": _iso(row.get("updated_at")),
        "requestHeaders": {},
        "requestBody": req_body,
        "responseHeaders": {},
        "responseBody": resp_body,
        "note": "素材匹配履历；有 request_id 时合并 http_request_traces。",
    }


def merge_http_trace_into_detail(
    base: Dict[str, Any],
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    将 http_request_traces 行合并进任务看板详情（camelCase 与 TaskBoardView 一致）。
    无 trace 时返回 base；有 trace 时覆盖同源字段，并附带 requestRid / 原始 snake 可序列化 dict。
    """
    if not trace:
        return base
    out = {**base}
    if trace.get("id"):
        out["requestRid"] = trace["id"]
    if trace.get("request_url"):
        out["requestUrl"] = trace["request_url"]
    if trace.get("http_method"):
        out["httpMethod"] = trace["http_method"]
    if trace.get("status_code") is not None:
        out["statusCode"] = trace["status_code"]
    if trace.get("duration_ms") is not None:
        out["durationMs"] = trace["duration_ms"]
    if trace.get("request_headers") is not None:
        out["requestHeaders"] = trace["request_headers"]
    if trace.get("response_headers") is not None:
        out["responseHeaders"] = trace["response_headers"]
    if trace.get("request_body") is not None:
        out["requestBody"] = trace["request_body"]
    if trace.get("response_body") is not None:
        out["responseBody"] = trace["response_body"]
    if trace.get("error_message"):
        out["errorMessage"] = trace["error_message"]
    if trace.get("trace_id"):
        out["traceId"] = trace["trace_id"]
    if trace.get("parent_trace_id"):
        out["parentTraceId"] = trace["parent_trace_id"]
    if trace.get("service_name"):
        out["serviceName"] = trace["service_name"]
    if trace.get("method_name"):
        out["methodName"] = trace["method_name"]
    if trace.get("business_type"):
        out["businessType"] = trace["business_type"]
    if trace.get("call_sequence") is not None:
        out["callSequence"] = trace["call_sequence"]
    if trace.get("process_id") is not None:
        out["processId"] = trace["process_id"]
    if trace.get("upstream_task_id"):
        out["upstreamTaskId"] = trace["upstream_task_id"]
    if trace.get("business_id"):
        out["businessId"] = trace["business_id"]
    if trace.get("business_success") is not None:
        out["businessSuccess"] = bool(trace["business_success"])
        out["businessStatusLabel"] = "成功" if trace["business_success"] else "失败"
    ta, ua = trace.get("created_at"), trace.get("updated_at")
    if ta is not None:
        out["httpTraceCreatedAt"] = _iso(ta)
    if ua is not None:
        out["httpTraceUpdatedAt"] = _iso(ua)
    out["httpTrace"] = trace
    out["note"] = "详情已合并表 http_request_traces（history.request_id → rid）；旧数据无 rid 时下方为历史表推断字段。"
    return out
