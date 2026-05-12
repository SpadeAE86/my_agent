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
