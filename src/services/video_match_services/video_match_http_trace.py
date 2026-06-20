"""HTTP 请求追踪 payload 压缩与裁剪（视频 / 素材匹配 OpenSearch 等）。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


def truncate_for_trace(obj: Any, max_bytes: int = 28000) -> Any:
    """避免 request_body 撑爆 JSON 列；尽量保留结构，超长时改为预览字符串。"""
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"_error": "non_json_serializable"}
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return obj
    cut = max_bytes - 120
    pref = s[:cut] if cut > 0 else ""
    return {"_truncated": True, "utf8_preview": pref + "…"}


def hits_for_db_with_truncated_explain(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    仅用于写入 MySQL 的命中列表：第 6 条起去掉 _explanation 以省体积；
    前 5 条保留 OpenSearch 返回的完整 explain 树（不再做字节截断），便于详情里对照 Top5 渲染。

    使用 deepcopy，不修改入参；管线里内存中的 enriched 仍可用于未落库场景下的完整展示。
    """
    out = copy.deepcopy(hits)
    for i, h in enumerate(out):
        if not isinstance(h, dict):
            continue
        if i >= 5:
            h.pop("_explanation", None)
    return out


def trace_response_top_hits_with_explain(enriched: List[Dict[str, Any]]) -> Any:
    """HTTP trace 摘要：仅前 5 条附带完整 _explanation，其余条仅存分数/ id；整体再套字节上限。"""
    slim: List[Dict[str, Any]] = []
    for i, h in enumerate(enriched):
        if not isinstance(h, dict):
            continue
        ex = h.get("_explanation") if i < 5 else None
        slim.append(
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "history_id": h.get("history_id"),
                "video_path": (str(h.get("video_path") or "").strip()[:512] or None),
                "_explanation": ex,
            }
        )
    return truncate_for_trace(slim, max_bytes=150_000)


def strip_large_numeric_vectors(obj: Any, *, max_list_len: int = 48) -> Any:
    """
    OpenSearch hybrid / KNN 请求体里的向量动辄数千维；写入 http_request_traces 前替换为占位，
    保留可读的 query 结构与短前缀，避免整条变成无法解析的 utf8 裁剪串。
    注意：业务重试匹配由 match_script_tags_segments(tags_json) 重新构造请求，从不反序列化本条入库体。
    """
    if isinstance(obj, list):
        if len(obj) > max_list_len:
            head_len = min(8, len(obj))
            head = obj[:head_len]
            if head and all(isinstance(x, (int, float)) for x in head):
                return {
                    "_omitted": "numeric_vector",
                    "length": len(obj),
                    "head_preview": [round(float(x), 6) for x in head],
                }
        return [strip_large_numeric_vectors(x, max_list_len=max_list_len) for x in obj]
    if isinstance(obj, dict):
        return {str(k): strip_large_numeric_vectors(v, max_list_len=max_list_len) for k, v in obj.items()}
    return obj


def trace_request_body_for_shot_search(m: Dict[str, Any], **extra: Any) -> Any:
    """
    分镜检索阶段写入 HTTP trace 的 request_body。

    语义对齐"HTTP trace 用于重试/重发"的设计意图：
    - request_body 存储「前端发给后端的请求参数」（job_id + 标签字段），
      而非 OpenSearch 内部查询体（后者体积大，由调用方 log.debug 打印）。
    - opensearch_body 由调用方在 log.debug 中打印，trace 里仅保留可读摘要。
    """
    payload: Dict[str, Any] = {
        "query_text": m.get("query_text"),
        "search_params": m.get("search_params"),
        "tags_json": m.get("tags_json") or m.get("segment"),
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return truncate_for_trace({k: v for k, v in payload.items() if v is not None})


def opensearch_body_for_debug_log(m: Dict[str, Any]) -> Any:
    """供调用方 log.debug 打印 OpenSearch 请求体（向量已压缩，体积尚可）。"""
    compact = strip_large_numeric_vectors({"opensearch_body": m.get("opensearch_body")})
    return truncate_for_trace(compact, max_bytes=8000)
