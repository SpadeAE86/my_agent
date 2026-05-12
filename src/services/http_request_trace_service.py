from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.http_request_trace import HttpRequestTrace


class HttpRequestTraceService:
    """http_request_traces 表：任务发起时 insert，结束时 finalize；history.request_id 指向本表 id。"""

    async def create_initial(
        self,
        *,
        request_url: str,
        http_method: str = "POST",
        request_headers: Optional[Dict[str, Any]] = None,
        request_body: Any = None,
        service_name: str = "my_bot_advance",
        method_name: Optional[str] = None,
        business_type: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        process_id: Optional[int] = None,
        upstream_task_id: Optional[str] = None,
        business_id: Optional[str] = None,
        call_sequence: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        rid = str(uuid.uuid4())
        async with mysql_connector.session_scope() as session:
            session.add(
                HttpRequestTrace(
                    id=rid,
                    request_url=request_url,
                    http_method=http_method,
                    request_headers=request_headers,
                    request_body=request_body,
                    service_name=service_name,
                    method_name=method_name,
                    business_type=business_type,
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    process_id=process_id,
                    upstream_task_id=upstream_task_id,
                    business_id=business_id,
                    call_sequence=call_sequence,
                    extra=extra,
                )
            )
            await session.commit()
        return rid

    async def finalize(
        self,
        rid: str,
        *,
        status_code: int,
        response_headers: Optional[Dict[str, Any]] = None,
        response_body: Any = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        business_success: Optional[bool] = None,
    ) -> None:
        key = (rid or "").strip()
        if not key:
            return
        async with mysql_connector.session_scope() as session:
            row = await session.get(HttpRequestTrace, key)
            if row is None:
                return
            row.status_code = status_code
            if response_headers is not None:
                row.response_headers = response_headers
            if response_body is not None:
                row.response_body = response_body
            if error_message is not None:
                row.error_message = error_message
            if duration_ms is not None:
                row.duration_ms = duration_ms
            if business_success is not None:
                row.business_success = business_success
            await session.commit()

    async def get_dict(self, rid: str) -> Optional[Dict[str, Any]]:
        key = (rid or "").strip()
        if not key:
            return None
        async with mysql_connector.session_scope() as session:
            row = await session.get(HttpRequestTrace, key)
            if row is None:
                return None
            return row.model_dump(exclude_none=True)


http_request_trace_service = HttpRequestTraceService()
