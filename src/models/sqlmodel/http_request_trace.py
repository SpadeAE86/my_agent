from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR


class HttpRequestTrace(SQLModel, table=True):
    """
    通用 HTTP / 上游调用记录（任务看板详情联表）。
    history 表通过 request_id 引用本表主键（UUID）。
    """

    __tablename__ = "http_request_traces"

    id: str = Field(sa_column=Column(VARCHAR(36), primary_key=True, nullable=False))

    request_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    http_method: str = Field(default="POST", sa_column=Column(VARCHAR(16), nullable=False))
    status_code: Optional[int] = Field(default=None, nullable=True)

    request_headers: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(MySQLJSON, nullable=True)
    )
    response_headers: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(MySQLJSON, nullable=True)
    )
    request_body: Optional[Any] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))
    response_body: Optional[Any] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    duration_ms: Optional[int] = Field(default=None, nullable=True)

    trace_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), nullable=True))
    parent_trace_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), nullable=True))

    service_name: str = Field(
        default="my_bot_advance", sa_column=Column(VARCHAR(128), nullable=False)
    )
    method_name: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(256), nullable=True))
    business_type: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(64), nullable=True))
    call_sequence: int = Field(default=0, nullable=False)

    # 对齐外部「调用记录」后台常见维度（与 OpenTelemetry trace 无关的业务编号）
    process_id: Optional[int] = Field(default=None, nullable=True)
    upstream_task_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(128), nullable=True))
    business_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(64), nullable=True))
    business_success: Optional[bool] = Field(default=None, nullable=True)

    extra: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(MySQLJSON, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
        )
    )
