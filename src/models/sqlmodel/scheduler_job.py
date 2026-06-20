from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, Text, func

class SchedulerJob(SQLModel, table=True):
    __tablename__ = "scheduler_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Unique ID within APScheduler
    job_id: str = Field(sa_column=Column(String(255), unique=True, nullable=False, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    job_type: str = Field(sa_column=Column(String(128), nullable=False))  # e.g., 'buy_ticket', 'cleanup'
    trigger_type: str = Field(sa_column=Column(String(50), nullable=False))  # 'cron', 'date', 'interval'
    schedule_expr: str = Field(sa_column=Column(String(255), nullable=False))  # cron string, ISO date, or interval secs
    args_json: Optional[str] = Field(sa_column=Column(Text, nullable=True))  # JSON array parameters
    is_enabled: bool = Field(default=True, nullable=False)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )

class SchedulerJobLog(SQLModel, table=True):
    __tablename__ = "scheduler_job_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    job_type: str = Field(sa_column=Column(String(128), nullable=False))
    scheduled_run_time: Optional[datetime] = Field(sa_column=Column(DateTime(timezone=True), nullable=True))
    start_time: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    end_time: Optional[datetime] = Field(sa_column=Column(DateTime(timezone=True), nullable=True))
    status: str = Field(sa_column=Column(String(50), nullable=False))  # 'running', 'success', 'failed'
    duration_ms: Optional[float] = Field(default=None, nullable=True)
    error_message: Optional[str] = Field(sa_column=Column(Text, nullable=True))
