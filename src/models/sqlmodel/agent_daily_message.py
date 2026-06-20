from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, Text, func

class AgentDailyMessage(SQLModel, table=True):
    __tablename__ = "agent_daily_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    agent_id: str = Field(sa_column=Column(String(255), nullable=False, index=True))  # e.g., 'neuro'
    date_str: str = Field(sa_column=Column(String(50), nullable=False, index=True))  # 'YYYY-MM-DD'
    content: str = Field(sa_column=Column(Text, nullable=False))
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
