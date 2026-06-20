from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, func

class TicketBuyer(SQLModel, table=True):
    __tablename__ = "ticket_buyers"

    id: Optional[int] = Field(default=None, primary_key=True)
    buyer_name: str = Field(sa_column=Column(String(255), nullable=False))
    tel: str = Field(sa_column=Column(String(20), nullable=False))
    sessdata: str = Field(sa_column=Column(String(512), nullable=False))
    device_id: Optional[str] = Field(sa_column=Column(String(255), nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )
