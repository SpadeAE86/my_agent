from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, Integer, SmallInteger, Text, func

class VolcoTimbre(SQLModel, table=True):
    __tablename__ = "volco_timbre"

    id: Optional[int] = Field(default=None, primary_key=True)
    voice_character: str = Field(sa_column=Column(String(255), nullable=False, unique=True, index=True))
    voice_code: str = Field(sa_column=Column(String(255), nullable=False))
    voice_model_type: str = Field(sa_column=Column(String(64), nullable=False, server_default="default", index=True))
    is_enabled: bool = Field(sa_column=Column(SmallInteger, nullable=False, server_default="1"))
    note: Optional[str] = Field(sa_column=Column(String(255), nullable=True))
    priority: int = Field(sa_column=Column(Integer, nullable=False, server_default="0", index=True))
    age_type: Optional[str] = Field(sa_column=Column(String(32), nullable=True))
    sex: Optional[str] = Field(sa_column=Column(String(16), nullable=True))
    full_voice: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    )
