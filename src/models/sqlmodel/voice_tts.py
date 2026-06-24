from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, Text, func

class VoiceTTSTask(SQLModel, table=True):
    __tablename__ = "voice_tts_task"

    id: Optional[int] = Field(default=None, primary_key=True)
    bubble_id: str = Field(sa_column=Column(String(128), index=True, nullable=False))
    voice_character: str = Field(sa_column=Column(String(64), index=True, nullable=False))
    full_text: str = Field(sa_column=Column(Text, nullable=False))
    oss_url: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    status: str = Field(sa_column=Column(String(32), default="pending", nullable=False)) # pending, completed, failed
    created_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    )

class VoiceTTSChunk(SQLModel, table=True):
    __tablename__ = "voice_tts_chunk"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(index=True, nullable=False)  # ForeignKey pointing to VoiceTTSTask.id
    chunk_index: int = Field(nullable=False)
    chunk_text: str = Field(sa_column=Column(String(255), index=True, nullable=False))
    platform_url: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now())
    )
