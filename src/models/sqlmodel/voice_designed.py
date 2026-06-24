from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, String, Integer, SmallInteger, Text, func

class VoiceDesigned(SQLModel, table=True):
    __tablename__ = "voice_designed"

    id: Optional[int] = Field(default=None, primary_key=True)
    voice_character: str = Field(sa_column=Column(String(255), nullable=False, unique=True, index=True))
    custom_speaker_id: str = Field(sa_column=Column(String(255), nullable=False, unique=True, index=True))
    base_speaker_id: str = Field(sa_column=Column(String(255), nullable=False))
    text_prompt: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    image_url: Optional[str] = Field(sa_column=Column(String(512), nullable=True))
    status: int = Field(sa_column=Column(SmallInteger, nullable=False, server_default="1")) # 1=Designing/Pending, 2=Success, 3=Failed, 4=Active
    language: int = Field(sa_column=Column(SmallInteger, nullable=False, server_default="0"))
    demo_audio_url: Optional[str] = Field(sa_column=Column(String(512), nullable=True))
    error_message: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    provider: str = Field(sa_column=Column(String(64), nullable=False, server_default="volcano"))
    is_online: bool = Field(sa_column=Column(SmallInteger, nullable=False, server_default="1"))
    tag: Optional[str] = Field(sa_column=Column(String(255), nullable=True))
    note: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    age_type: Optional[str] = Field(sa_column=Column(String(64), nullable=True))
    sex: Optional[str] = Field(sa_column=Column(String(64), nullable=True))
    available_training_times: Optional[int] = Field(sa_column=Column(Integer, nullable=True))
    update_to: Optional[int] = Field(sa_column=Column(Integer, nullable=True))
    ref_audio_url: Optional[str] = Field(sa_column=Column(String(512), nullable=True))
    demo_text: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    )
