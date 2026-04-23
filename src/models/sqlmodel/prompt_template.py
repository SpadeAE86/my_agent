from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text, DateTime, func, String


class PromptTemplate(SQLModel, table=True):
    __tablename__ = "prompt_templates"

    id: Optional[int] = Field(default=None, primary_key=True)
    # MySQL cannot index TEXT without key length; use VARCHAR(255) for indexed/unique name.
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False, index=True))
    content: str = Field(sa_column=Column(Text, nullable=False))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )

