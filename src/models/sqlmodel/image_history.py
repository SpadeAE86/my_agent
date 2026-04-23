from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict, List

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text, DateTime, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR


class ImageHistoryCard(SQLModel, table=True):
    """
    Store each generated item as a row.
    This maps closely to the frontend history card structure.
    """

    __tablename__ = "image_history_cards"

    # Use the same id the frontend uses
    id: str = Field(sa_column=Column(VARCHAR(64), primary_key=True, nullable=False))

    prompt: str = Field(sa_column=Column(Text, nullable=False))
    model: str = Field(sa_column=Column(Text, nullable=False))

    size: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    resolution: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    ratio: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    duration: Optional[int] = Field(default=None, nullable=True)

    # Keep both upstream (doubao) url and our mirrored obs url
    doubao_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    obs_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    time: str = Field(sa_column=Column(Text, nullable=False))
    type: str = Field(sa_column=Column(Text, nullable=False))

    referenceMedia: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
    )
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    taskId: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    status: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )

