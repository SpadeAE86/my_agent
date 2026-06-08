from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text, DateTime, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR


class Workspace(SQLModel, table=True):
    """
    画布工程的主表
    """
    __tablename__ = "workspaces"

    id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(64), primary_key=True),
    )
    name: str = Field(sa_column=Column(VARCHAR(255), nullable=False))
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class WorkspaceNode(SQLModel, table=True):
    """
    画布节点表
    """
    __tablename__ = "workspace_nodes"

    id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(64), primary_key=True),
    )
    workspace_id: str = Field(sa_column=Column(VARCHAR(64), nullable=False, index=True))
    type: str = Field(sa_column=Column(VARCHAR(50), nullable=False))
    
    # 坐标位置
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)

    # 业务数据 payload (image_url, prompt, status 等)
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )


class WorkspaceEdge(SQLModel, table=True):
    """
    画布边（连接线）表
    """
    __tablename__ = "workspace_edges"

    id: Optional[str] = Field(
        default=None,
        sa_column=Column(VARCHAR(64), primary_key=True),
    )
    workspace_id: str = Field(sa_column=Column(VARCHAR(64), nullable=False, index=True))
    
    source_node_id: str = Field(sa_column=Column(VARCHAR(64), nullable=False))
    target_node_id: str = Field(sa_column=Column(VARCHAR(64), nullable=False))
    
    # 连接线上的标签/演进描述
    label: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    )
