from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text, DateTime, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON, VARCHAR


class ThemeSpace(SQLModel, table=True):
    """
    主题空间：用户自定义的分类目录，含有特定主题的标签池（如：赛博朋克、女仆cos）
    """
    __tablename__ = "theme_spaces"

    id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), primary_key=True))
    name: str = Field(sa_column=Column(VARCHAR(100), nullable=False))
    
    # 空间类别 (media: 图片视频 | template: 提示词模板 | prompt: 纯文本提示词)
    category: str = Field(sa_column=Column(VARCHAR(20), nullable=False, default="media"))
    description: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    
    # 空间专属的标准标签列表
    space_tags: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
        description="空间内置的候选标签池"
    )
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )


class CollectionItem(SQLModel, table=True):
    """
    万能收藏项：支持多种数据实体类型（图像、视频、提示词模板、纯提示词）
    """
    __tablename__ = "collection_items"

    id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), primary_key=True))
    space_id: Optional[str] = Field(default=None, sa_column=Column(VARCHAR(36), nullable=True), description="所属主题空间ID")
    
    # 类型 (media / template / prompt)
    item_type: str = Field(sa_column=Column(VARCHAR(20), nullable=False))
    title: str = Field(sa_column=Column(VARCHAR(255), nullable=False))
    cover_url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True), description="封面图/视频预览封面")
    
    # 核心数据载荷 (JSON)。例如：
    # - media: { "url": "xxx", "source": "generate", "prompt": "xxx", "model": "xxx" }
    # - template: { "template_text": "xxx", "name": "xxx" }
    # - prompt: { "prompt": "xxx" }
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
        description="收藏对象的多态数据体"
    )
    
    # 自定义标签
    tags: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(MySQLJSON, nullable=True),
        description="用户为该收藏打的标签"
    )
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
