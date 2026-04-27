from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScriptMatchRequest(BaseModel):
    script: str = Field(..., description="口播/混剪脚本（长文本）", min_length=1)
    top_k: int = Field(5, description="每个分镜返回 topK 命中", ge=1, le=50)
    search_pipeline: Optional[str] = Field("nlp-search-pipeline", description="OpenSearch search_pipeline；为空则不传")
    mode: str = Field("lite", description="检索模式：lite(1 BM25 + 1 KNN) / full(1 BM25 + 多路KNN)", pattern="^(lite|full)$")

