# core/tools/tool_schema.py — 工具 Schema 基类
# 设计原则:
#   每个字段必须有 Field(description=...) 描述, 方便 LLM 理解
#   通过 .model_json_schema() 自动生成符合 function calling 格式的 JSON

from __future__ import annotations

from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


# ─── 工具入参/出参基类 ───────────────────────────────────────────
class ToolInput(BaseModel):
    """所有工具入参的 Pydantic 基类，子类通过扩展字段定义具体参数。"""
    pass


class ToolOutput(BaseModel):
    """所有工具出参的 Pydantic 基类。"""
    success: bool = Field(default=True, description="执行是否成功")
    message: str = Field(default="", description="人类可读的结果描述")
    data: dict[str, Any] = Field(default_factory=dict, description="结构化返回数据")


# ─── 工具定义 ────────────────────────────────────────────────────
class ToolDef(BaseModel):
    """完整的工具定义，注册到 ToolManager 后可被 LLM 发现和调用。"""
    name: str = Field(..., description="工具唯一标识符")
    description: str = Field(..., description="工具功能描述（给 LLM 看）")
    input_schema: type[ToolInput] = Field(..., description="入参 Pydantic 模型类")
    output_schema: type[ToolOutput] = Field(default=ToolOutput, description="出参 Pydantic 模型类")
    handler: Callable[..., Awaitable[ToolOutput]] = Field(..., description="异步执行函数")
    tags: list[str] = Field(default_factory=list, description="分类标签 (如 'file', 'search', 'agent')")
    timeout: float = Field(default=30.0, description="最大执行时间 (秒)")
    is_concurrency_safe: bool = Field(default=True, description="是否可与其他工具并发执行")

    model_config = {"arbitrary_types_allowed": True}

    def to_llm_schema(self) -> dict[str, Any]:
        """生成 OpenAI/Anthropic function calling 格式的 JSON schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }
