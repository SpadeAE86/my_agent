# core/agent/event.py — 统一事件 Schema
# 设计原则:
#   1. 事件是纯数据载体，不包含处理逻辑
#   2. 使用 Literal discriminator 实现类型安全的多态
#   3. 通过 AgentEvent 联合类型，可以用 Pydantic 自动序列化/反序列化
#   4. 所有事件共享 BaseEvent 的公共字段 (timestamp, agent_id)

from __future__ import annotations

from time import time
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ─── 基类 ───────────────────────────────────────────────────────
class BaseEvent(BaseModel):
    """所有事件的基类，仅定义公共字段，不包含业务逻辑。"""
    timestamp: float = Field(default_factory=time, description="事件发生的 Unix 时间戳")
    agent_id: str = Field(default="", description="产生该事件的 Agent ID")

    # event_type 由每个子类通过 Literal 覆盖，用于序列化时的类型判别
    # 不在基类声明，交给子类用 Literal 固定


# ─── 具体事件 ────────────────────────────────────────────────────
class UserMessage(BaseEvent):
    """用户输入事件"""
    event_type: Literal["user_message"] = "user_message"
    content: str = Field(..., description="用户消息内容")
    media: list[str] = Field(default_factory=list, description="附带的媒体文件路径")


class AgentThought(BaseEvent):
    """Agent 中间推理过程（可选暴露给前端）"""
    event_type: Literal["agent_thought"] = "agent_thought"
    content: str = Field(..., description="推理内容")


class TextChunk(BaseEvent):
    """流式文本输出片段"""
    event_type: Literal["text_chunk"] = "text_chunk"
    content: str = Field(..., description="文本片段")
    is_final: bool = Field(default=False, description="是否是最后一个片段")


class ToolCall(BaseEvent):
    """工具调用请求"""
    event_type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(..., description="本次调用的唯一 ID")
    tool_name: str = Field(..., description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="调用参数")


class ToolResult(BaseEvent):
    """工具执行结果"""
    event_type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(..., description="对应 ToolCall 的 ID")
    tool_name: str = Field(..., description="工具名称")
    output: str = Field(default="", description="执行输出")
    error: str | None = Field(default=None, description="错误信息（成功时为 None）")
    success: bool = Field(default=True)


class PlanUpdate(BaseEvent):
    """plan.md 的创建或更新"""
    event_type: Literal["plan_update"] = "plan_update"
    action: Literal["create", "update"] = Field(..., description="创建还是更新")
    content: str = Field(..., description="plan 的完整内容或 diff")


class TaskDispatch(BaseEvent):
    """子任务派发事件"""
    event_type: Literal["task_dispatch"] = "task_dispatch"
    task_id: str = Field(..., description="任务 ID")
    target_agent_id: str = Field(..., description="目标子 Agent ID")
    objective: str = Field(..., description="任务目标描述")
    tools: list[str] = Field(default_factory=list, description="分配的工具列表")


class TaskComplete(BaseEvent):
    """任务完成信号"""
    event_type: Literal["task_complete"] = "task_complete"
    task_id: str = Field(default="", description="任务 ID（主任务可空）")
    success: bool = Field(default=True)
    summary: str = Field(default="", description="完成摘要")
    reason: str = Field(default="", description="完成/终止原因")


class ErrorEvent(BaseEvent):
    """异常与降级事件"""
    event_type: Literal["error"] = "error"
    error_code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误描述")
    recoverable: bool = Field(default=True, description="是否可恢复")
    details: dict[str, Any] = Field(default_factory=dict)


class StatusUpdate(BaseEvent):
    """Agent 状态变更通知（给前端用于 UI 状态指示）"""
    event_type: Literal["status_update"] = "status_update"
    status: str = Field(..., description="当前状态: thinking / calling_tool / waiting / done")
    message: str = Field(default="", description="人类可读的状态描述")


# ─── 联合类型（核心！）──────────────────────────────────────────
# 用 Discriminated Union，Pydantic 可以根据 event_type 自动选择正确的子类反序列化
AgentEvent = Union[
    UserMessage,
    AgentThought,
    TextChunk,
    ToolCall,
    ToolResult,
    PlanUpdate,
    TaskDispatch,
    TaskComplete,
    ErrorEvent,
    StatusUpdate,
]