# core/agent/agent_loop.py — Generator 核心驱动循环
# 职责:
#   1. 以 async generator 形式驱动 Agent 的完整生命周期
#   2. 循环: 组装 prompt → 调用 LLM → 解析响应 → 执行工具/生成事件 → yield 事件
#   3. 内置安全阀: 最大轮次限制、重复错误检测
# 事件流:
#   yield AgentThought   — Agent 的思考过程
#   yield ToolCall        — 工具调用请求 (先 yield, 再执行, 再 yield 结果)
#   yield ToolResult      — 工具执行结果
#   yield TextChunk       — 流式文本片段
#   yield PlanUpdate      — plan.md 变更通知
#   yield TaskComplete    — 任务完成信号

from __future__ import annotations

from typing import AsyncGenerator, TYPE_CHECKING

from core.agent.event import (
    AgentEvent, AgentThought, TextChunk, ToolCall,
    ToolResult, PlanUpdate, TaskComplete, StatusUpdate,
)

if TYPE_CHECKING:
    from core.agent.agent import Agent
    from core.tools.tool_manager import ToolManager


async def main_loop(
    agent: Agent,
    initial_input: str,
    tool_manager: ToolManager | None = None,
    reference_image_list: list[str] | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Agent 的核心驱动循环。

    :param agent: Agent 实例
    :param initial_input: 用户的初始请求
    :param tool_manager: 工具管理器 (为 None 时使用 agent.tool_manager)
    :param reference_image_list: 初始参考图公网URL列表
    """
    tm = tool_manager or getattr(agent, "tool_manager", None)

    iteration = 0

    while iteration < agent.max_iterations:
        iteration += 1

        # 1. 组装 prompt (第一轮传用户输入和参考图片, 后续续轮传 None — tool result 已在 messages 里)
        user_input = initial_input if iteration == 1 else None
        ref_images = reference_image_list if iteration == 1 else None
        prompt = await agent.build_prompt(user_input, ref_images)

        # Check if compaction occurred during build_prompt and yield event
        if getattr(agent, "just_compacted", False):
            from core.agent.event import SessionCompacted
            yield SessionCompacted(
                summary=getattr(agent, "compaction_summary", ""),
                agent_id=agent.agent_id
            )
            agent.just_compacted = False

        # 2. 调用 LLM 并流式处理所有推理和文本块
        yield StatusUpdate(status="thinking", message=f"第 {iteration} 轮推理中...")
        
        async for event in agent.call_llm_stream(prompt):
            if isinstance(event, (AgentThought, TextChunk)):
                yield event
                
            elif isinstance(event, ToolCall):
                # 先 yield ToolCall (通知前端"要调工具了")
                yield event
                
                # 执行工具
                if tm is not None:
                    tool_result = await tm.execute(
                        event,
                        agent=agent,           # 传给 handler 的上下文
                        tool_manager=tm,       # spawn_agent 需要它来给子 Agent 用
                    )
                else:
                    tool_result = ToolResult(
                        call_id=event.call_id,
                        tool_name=event.tool_name,
                        output="",
                        error="ToolManager 未初始化",
                        success=False,
                        agent_id=event.agent_id,
                    )
                
                # yield ToolResult (通知前端"工具执行完了")
                yield tool_result
                
                # 把结果注入 Agent 的消息历史，下一轮 LLM 能看到
                agent.handle_tool_result(tool_result)
                
            elif isinstance(event, PlanUpdate):
                yield event
                
            elif isinstance(event, TaskComplete):
                yield event
                return  # 任务完成, 退出循环

        # 5. 安全阀检查
        if agent.should_terminate():
            yield TaskComplete(
                success=False,
                reason=f"安全阀触发: 已达 {iteration} 轮",
                agent_id=agent.agent_id,
            )
            return

    # 超过最大迭代次数
    yield TaskComplete(
        success=False,
        reason=f"已达最大迭代次数 {agent.max_iterations}",
        agent_id=agent.agent_id,
    )