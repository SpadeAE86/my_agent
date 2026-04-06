# core/tools/builtin/spawn_agent.py — 内置工具: 派生子 Agent
# 这是最关键的"元工具": 让主 Agent 可以创建子 Agent 来执行子任务。
#
# 设计参考 claude-code 的 AgentTool:
#   - 入参就是 prompt + 描述, 一步到位, 不需要单独的 createTask
#   - 子 Agent 运行完整的 main_loop, 结果作为 ToolOutput 返回
#   - 支持限定子 Agent 可用的工具列表 (安全隔离)

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput


# ─── 入参定义 ─────────────────────────────────────────────────────
class SpawnAgentInput(ToolInput):
    """spawn_agent 工具的入参。LLM 生成这些字段来创建子 Agent。"""

    prompt: str = Field(
        ...,
        description="分配给子 Agent 的完整任务描述。必须自包含，包含所有必要上下文，"
        "因为子 Agent 看不到父 Agent 的对话历史。",
    )
    description: str = Field(
        ...,
        description="3-5 个词的简短描述，用于 UI 展示和日志（如 '修复 auth 模块'）",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="子 Agent 可使用的工具白名单。为 None 时继承父 Agent 的全部工具"
        "（但不含 spawn_agent 自身，防止无限递归）。",
    )
    max_iterations: int = Field(
        default=10,
        description="子 Agent 的最大迭代轮次",
        ge=1,
        le=50,
    )
    mode: Literal["swarm", "coordinator"] = Field(
        default="swarm",
        description="子 Agent 的运行模式",
    )


# ─── 出参定义 ─────────────────────────────────────────────────────
class SpawnAgentOutput(ToolOutput):
    """spawn_agent 工具的出参。"""

    agent_id: str = Field(default="", description="子 Agent 的 ID")
    summary: str = Field(default="", description="子 Agent 完成任务后的摘要")
    iterations_used: int = Field(default=0, description="子 Agent 实际使用的迭代轮次")


# ─── 执行函数 ─────────────────────────────────────────────────────
async def handle_spawn_agent(
    params: SpawnAgentInput,
    *,
    agent: Any = None,       # 父 Agent 实例, 由 ToolManager.execute 通过 context 传入
    tool_manager: Any = None, # ToolManager 实例
    **_kwargs: Any,
) -> SpawnAgentOutput:
    """
    创建子 Agent 并运行其完整的 main_loop。

    核心流程 (参考 claude-code runAgent.ts):
    1. 从父 Agent 继承配置, 创建子 Agent 实例
    2. 确定子 Agent 的工具列表 (白名单或继承, 但排除 spawn_agent 自身)
    3. 运行子 Agent 的 main_loop, 收集所有事件
    4. 把子 Agent 的最终结果打包成 ToolOutput 返回给父 Agent
    """
    # 延迟导入, 避免循环依赖
    from core.agent.agent import Agent
    from core.agent.agent_loop import main_loop
    from core.agent.event import TaskComplete

    if agent is None:
        return SpawnAgentOutput(
            success=False,
            message="spawn_agent 必须在 Agent 上下文中调用",
        )

    # ── 1. 确定子 Agent 的工具列表 ──
    if params.allowed_tools is not None:
        sub_tools = params.allowed_tools
    else:
        # 继承父 Agent 的工具, 但排除 spawn_agent 自身 (防止递归炸弹)
        sub_tools = [t for t in agent.tools if t != "spawn_agent"]

    # ── 2. 创建子 Agent ──
    sub_agent = Agent(
        user_id=agent.user_id,
        llm=agent.llm,
        tools=sub_tools,
        skills=agent.skills,
        session_id=agent.session_id,  # 共享 session, 方便追溯
        max_iteration=params.max_iterations,
        mode=params.mode,
        is_base=False,  # 标记为子 Agent
        max_token=agent.max_token,
    )

    # 把子 Agent 注册到父 Agent 的管理列表
    agent.sub_agents.append(sub_agent)

    # ── 3. 运行子 Agent 的 main_loop ──
    summary = ""
    iterations_used = 0

    try:
        async for event in main_loop(sub_agent, params.prompt):
            iterations_used += 1

            # 子 Agent 完成时, 提取摘要
            if isinstance(event, TaskComplete):
                summary = event.summary or "子 Agent 已完成任务"
                break

            # 其他事件可以在这里做日志/转发给前端
            # 例如: logger.debug("SubAgent[%s] event: %s", sub_agent.agent_id, event.event_type)

    except Exception as e:
        return SpawnAgentOutput(
            success=False,
            message=f"子 Agent 执行失败: {type(e).__name__}: {e}",
            agent_id=sub_agent.agent_id,
            iterations_used=iterations_used,
        )

    # ── 4. 返回结果 ──
    return SpawnAgentOutput(
        success=True,
        message=summary,
        data={"description": params.description},
        agent_id=sub_agent.agent_id,
        summary=summary,
        iterations_used=iterations_used,
    )


# ─── 工具定义 (模块级变量, 供 ToolManager.auto_discover() 扫描) ──
tool_def = ToolDef(
    name="spawn_agent",
    description=(
        "派生一个子 Agent 来执行复杂的子任务。子 Agent 拥有独立的对话循环, "
        "可以调用工具、推理、多轮迭代直到完成任务。适用于: "
        "① 需要多步骤工具调用才能完成的研究类任务; "
        "② 可以并行处理的独立子任务; "
        "③ 需要隔离上下文以避免污染主对话的任务。"
        "注意: prompt 必须自包含所有上下文, 子 Agent 看不到父对话历史。"
    ),
    input_schema=SpawnAgentInput,
    output_schema=SpawnAgentOutput,
    handler=handle_spawn_agent,
    tags=["agent", "orchestration"],
    timeout=300.0,  # 子 Agent 可能需要较长时间
    is_concurrency_safe=True,  # 多个子 Agent 可以并发
)
