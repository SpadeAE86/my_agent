# core/agent/ — Agent 实体与驱动循环
# 子模块:
#   main_agent  — 主 Agent: 拥有规划、创建子 Agent、修改 plan/task 的完整权限
#   sub_agent   — 子 Agent: 仅拥有被分配的工具与上下文, 无规划权限
#   agent_loop  — Generator 驱动的核心循环: 接收事件 → 决策 → 执行 → yield 事件
#   event       — 统一事件 Schema (UserMessage, ToolCall, ToolResult, AgentThought, PlanUpdate...)
