# core/state/agent_state.py — Agent 内部状态
# 字段:
#   agent_id     — 唯一标识
#   role         — "main" | "sub"
#   status       — "idle" | "thinking" | "tool_calling" | "waiting" | "done" | "error"
#   tools        — 当前可用工具列表
#   turn_count   — 已执行轮次 (用于安全阀)
#   token_used   — 已消耗 token 量
#   task_ref     — 关联的 task.json 引用 (子 Agent)
