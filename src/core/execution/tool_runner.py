# core/execution/tool_runner.py — 工具调用运行时
# 职责:
#   1. 接收 agent_loop 产生的 ToolCall 事件
#   2. 通过 registry 查找工具 → 通过 executor 安全执行
#   3. 将执行结果包装为 ToolResult 事件返回给 agent_loop
#   4. 处理并行工具调用 (多个 ToolCall 同时执行)
