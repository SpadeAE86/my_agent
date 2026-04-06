# infra/logging/trace.py — Agent 执行链路追踪
# 职责:
#   1. 记录 Agent 的完整决策链 (input → think → tool_call → result → output)
#   2. 为每次 agent_loop 迭代生成唯一 trace_id
#   3. 支持可视化回放 (后续对接前端 timeline 组件)
#   4. 标记异常路径与重试节点
