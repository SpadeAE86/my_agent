# routers/agent.py — Agent 管理路由
# 端点:
#   POST   /agent/create   — 手动创建子 Agent 实例
#   GET    /agent/status    — 查看当前所有 Agent 状态
#   DELETE /agent/{id}      — 停止并销毁指定 Agent
# 依赖: services.agent_service
