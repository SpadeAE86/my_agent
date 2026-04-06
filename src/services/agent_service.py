# services/agent_service.py — Agent 实例化与生命周期管理
# 职责:
#   1. 创建主 Agent 或子 Agent 实例
#   2. 维护活跃 Agent 注册表 (内存中)
#   3. 处理 Agent 的启动、暂停、恢复、销毁
#   4. 子 Agent 完成后回收资源并通知主 Agent
