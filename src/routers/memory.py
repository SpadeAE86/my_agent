# routers/memory.py — 记忆管理路由
# 端点:
#   GET    /memory/{user_id}               — 获取用户长期记忆摘要
#   GET    /memory/{user_id}/sessions      — 列出所有会话的短期记忆
#   POST   /memory/{user_id}/search        — 搜索记忆内容 (RAG)
#   PUT    /memory/{user_id}/sync          — 触发云端同步
#   DELETE /memory/{user_id}/{session_id}  — 删除指定会话记忆
# 依赖: services.memory_service
