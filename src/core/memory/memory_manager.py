# core/memory/memory_manager.py — 记忆统一调度接口
# 职责:
#   1. 对外暴露统一的 read/write/search 方法
#   2. 根据请求类型路由到对应的记忆层 (short/mid/long)
#   3. 上下文组装: 为 prompt 自动拼接相关的短期 + 检索到的长期记忆
#   4. 管理 diskcache 锁, 协调多 Agent 并发访问
