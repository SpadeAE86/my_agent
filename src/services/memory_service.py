# services/memory_service.py — 跨层记忆同步与管理
# 职责:
#   1. 提供统一的记忆读写接口 (屏蔽短/中/长期差异)
#   2. 协调 diskcache 与文件系统的一致性
#   3. 触发云端同步 (增量推送/拉取)
#   4. 处理记忆搜索请求, 调用 core.memory.retriever
