# infra/storage/diskcache_store.py — Diskcache (SQLite) 封装
# 职责:
#   1. 封装 diskcache.Cache, 提供 get/set/delete/expire 接口
#   2. 用作多 Agent 并发场景下的分布式锁
#   3. 索引记忆内容, 加速关键词检索
#   4. 管理 TTL (自动过期清理临时数据)
#   5. 数据库路径: data/cache/diskcache/
