# infra/cache/redis_cache.py — Redis 缓存实现
# 职责:
#   1. 实现 BaseCache 接口, 底层使用 redis-py
#   2. 支持序列化/反序列化 (JSON / pickle)
#   3. 连接池管理与自动重连
#   4. 可选: 生产环境使用, 本地开发可不启用
