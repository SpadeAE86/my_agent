# infra/cache/ — 缓存层
# 子模块:
#   base        — Cache 抽象接口 (get, set, delete, exists, expire)
#   redis_cache — Redis 实现 (生产环境, 可选)
#   local_cache — 本地内存缓存 (开发环境, 基于 dict + TTL)
