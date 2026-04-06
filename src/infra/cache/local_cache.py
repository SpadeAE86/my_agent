# infra/cache/local_cache.py — 本地内存缓存实现
# 职责:
#   1. 实现 BaseCache 接口, 底层使用 dict + TTL 管理
#   2. 适用于开发环境和单进程场景
#   3. 定期清理过期 key (被动过期 + 主动扫描)
#   4. 无需外部依赖, 零配置可用
