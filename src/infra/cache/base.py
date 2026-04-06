# infra/cache/base.py — Cache 抽象接口
# 定义:
#   class BaseCache(ABC):
#       get(key) -> Optional[Any]
#       set(key, value, ttl=None) -> None
#       delete(key) -> bool
#       exists(key) -> bool
#       clear() -> None
# 所有缓存实现 (Redis, Local) 必须继承此接口
