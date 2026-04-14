import logging
from redis import ConnectionPool, Redis as SyncRedis
import redis.asyncio as aioredis
from config.config import MY_CONFIG, ENV

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis 管理器 (懒加载单例)"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance._sync_client = None
            cls._instance._async_client = None
            cls._instance._cfg = MY_CONFIG.get("redis", {}).get(ENV, {})
        return cls._instance

    @property
    def sync_client(self):
        """懒汉式获取同步连接"""
        if self._sync_client is None:
            logger.info("Initializing Redis Sync Pool (Lazy)...")
            pool = ConnectionPool(
                host=self._cfg.get("host", "127.0.0.1"),
                port=self._cfg.get("port", 6379),
                db=self._cfg.get("database", 0),
                password=self._cfg.get("password", ""),
                decode_responses=True,
                health_check_interval=30,  # 心跳保活
                max_connections=100
            )
            self._sync_client = SyncRedis(connection_pool=pool)
        return self._sync_client

    @property
    def async_client(self):
        """懒汉式获取异步连接"""
        if self._async_client is None:
            logger.info("Initializing Redis Async Pool (Lazy)...")
            self._async_client = aioredis.Redis(
                host=self._cfg.get("host", "127.0.0.1"),
                port=self._cfg.get("port", 6379),
                db=self._cfg.get("database", 0),
                password=self._cfg.get("password", ""),
                decode_responses=True,
                options={"health_check_interval": 30},  # 心跳保活
                max_connections=500
            )
        return self._async_client

    async def aclose(self):
        """留给服务退出时优雅释放资源的钩子"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
            
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

# 全局唯一管理器
redis_manager = RedisManager()
