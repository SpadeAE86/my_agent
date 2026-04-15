# Redis的可复用异步Manager，使用前需要手动init, 然后才能访问client属性，需要await close()释放
from infra.base.base_connector import BaseConnector
from infra.logging.logger import logger as log
from asyncio import Lock
from typing import Optional
from redis.asyncio import Redis
from config.config import MY_CONFIG, ENV

# 1. RedisConnector 负责管理 Redis 连接池，提供安全的异步初始化和优雅关闭
class RedisConnector(BaseConnector):
    def __init__(self):
        # 1. 所有的变量声明都在 init 里，一清二楚
        super().__init__()
        self._client: Optional[Redis] = None
        self._init_lock = Lock()  # 异步锁，保护初始化过程

    async def init(self):
        """负责安全的异步初始化"""
        #双重锁校验，节省不必要的获取锁开销
        if self._client is None:
            async with self._init_lock:
                if self._client is None:
                    log.info("Initializing Redis Async Pool...")
                    self._cfg = MY_CONFIG.get("redis", {}).get(ENV, {})
                    self._client = Redis(
                        host=self._cfg.get("host", "127.0.0.1"),
                        port=self._cfg.get("port", 6379),
                        password=self._cfg.get("password"),
                        db=self._cfg.get("database", 0),
                        decode_responses=True,
                        max_connections=512,
                        health_check_interval=30,
                        retry_on_timeout=True
                    )

    async def close(self):
        """优雅关闭"""
        if self._client:
            await self._client.aclose()
            self._client = None
            log.info("Redis async client closed.")

redis_connector = RedisConnector()