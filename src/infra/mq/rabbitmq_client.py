import logging
import asyncio
import aio_pika
from config.config import MY_CONFIG, ENV

logger = logging.getLogger(__name__)

class RabbitMQManager:
    """RabbitMQ 管理器 (懒加载单例池)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RabbitMQManager, cls).__new__(cls)
            cls._instance._pool = None
            cls._instance._cfg = MY_CONFIG.get("rabbitmq", {}).get(ENV, {})
            
            user = cls._instance._cfg.get("username", "guest")
            pwd = cls._instance._cfg.get("password", "guest")
            host = cls._instance._cfg.get("host", "127.0.0.1")
            port = cls._instance._cfg.get("port", 5672)
            vhost = cls._instance._cfg.get("vhost", "/")
            
            cls._instance._url = f"amqp://{user}:{pwd}@{host}:{port}/{vhost.lstrip('/')}"
        return cls._instance

    async def _get_connection(self) -> aio_pika.RobustConnection:
        # connect_robust 内置防断网和恢复机制
        return await aio_pika.connect_robust(
            self._url,
            timeout=10,          
            heartbeat=60         
        )

    async def get_pool(self):
        """懒汉式异步获取连接池"""
        if self._pool is None:
            logger.info("Initializing RabbitMQ Pool (Lazy)...")
            # 必须在协程内获取 running_loop，避免跨 loop 污染
            loop = asyncio.get_running_loop()
            self._pool = aio_pika.pool.Pool(
                self._get_connection,
                max_size=10, 
                loop=loop
            )
        return self._pool

    async def acquire_channel(self):
        """获取一个可用流路 (Channel)"""
        pool = await self.get_pool()
        async with pool.acquire() as connection:
            return await connection.channel()
            
    async def close(self):
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None

# 全局唯一管理器
rabbitmq_manager = RabbitMQManager()
