import asyncio
import aio_pika
from aio_pika.abc import AbstractRobustConnection
from aio_pika.pool import Pool
from contextlib import asynccontextmanager
from infra.base.base_connector import ResourceConnector
from infra.logging.logger import logger as log
from config.config import MY_CONFIG, ENV


class RabbitMQConnector(ResourceConnector):
    def __init__(self):
        super().__init__()
        # 在 RabbitMQ 中，_client 存储的是 Connection Pool
        self._pool: Pool = None
        self._url: str = ""

    async def ping(self) -> bool:
        if not self._pool:
            return False
        async with self._pool.acquire() as connection:
            if getattr(connection, "is_closed", False):
                return False
            channel = await connection.channel()
            await channel.close()
        return True

    async def _get_connection(self) -> AbstractRobustConnection:
        """底层方法：创建强壮的重连连接"""
        return await aio_pika.connect_robust(
            self._url,
            timeout=10,
            heartbeat=60
        )

    async def init(self):
        """负责 RabbitMQ 连接池的异步初始化"""
        log.info("RabbitMQConnector: Initializing Connection Pool...")

        cfg = MY_CONFIG.get("rabbitmq", {}).get(ENV, {})
        user = cfg.get("username", "guest")
        pwd = cfg.get("password", "guest")
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 5672)
        vhost = cfg.get("vhost", "/")
        use_ssl = bool(cfg.get("use_ssl", False))

        scheme = "amqps" if use_ssl else "amqp"
        self._url = f"{scheme}://{user}:{pwd}@{host}:{port}/{vhost.lstrip('/')}"

        # 获取当前运行的 loop，确保连接池绑定到正确的 loop 上
        loop = asyncio.get_running_loop()

        # 初始化 aio-pika 连接池
        self._pool = Pool(
            self._get_connection,
            max_size=10,
            loop=loop
        )
        # 将 pool 赋值给基类的 _client 属性，这样基类的 ensure_init 就能正确判断
        self._client = self._pool
        log.info("RabbitMQConnector: Pool ready.")
        await self.ping()

    @asynccontextmanager
    async def channel_scope(self, **channel_kwargs):
        """
        获取一个短生命周期的 Channel（类似 SQLAlchemy 的 session_scope）。
        用法：
            async with rabbitmq_connector.channel_scope() as channel:
                await channel.default_exchange.publish(...)
        """
        pool = await self.get_client()
        async with pool.acquire() as connection:
            channel = await connection.channel(**channel_kwargs)
            try:
                yield channel
            finally:
                await channel.close()

    def channel_pool(self, max_size: int = 10) -> Pool:
        """
        创建一个信道池。
        注意：RabbitMQ 建议复用 Connection，但频繁开关 Channel。
        这里返回一个基于当前连接池的信道池。
        """

        async def get_channel() -> aio_pika.Channel:
            pool = await self.get_client()
            async with pool.acquire() as connection:
                return await connection.channel()

        return Pool(get_channel, max_size=max_size, loop=asyncio.get_running_loop())

    async def close(self):
        """优雅关闭"""
        if self._client:
            await self._client.close()
            self._client = None
            log.info("RabbitMQConnector: Pool closed.")


# 全局唯一实例
rabbitmq_connector = RabbitMQConnector()