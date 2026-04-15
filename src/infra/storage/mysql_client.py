from asyncio import Lock
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from config.config import MY_CONFIG, ENV
from infra.base.base_connector import BaseConnector
from infra.logging.logger import logger as log


class MySQLConnector(BaseConnector):
    def __init__(self):
        super().__init__()
        self._engine = None
        # _client 此时就是 Session 工厂，它就是 SQL 组件的“核心入口”
        self._client: Optional[async_sessionmaker[AsyncSession]] = None
        self._init_lock = Lock()  # 异步锁，保护初始化过程

    async def init(self):
        if self._client is None:
            async with self._init_lock:
                if self._client is None:
                    log.info("MySQLConnector: Initializing...")
                    cfg = MY_CONFIG.get("mysql", {}).get(ENV, {})
                    url = f"mysql+aiomysql://{cfg['user']}:{cfg['pwd']}@{cfg['host']}:{cfg['port']}/{cfg['db']}"

                    self._engine = create_async_engine(url, pool_pre_ping=True)
                    # 这里的 _client 就是对外暴露的“出口”
                    self._client = async_sessionmaker(
                        self._engine,
                        expire_on_commit=False
                    )

    async def get_db_session(self):
        """
        通用数据库 Session 生成器。
        它只依赖 mysql_connector.client 这个『标准出口』。
        """
        # 这里直接调用标准的 .client 属性，就像 Redis 一样
        async with self.client() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def session_scope(self):
        async with self.client() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


    async def close(self):
        if self._engine:
            await self._engine.dispose()
            self._client = None
            log.info("MySQLConnector: Disposed.")


mysql_connector = MySQLConnector()