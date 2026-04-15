from asyncio import Lock
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from config.config import MY_CONFIG, ENV
from infra.base.base_connector import ResourceConnector
from infra.logging.logger import logger as log


class MySQLConnector(ResourceConnector):

    def __init__(self):
        super().__init__()
        self._engine = None
        # _client 此时就是 Session 工厂，它就是 SQL 组件的“核心入口”
        self._client: Optional[async_sessionmaker[AsyncSession]] = None


    async def init(self):
        log.info("MySQLConnector: Initializing...")
        cfg = MY_CONFIG.get("mysql", {}).get(ENV, {})
        url = f"mysql+aiomysql://{cfg['user']}:{cfg['pwd']}@{cfg['host']}:{cfg['port']}/{cfg['db']}"

        self._engine = create_async_engine(url, pool_pre_ping=True)
        # 这里的 _client 就是对外暴露的“出口”
        self._client = async_sessionmaker(
            self._engine,
            expire_on_commit=False
        )

    @property
    def engine(self):
        """改用 property 访问，调用时 redis_manager.engine 即可，更有『没存在感』的丝滑感"""
        if self._engine is None:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized. Call init() first.")
        return self._engine

    async def get_engine(self):
        """
        通用入口
        通用数据库连接池 Engine。
        """
        if self._engine is None:
            await self.ensure_init()
        return self._engine

    async def get_db_session(self):
        """
        通用入口
        通用数据库会话 Session 生成器。
        需要业务里面手动commit
        """
        if self._client is None:
            await self.ensure_init()
        # 这里直接调用标准的 .client 属性，就像 Redis 一样
        async with self.client() as session:
            try:
                yield session
                # await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def session_scope(self):
        """
        FastAPI入口
        通用数据库 Session 生成器。
        需要业务里面手动commit
        """
        if self._client is None:
            await self.ensure_init()
        async with self.client() as session:
            try:
                yield session
                # await session.commit()
            except Exception:
                await session.rollback()
                raise


    async def close(self):
        if self._engine:
            await self._engine.dispose()
            self._client = None
            log.info("MySQLConnector: Disposed.")


mysql_connector = MySQLConnector()