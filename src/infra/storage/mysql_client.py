import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from config.config import MY_CONFIG, ENV

logger = logging.getLogger(__name__)

class MySQLManager:
    """MySQL 管理器 (懒加载单例)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MySQLManager, cls).__new__(cls)
            cls._instance._sync_engine = None
            cls._instance._async_engine = None
            cls._instance._sync_session_factory = None
            cls._instance._async_session_factory = None
            
            cfg = MY_CONFIG.get("mysql", {}).get(ENV, {})
            user = cfg.get("username", "root")
            pwd = cfg.get("password", "")
            host = cfg.get("host", "127.0.0.1")
            port = cfg.get("port", 3306)
            db = cfg.get("database", "")
            
            cls._instance._sync_url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
            cls._instance._async_url = f"mysql+aiomysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
        return cls._instance

    @property
    def sync_session_factory(self):
        if self._sync_session_factory is None:
            logger.info("Initializing Sync MySQL Engine (Lazy)...")
            self._sync_engine = create_engine(
                self._sync_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,      # 二次确认连接是否有效
                pool_recycle=3600,       # 回收超时连接
                echo=False
            )
            self._sync_session_factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self._sync_engine))
        return self._sync_session_factory

    @property
    def async_session_factory(self):
        if self._async_session_factory is None:
            logger.info("Initializing Async MySQL Engine (Lazy)...")
            self._async_engine = create_async_engine(
                self._async_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            self._async_session_factory = sessionmaker(self._async_engine, class_=AsyncSession, expire_on_commit=False)
        return self._async_session_factory

    def get_sync_session(self):
        """FastAPI Dependency 注入所用"""
        db = self.sync_session_factory()
        try:
            yield db
        finally:
            db.close()
            
    async def get_async_session(self):
        """FastAPI Dependency 注入所用 (Async)"""
        async with self.async_session_factory() as session:
            yield session

# 全局唯一管理器
mysql_manager = MySQLManager()
