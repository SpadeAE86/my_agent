from abc import ABC, abstractmethod
from asyncio import Lock
from typing import Any


#所有需要预加载组件的基类
class ResourceConnector(ABC):

    def __init__(self):
        self._client: Any = None
        self._cfg: Any = None
        self._init_lock = Lock()  # 异步锁，保护初始化过程

    # 创建单例的抽象方法，子类必须实现
    @abstractmethod
    async def init(self):
        pass

    # 关闭资源的抽象方法，子类必须实现
    @abstractmethod
    async def close(self):
        pass

    async def get_client(self):
        """通用入口，支持懒加载"""
        await self.ensure_init()
        return self._client

    async def ensure_init(self):
        if self._client is None:
            async with self._init_lock:
                if self._client is None:
                    await self.init()

    @property
    def client(self):
        """改用 property 访问，调用时 redis_manager.client 即可，更有『没存在感』的丝滑感"""
        if self._client is None:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized. Call init() first.")
        return self._client
