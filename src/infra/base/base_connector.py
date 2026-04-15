from abc import ABC, abstractmethod
from typing import Any


#所有需要预加载组件的基类
class BaseConnector(ABC):

    def __init__(self):
        self._client: Any = None
        self._cfg: Any = None

    # 创建单例的抽象方法，子类必须实现
    @abstractmethod
    async def init(self):
        pass

    # 关闭资源的抽象方法，子类必须实现
    @abstractmethod
    async def close(self):
        pass

    @property
    def client(self):
        """改用 property 访问，调用时 redis_manager.client 即可，更有『没存在感』的丝滑感"""
        if self._client is None:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized. Call init() first.")
        return self._client