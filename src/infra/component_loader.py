from typing import List

from infra.base.base_connector import BaseConnector
from infra.cache.redis_connector import redis_connector
from infra.storage.mysql_client import mysql_manager


class ComponentLoader:
    def __init__(self):
        self.managers: List[BaseConnector] = [
            redis_connector,
            mysql_manager,
            rabbitmq_manager,
            opensearch_manager
        ]

    async def startup(self):
        for mgr in self.managers:
            await mgr.init()

    async def shutdown(self):
        # 逆序关闭，先开的后关
        for mgr in reversed(self.managers):
            await mgr.close()

loader = ComponentLoader()