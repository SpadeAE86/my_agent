from typing import List

from infra.base.base_connector import ResourceConnector
from infra.cache.redis_connector import redis_connector
from infra.storage.mysql_connector import mysql_connector
from infra.mq.rabbitmq_connector import rabbitmq_connector
from infra.storage.opensearch_connector import opensearch_connector


class ResourceLoader:
    def __init__(self):
        self.connectors: List[ResourceConnector] = [
            redis_connector,
            mysql_connector,
            rabbitmq_connector,
            opensearch_connector
        ]

    async def startup(self):
        for connector in self.connectors:
            await connector.init()

    async def shutdown(self):
        # 逆序关闭，先开的后关
        for connector in reversed(self.connectors):
            await connector.close()

connector_loader = ResourceLoader()