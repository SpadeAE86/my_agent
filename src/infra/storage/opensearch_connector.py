from infra.logging.logger import logger as log
from opensearchpy import AsyncOpenSearch
from config.config import MY_CONFIG, ENV
from infra.base.base_connector import ResourceConnector


class OpenSearchConnector(ResourceConnector):
    def __init__(self):
        # 继承基类的 _client, _cfg, _init_lock
        super().__init__()

    async def ping(self) -> bool:
        if not self._client:
            return False
        return bool(await self._client.ping())

    async def init(self):
        """负责 OpenSearch 异步客户端的正式初始化"""
        log.info("OpenSearchConnector: Initializing Async Client...")

        # 获取配置
        cfg = MY_CONFIG.get("opensearch", {}).get(ENV, {})
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 9200)
        user = cfg.get("username", "admin")
        pwd = cfg.get("password", "")
        use_ssl = bool(cfg.get("use_ssl", False))
        verify_certs = bool(cfg.get("verify_certs", use_ssl))

        # 注意：这里切换为 AsyncOpenSearch 以匹配整体异步架构
        self._client = AsyncOpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(user, pwd),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            sniff_on_start=False,
            sniff_on_connection_fail=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        await self.ping()
        log.info("OpenSearchConnector: Async Client ready.")

    async def close(self):
        """优雅关闭"""
        if self._client:
            # AsyncOpenSearch 需要调用 .close() 释放底层连接池
            await self._client.close()
            self._client = None
            log.info("OpenSearchConnector: Closed.")


# 全局唯一实例
opensearch_connector = OpenSearchConnector()
