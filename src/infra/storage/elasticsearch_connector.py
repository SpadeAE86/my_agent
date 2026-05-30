from infra.logging.logger import logger as log
from elasticsearch import AsyncElasticsearch
from config.config import MY_CONFIG, ENV
from infra.base.base_connector import ResourceConnector


class ElasticsearchConnector(ResourceConnector):
    def __init__(self):
        # 继承基类的 _client, _cfg, _init_lock
        super().__init__()

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            return bool(await self._client.ping())
        except Exception as e:
            log.warning(f"Elasticsearch ping failed: {e}")
            return False

    async def init(self):
        """负责 Elasticsearch 异步客户端的正式初始化"""
        log.info("ElasticsearchConnector: Initializing Async Client...")

        # 获取配置
        cfg = MY_CONFIG.get("elasticsearch", {}).get(ENV, {})
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 9201)
        user = cfg.get("username", "elastic")
        pwd = cfg.get("password", "")
        use_ssl = bool(cfg.get("use_ssl", False))
        verify_certs = bool(cfg.get("verify_certs", use_ssl))

        scheme = "https" if use_ssl else "http"
        hosts = [f"{scheme}://{host}:{port}"]
        log.info(f"Elasticsearch 连接配置: hosts={hosts}, user={user}, use_ssl={use_ssl}, verify_certs={verify_certs}")
        
        self._client = AsyncElasticsearch(
            hosts=hosts,
            basic_auth=(user, pwd) if user and pwd else None,
            verify_certs=verify_certs,
            ssl_show_warn=False,  # 隐藏自签名证书警告
            sniff_on_start=False,
            request_timeout=300,
            max_retries=5,
            retry_on_timeout=True
        )
        await self.ping()
        log.info("ElasticsearchConnector: Async Client ready.")
        try:
            from infra.storage.elasticsearch.index_helper import print_elasticsearch_info
            await print_elasticsearch_info(self._client)
        except Exception as e:
            log.warning(f"Elasticsearch failed to print startup info: {e}")


    async def close(self):
        """优雅关闭"""
        if self._client:
            await self._client.close()
            self._client = None
            log.info("ElasticsearchConnector: Closed.")


# 全局唯一实例
elasticsearch_connector = ElasticsearchConnector()
