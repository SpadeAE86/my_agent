import logging
from opensearchpy import OpenSearch, AsyncOpenSearch
from config.config import MY_CONFIG, ENV

logger = logging.getLogger(__name__)

class OpenSearchManager:
    """OpenSearch 管理器 (懒加载单例)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenSearchManager, cls).__new__(cls)
            cls._instance._client = None
            cls._instance._cfg = MY_CONFIG.get("opensearch", {}).get(ENV, {})
        return cls._instance

    @property
    def client(self) -> OpenSearch:
        if self._client is None:
            logger.info("Initializing OpenSearch Client (Lazy)...")
            host = self._cfg.get("host", "127.0.0.1")
            port = self._cfg.get("port", 9200)
            user = self._cfg.get("username", "admin")
            pwd = self._cfg.get("password", "")
            
            # sniff_on_connection_fail 是断线重嗅探健康节点的关键
            self._client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                http_auth=(user, pwd),
                use_ssl=False,               
                verify_certs=False,
                sniff_on_start=False,        
                sniff_on_connection_fail=True, 
                sniffer_timeout=60,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
        return self._client

# 全局唯一管理器
opensearch_manager = OpenSearchManager()
