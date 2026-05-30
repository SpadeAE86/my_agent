from infra.logging.logger import logger as log
from elasticsearch import AsyncElasticsearch


async def print_elasticsearch_info(client: AsyncElasticsearch):
    """
    启动时打印 Elasticsearch 中的所有索引信息：
    1. 当前有几条索引
    2. 索引的名字
    3. 每个索引里的数据量 (doc count)
    区分业务索引和系统索引 (以 . 开头)
    """
    try:
        # 获取索引元数据列表
        indices = await client.cat.indices(format="json", expand_wildcards="all")
        
        business_indices = []
        system_indices = []
        
        for idx in indices:
            name = idx.get("index", "")
            doc_count = idx.get("docs.count", "0")
            
            try:
                doc_count = int(doc_count)
            except (TypeError, ValueError):
                doc_count = 0
                
            info = {"name": name, "docs": doc_count}
            if name.startswith("."):
                system_indices.append(info)
            else:
                business_indices.append(info)
                
        log.info("=" * 50)
        log.info("[Elasticsearch] Startup Status Summary")
        log.info(f"Total Indices Found: {len(indices)} (Business: {len(business_indices)}, System: {len(system_indices)})")
        
        if business_indices:
            log.info("Business Indices:")
            for idx in business_indices:
                log.info(f"  - {idx['name']} ({idx['docs']} documents)")
        else:
            log.info("Business Indices: None")
            
        if system_indices:
            log.info(f"System Indices: {', '.join([x['name'] for x in system_indices])}")
        log.info("=" * 50)
        
    except Exception as e:
        log.error(f"Failed to fetch and print Elasticsearch info: {e}")
