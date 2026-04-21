import sys
import asyncio
sys.path.append('C:\\AI\\AiGithubProject\\DIYProject\\src')

from infra.storage.opensearch.create_index import index_manager
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis

async def main():
    # 1. 定义索引设置 (必须开启 knn 才能支持向量检索)
    settings = {
        "index": {
            "knn": True,  # 开启向量检索支持
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    # 2. 定义字段映射 (OpenSearch 的 mapping)
    # 根据 CarInteriorAnalysis 的字段类型，映射到 OpenSearch 的数据类型
    field_types = {
        "id": {"type": "keyword"},
        "description": {"type": "text", "analyzer": "standard"},
        "subject": {"type": "text", "analyzer": "standard"},
        "object": {"type": "keyword"},
        "movement": {"type": "text", "analyzer": "standard"},
        "adjective": {"type": "keyword"},
        "search_tags": {"type": "keyword"},
        "marketing_tags": {"type": "keyword"},
        "appealing_audience": {"type": "keyword"},
        "visual_quality": {"type": "float"},
        
        # 向量字段映射 (维度 384 是根据你代码里的 [0.0] * 384 推断的，通常对应 all-MiniLM-L6-v2 等模型)
        "description_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene"
            }
        },
        "subject_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene"
            }
        },
        "combined_vector": {
            "type": "knn_vector",
            "dimension": 384,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene"
            }
        }
    }

    # 3. 调用创建索引
    print("开始创建索引...")
    success = await index_manager.create_index(
        model_class=CarInteriorAnalysis,
        field_types=field_types,
        settings=settings,
        overwrite=True  # 如果存在则覆盖，方便测试
    )
    print(f"索引创建结果: {success}")

if __name__ == "__main__":
    asyncio.run(main())
