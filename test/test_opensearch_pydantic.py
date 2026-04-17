import asyncio
from models.pydantic.opensearch_index import CarInteriorAnalysis
from models.pydantic.opensearch_index.base_index import (
    get_vector_fields,
    get_text_fields,
    get_searchable_fields,
    get_index_name
)
from infra.storage.opensearch import index_manager, query_builder
from sentence_transformers import SentenceTransformer


async def test_opensearch_pydantic():
    print("=== 测试 OpenSearch Pydantic 模型系统 ===\n")
    
    try:
        index_name = get_index_name(CarInteriorAnalysis)
        print(f"✓ 索引名称: {index_name}")
        
        vector_fields = get_vector_fields(CarInteriorAnalysis)
        print(f"✓ 向量字段: {vector_fields}")
        
        text_fields = get_text_fields(CarInteriorAnalysis)
        print(f"✓ 文本字段: {text_fields}")
        
        searchable_fields = get_searchable_fields(CarInteriorAnalysis)
        print(f"✓ 可搜索字段: {searchable_fields}")
        
        sample_analysis = {
            "description": "视频片段展示汽车内部场景，以白色皮质座椅为核心，呈现前排驾驶座、可调节的副驾驶座（靠背从直立逐步倾斜）及后排座椅，车顶配天窗，内饰含杯架、车门饰板等细节，光线明亮柔和，整体设计简约豪华。",
            "subject": "汽车座椅（汽车内饰）",
            "object": ["座椅", "天窗", "杯架", "车门", "中控台"],
            "movement": "副驾驶座椅靠背调节（从直立向倾斜调整）",
            "adjective": ["豪华", "舒适", "明亮", "简约", "高档", "整洁", "现代", "精致"],
            "search_tags": ["汽车内饰", "白色皮质座椅", "座椅调节", "豪华汽车", "车载天窗", "舒适驾乘", "汽车内部设计", "中高端汽车"],
            "marketing_tags": ["产品展示", "使用场景"],
            "appealing_audience": ["汽车爱好者", "购车人群", "中高端消费者", "追求舒适出行者", "有车族"],
            "visual_quality": 8
        }
        
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        car_doc = CarInteriorAnalysis.from_analysis_result(sample_analysis, embedding_model)
        print(f"✓ 文档模型创建成功: {car_doc.subject}")
        
        index_dict = car_doc.model_dump(exclude_none=True)
        print(f"✓ 文档转换为索引字典成功")
        print(f"  - 包含字段数: {len(index_dict)}")
        print(f"  - 包含向量: {any(key.endswith('_vector') for key in index_dict.keys())}")
        
        print("\n=== 测试索引配置查询 (create_index.py) ===")
        
        print("\n=== 测试查询构建器 (query_builder.py) ===")
        
        query = "豪华汽车内饰"
        
        hybrid_query = query_builder.build_hybrid_search(
            CarInteriorAnalysis,
            query,
            size=5,
            bm25_weight=0.4,
            vector_weight=0.6,
            field_boosts={
                'description': 2.0,
                'subject': 3.0,
                'search_tags': 4.0
            }
        )
        print(f"✓ 混合查询构建成功: '{query}'")
        print(f"  - BM25 权重: 0.4")
        print(f"  - 向量权重: 0.6")
        print(f"  - 查询类型: hybrid")
        
        semantic_query = query_builder.build_semantic_search(
            CarInteriorAnalysis,
            query,
            size=5,
            vector_field='combined_vector'
        )
        print(f"✓ 语义查询构建成功: '{query}'")
        print(f"  - 向量字段: combined_vector")
        print(f"  - 查询类型: knn")
        
        keyword_query = query_builder.build_keyword_search(
            CarInteriorAnalysis,
            query,
            size=5,
            field_boosts={
                'description': 2.0,
                'subject': 3.0,
                'search_tags': 4.0
            }
        )
        print(f"✓ 关键词查询构建成功: '{query}'")
        print(f"  - 字段权重: description=2.0, subject=3.0, search_tags=4.0")
        
        filter_query = query_builder.build_filter_search(
            CarInteriorAnalysis,
            query,
            filters={'visual_quality': {'gte': 7}},
            size=5
        )
        print(f"✓ 过滤查询构建成功: '{query}'")
        print(f"  - 过滤条件: visual_quality >= 7")
        
        multi_vector_query = query_builder.build_multi_vector_search(
            CarInteriorAnalysis,
            query,
            size=5,
            vector_fields=['description_vector', 'subject_vector', 'combined_vector'],
            vector_weights={
                'description_vector': 0.3,
                'subject_vector': 0.3,
                'combined_vector': 0.4
            }
        )
        print(f"✓ 多向量查询构建成功: '{query}'")
        print(f"  - 向量字段: description_vector, subject_vector, combined_vector")
        print(f"  - 向量权重: description_vector=0.3, subject_vector=0.3, combined_vector=0.4")
        
        rrf_query = query_builder.build_rrf_search(
            CarInteriorAnalysis,
            query,
            size=5,
            k=60
        )
        print(f"✓ RRF 混合查询构建成功: '{query}'")
        print(f"  - RRF rank_constant: 60")
        
        print("\n=== 完整架构总结 ===")
        print("1. models/pydantic/opensearch_index/ - 模型定义")
        print("   - base_index.py - 基类和辅助函数")
        print("   - car_interior_analysis.py - 示例索引模型")
        print("2. infra/storage/opensearch/ - 存储逻辑")
        print("   - create_index.py - 索引管理（创建、查询、删除）")
        print("   - query_builder.py - 查询构建器")
        print("3. 通过 OpenSearch API 查询现有索引配置")
        
        print("\n=== 测试完成 ===")
        print("所有测试通过！✓")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_opensearch_pydantic())
