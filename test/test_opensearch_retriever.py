import asyncio
from infra.storage.opensearch_retriever import opensearch_retriever
from infra.storage.opensearch_connector import opensearch_connector


async def test_opensearch_retriever():
    """测试 OpenSearch 混合检索"""
    index_name = "car_interior_analysis"
    
    try:
        # 1. 初始化 OpenSearch 连接器
        await opensearch_connector.init()
        print("OpenSearch 连接器初始化成功")
        
        # 2. 创建索引
        await opensearch_retriever.create_index(index_name)
        print(f"索引 {index_name} 创建成功")
        
        # 3. 准备示例文档
        sample_document = {
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
        
        # 4. 索引文档
        doc_id = "test_doc_1"
        await opensearch_retriever.index_document(index_name, doc_id, sample_document)
        print(f"文档 {doc_id} 索引成功")
        
        # 5. 等待索引刷新
        await asyncio.sleep(2)
        
        # 6. 测试混合检索
        print("\n=== 测试混合检索 ===")
        query = "豪华汽车内饰"
        results = await opensearch_retriever.hybrid_search(index_name, query)
        print(f"混合检索结果 ({len(results)} 条):")
        for i, result in enumerate(results):
            print(f"{i+1}. 得分: {result['score']:.4f}, 主题: {result['subject']}")
        
        # 7. 测试不同权重的混合检索
        print("\n=== 测试不同权重的混合检索 ===")
        weights = {
            'bm25': 0.3,
            'vector': 0.7
        }
        results = await opensearch_retriever.hybrid_search(index_name, query, weights=weights)
        print(f"混合检索结果（向量权重更高） ({len(results)} 条):")
        for i, result in enumerate(results):
            print(f"{i+1}. 得分: {result['score']:.4f}, 主题: {result['subject']}")
        
        # 8. 测试纯语义检索
        print("\n=== 测试纯语义检索 ===")
        results = await opensearch_retriever.semantic_search(index_name, query)
        print(f"纯语义检索结果 ({len(results)} 条):")
        for i, result in enumerate(results):
            print(f"{i+1}. 得分: {result['score']:.4f}, 主题: {result['subject']}")
        
        # 9. 测试纯关键词检索
        print("\n=== 测试纯关键词检索 ===")
        results = await opensearch_retriever.keyword_search(index_name, query)
        print(f"纯关键词检索结果 ({len(results)} 条):")
        for i, result in enumerate(results):
            print(f"{i+1}. 得分: {result['score']:.4f}, 主题: {result['subject']}")
        
        # 10. 测试不同查询
        print("\n=== 测试不同查询 ===")
        queries = ["白色皮质座椅", "座椅调节", "车载天窗", "舒适驾乘"]
        for q in queries:
            results = await opensearch_retriever.hybrid_search(index_name, q)
            print(f"\n查询: '{q}' ({len(results)} 条结果):")
            for i, result in enumerate(results):
                print(f"{i+1}. 得分: {result['score']:.4f}, 主题: {result['subject']}")
        
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        # 11. 关闭连接器
        await opensearch_connector.close()
        print("\nOpenSearch 连接器关闭成功")


if __name__ == "__main__":
    asyncio.run(test_opensearch_retriever())
