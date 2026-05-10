import asyncio
import os
import sys
import uuid

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.document_writer import bulk_index
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from infra.logging.logger import logger as log

async def test_idempotent_indexing():
    log.info("=== 开始模拟去重入库测试 ===")
    os.environ["env"] = "local"
    await opensearch_connector.init()
    
    stable_id = "test_video_md5_unique_123"
    scene_id = 1
    doc_id = f"{stable_id}_scene_{scene_id}"
    
    try:
        # 1. 模拟第一次入库
        log.info(f"第一次入库: ID={doc_id}")
        doc1 = CarInteriorAnalysisV2(
            id=doc_id,
            description="这是第一次分析的结果",
            subject="汽车方向盘",
            movement="静止",
            search_tags=["测试", "第一次"],
            visual_quality=[8, 8, 8, 8]
        )
        await bulk_index(CarInteriorAnalysisV2, [doc1], refresh=True)
        
        # 2. 模拟第二次入库 (内容改变，但 ID 不变)
        log.info(f"第二次入库 (覆盖): ID={doc_id}")
        doc2 = CarInteriorAnalysisV2(
            id=doc_id,
            description="这是第二次覆盖后的结果",
            subject="汽车方向盘",
            movement="旋转",
            search_tags=["测试", "第二次覆盖"],
            visual_quality=[9, 9, 9, 9]
        )
        await bulk_index(CarInteriorAnalysisV2, [doc2], refresh=True)
        
        # 3. 验证结果
        log.info("正在查询 OpenSearch 验证结果...")
        client = opensearch_connector._client
        resp = await client.get(index="car_interior_analysis_v2", id=doc_id)
        
        content = resp["_source"].get("description")
        log.info(f"查询到的内容: {content}")
        
        if content == "这是第二次覆盖后的结果":
            log.info("✅ 测试成功：文档已正确覆盖，未产生重复记录。")
        else:
            log.error("❌ 测试失败：文档内容不匹配。")
            
    except Exception as e:
        log.error(f"测试发生异常: {e}")
    finally:
        await opensearch_connector.close()

if __name__ == "__main__":
    asyncio.run(test_idempotent_indexing())
