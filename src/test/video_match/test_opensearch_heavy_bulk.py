import asyncio
import os
import sys
import numpy as np

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.document_writer import bulk_index
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from infra.logging.logger import logger as log

async def test_heavy_bulk_indexing():
    log.info("=== 开始大规模入库压力测试 ===")
    os.environ["env"] = "local"
    await opensearch_connector.init()
    
    # 模拟一个 50 分镜的视频，每个分镜带 7 路向量
    # 这将产生一个巨大的 JSON 载荷发给 OpenSearch
    video_id = "test_heavy_video_20260120"
    doc_count = 50
    
    docs = []
    for i in range(1, doc_count + 1):
        # 模拟生成 384 维向量
        dummy_embedding = np.random.rand(384).tolist()
        
        doc = CarInteriorAnalysisV2(
            id=f"{video_id}_scene_{i}",
            description=f"这是大规模测试的第 {i} 个分镜描述，模拟复杂的文本内容和向量载荷。",
            subject="测试主体",
            movement="测试动作",
            search_tags=["测试", "压力测试", "大批量"],
            # 模拟 7 路向量
            description_vector=dummy_embedding,
            subject_vector=dummy_embedding,
            object_vector=dummy_embedding,
            movement_vector=dummy_embedding,
            adjective_vector=dummy_embedding,
            search_tags_vector=dummy_embedding,
            marketing_tags_vector=dummy_embedding,
            visual_quality=[8.5] * 4
        )
        docs.append(doc)
    
    log.info(f"载荷构建完成，共 {len(docs)} 条文档，准备执行 bulk_index...")
    
    try:
        # 执行入库
        import time
        start_time = time.time()
        resp = await bulk_index(CarInteriorAnalysisV2, docs, refresh=True)
        end_time = time.time()
        
        log.info(f"入库成功！耗时: {end_time - start_time:.2f} 秒")
        log.info(f"OpenSearch 响应摘要: {str(resp)[:200]}...")
        
    except Exception as e:
        log.error(f"❌ 入库失败！可能是因为 Payload 过大或连接超时: {e}")
        if "ServerDisconnectedError" in str(e):
            log.warning("警告：检测到 ServerDisconnectedError，说明网关无法处理单次大数据量请求。")
            log.info("建议：需要修改 document_writer.py 增加分批（Chunking）逻辑。")
            
    finally:
        await opensearch_connector.close()

if __name__ == "__main__":
    asyncio.run(test_heavy_bulk_indexing())
