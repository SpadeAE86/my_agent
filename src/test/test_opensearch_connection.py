import asyncio
import os
import sys

# 将 src 目录添加到路径
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from infra.storage.opensearch_connector import opensearch_connector
from infra.logging.logger import logger as log

async def test_connection():
    log.info("=== OpenSearch 连接测试开始 ===")
    try:
        # 强制设置环境为 local (刚才已经把 local 改成 test 的配置了)
        os.environ["env"] = "local"
        
        log.info("正在初始化连接器...")
        await opensearch_connector.init()
        
        log.info("正在发送 Ping 请求...")
        is_alive = await opensearch_connector.ping()
        
        if is_alive:
            log.info("✅ Ping 成功！OpenSearch 连通性正常。")
            
            # 尝试获取集群信息
            client = opensearch_connector._client
            info = await client.info()
            log.info(f"集群名称: {info.get('cluster_name')}")
            log.info(f"版本信息: {info.get('version', {}).get('number')}")
        else:
            log.error("❌ Ping 失败，连接虽然建立但服务端未响应。")
            
    except Exception as e:
        log.error(f"❌ 连接发生异常: {type(e).__name__}")
        log.error(f"异常详情: {str(e)}")
        
        if "ServerDisconnectedError" in str(e) or "Connection reset" in str(e):
            log.warning("提示: 这种错误通常意味着 SSL/HTTPS 握手失败。请确认配置中的 use_ssl 是否与服务端匹配。")
            
    finally:
        await opensearch_connector.close()
        log.info("=== 测试结束 ===")

if __name__ == "__main__":
    asyncio.run(test_connection())
