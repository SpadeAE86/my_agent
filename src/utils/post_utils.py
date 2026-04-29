import httpx, asyncio
from infra.logging.logger import logger as log
from config.config import MY_CONFIG

# 从配置读取 HTTP 请求参数
http_config = MY_CONFIG.get("http", {})
post_config = http_config.get("post", {})
get_config = http_config.get("get", {})

# 默认值（如果配置不存在则使用这些值）
DEFAULT_POST_RETRY = post_config.get("retry", 4)
DEFAULT_POST_TIMEOUT = post_config.get("timeout", 60.0)
DEFAULT_POST_RETRY_SLEEP = post_config.get("retry_sleep", 10)
DEFAULT_GET_RETRY = get_config.get("retry", 4)
DEFAULT_GET_TIMEOUT = get_config.get("timeout", 10.0)
DEFAULT_GET_RETRY_SLEEP = get_config.get("retry_sleep", 10)

async def post(host, resp_vo, retry=None, task_id="test", headers=None):
    """
    异步POST请求函数，带重试机制
    
    向指定地址发送POST请求，请求体为JSON格式。
    如果请求失败（状态码非200），会根据配置的重试次数和重试间隔自动重试。
    
    Args:
        host: 请求地址，目标服务器的URL
        resp_vo: 请求体，字典类型，会被序列化为JSON发送
        retry: 重试次数，如果为None则使用配置文件中的默认值（DEFAULT_POST_RETRY）
        task_id: 任务ID，用于日志记录和追踪
        headers: 请求头，字典类型，如果为None则使用空字典
    
    Returns:
        Optional[Dict]: 成功时返回响应JSON字典，失败时返回None
    """
    result = None
    if headers is None:
        headers = {}
    if retry is None:
        retry = DEFAULT_POST_RETRY
    
    async with httpx.AsyncClient() as client:
        for i in range(retry):
            msg = await client.post(host, json=resp_vo, headers=headers,
                                    timeout=DEFAULT_POST_TIMEOUT)
            if msg.status_code == 200:
                log.info(f"回调成功, msg: {msg.json()}")
                result = msg.json()
                break
            log.info(f"第{i}次回调失败，{DEFAULT_POST_RETRY_SLEEP}秒后重试")
            await asyncio.sleep(DEFAULT_POST_RETRY_SLEEP)
        log.info(f"{task_id}处理完成")
    return result

async def get(host, params=None, retry=None, task_id="test", headers=None):
    """
    异步GET请求函数，带重试机制
    
    向指定地址发送GET请求，支持URL参数。
    如果请求失败（状态码非200）或发生异常（超时、网络错误等），会根据配置的重试次数和重试间隔自动重试。
    
    Args:
        host: 请求地址，目标服务器的URL
        params: 请求参数，字典类型，会被转换为URL查询参数，如果为None则使用空字典
        retry: 重试次数，如果为None则使用配置文件中的默认值（DEFAULT_GET_RETRY）
        task_id: 任务ID，用于日志记录和追踪
        headers: 请求头，字典类型，如果为None则使用空字典
    
    Returns:
        Optional[Dict]: 成功时返回响应JSON字典，失败时返回None
    """
    if headers is None:
        headers = {}
    if params is None:
        params = {}
    if retry is None:
        retry = DEFAULT_GET_RETRY

    async with httpx.AsyncClient() as client:
        for i in range(retry):
            try:
                msg = await client.get(host, params=params, headers=headers, timeout=DEFAULT_GET_TIMEOUT)
                if msg.status_code == 200:
                    log.info(f"GET请求成功, msg: {msg.json()}")
                    return msg.json()  # 返回响应数据
                log.info(f"第{i + 1}次GET请求失败，状态码: {msg.status_code}, {DEFAULT_GET_RETRY_SLEEP}秒后重试")
            except httpx.TimeoutException:
                log.info(f"第{i + 1}次GET请求超时，{DEFAULT_GET_RETRY_SLEEP}秒后重试")
            except httpx.RequestError as e:
                log.info(f"第{i + 1}次GET请求错误: {str(e)}，{DEFAULT_GET_RETRY_SLEEP}秒后重试")
            except Exception as e:
                log.info(f"第{i + 1}次GET请求发生异常: {str(e)}，{DEFAULT_GET_RETRY_SLEEP}秒后重试")

            await asyncio.sleep(DEFAULT_GET_RETRY_SLEEP)
        log.info(f"{task_id} GET请求处理完成")
        return None  # 所有重试都失败后返回None