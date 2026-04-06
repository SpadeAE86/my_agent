# core/llm/client.py — 多 Provider 统一 LLM 客户端
# 职责:
#   1. 抽象 BaseLLMClient 接口 (chat, stream_chat, embed)
#   2. 具体实现: OpenAIClient, ClaudeClient, GeminiClient...
#   3. 统一处理: API Key 管理、重试策略、速率限制
#   4. 支持 function calling / tool_use 的标准化转换
#   5. Token 使用量追踪与预算检查
from typing import Dict, Any
from cachetools import TTLCache
from openai import AsyncOpenAI

class LlmConnectionManager:
    def __init__(self):
        self.connection_pool = TTLCache(maxsize=100, ttl=3600)  # 每小时过期，最多缓存100个连接

    def get_client(self, api_key, host: str = None) -> Any:
        if api_key in self.connection_pool[api_key]:
            return self.connection_pool[api_key]
        else:
            try:
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=host,
                )
                self.connection_pool[api_key] = client
                return self.connection_pool[host]
            except Exception:
                raise ValueError(f"Fail to connect to host {host} or invalid api key")
