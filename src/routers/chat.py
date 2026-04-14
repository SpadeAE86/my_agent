# routers/chat.py — 基础对话路由
# 端点:
#   POST /chat          — 发送消息, SSE 流式返回 Agent 事件
#   GET  /chat/history  — 获取当前会话的对话历史
#
# ═══════════════════════════════════════════════════════════════
# SSE 协议要点 (和普通 JSON 接口的核心区别):
#
# 1. 普通接口: 一问一答, 等全部处理完才返回一整个 JSON
#    SSE:      连接建立后, 服务端持续推送多条消息, 最后关闭连接
#
# 2. SSE 每条消息的格式 (纯文本, 不是 JSON body):
#    data: {"event_type": "text_chunk", "content": "你好"}\n\n
#    data: {"event_type": "tool_call", "tool_name": "grep", ...}\n\n
#    data: [DONE]\n\n    ← finish 标记!
#
# 3. 关键区别:
#    - Content-Type 是 text/event-stream, 不是 application/json
#    - 每条数据前缀 "data: ", 以 \n\n 结尾
#    - 浏览器用 EventSource API 或 fetch + ReadableStream 消费
#    - finish 标记: 发送 data: [DONE]\n\n (OpenAI 的事实标准)
#
# 4. 为什么用 SSE 而不是 WebSocket:
#    - SSE 是单向的 (服务端 → 客户端), 刚好适合 Agent 事件流
#    - 天然支持断线重连 (浏览器自动)
#    - 不需要额外协议握手, 比 WebSocket 简单得多
# ═══════════════════════════════════════════════════════════════

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from models.pydantic.request import ChatRequest
from core.agent.agent import Agent
from core.agent.agent_loop import main_loop
from core.agent.event import TaskComplete, ErrorEvent
from core.tools.tool_manager import ToolManager
from infra.logging.logger import logger as log

chat_router = APIRouter(prefix="/chat", tags=["chat"])

# ─── LLM 客户端 (后续可移到 config 层) ────────────────────────────
_llm_client = AsyncOpenAI(
    base_url="https://z.apiyihe.org/v1",
    api_key="sk-TMd7SbPPbVw1JMx0GYKflkWkv8Mzi1tb0B64Y9HqBQ53TaqW",
)

# ToolManager 单例 (启动时自动发现工具)
_tool_manager = ToolManager()
_tool_manager.auto_discover()


@chat_router.post("")
async def chat_sse(req: ChatRequest):
    """
    SSE 流式对话端点。

    前端调用方式:
        fetch('/chat', { method: 'POST', body: JSON.stringify({message: '...'}), ... })
        然后用 ReadableStream 逐行读取 "data: {...}" 事件。

    每条 SSE 消息格式:
        data: {"event_type": "status_update", "status": "thinking", ...}

    结束标记:
        data: [DONE]
    """

    async def event_generator():
        try:
            # 1. 创建 Agent
            agent = Agent(
                user_id=req.user_id,
                llm={"client": _llm_client, "model": req.model},
                tools=_tool_manager.list_names(),
                skills={},
                session_id=req.session_id,
                max_iteration=req.max_iterations,
                mode="swarm",
                is_base=True,
                max_token=4096,
                tool_manager=_tool_manager,
                language="中文",
            )

            # 2. 驱动 main_loop, 把每个 AgentEvent 序列化为 SSE data 行
            async for event in main_loop(agent, req.message, tool_manager=_tool_manager):
                # Pydantic model → dict → JSON string
                payload = event.model_dump()
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                # 日志: 工具失败时打印 error 内容
                evt_type = payload.get('event_type', '?')
                if evt_type == 'tool_result' and not payload.get('success'):
                    log.error(f"SSE → {evt_type} FAILED: {payload.get('error', '?')}")
                else:
                    log.info(f"SSE → {evt_type}")

                yield line

            # 3. 发送结束标记 [DONE] (OpenAI 的事实标准)
            yield "data: [DONE]\n\n"

        except Exception as e:
            # 异常也通过 SSE 推给前端, 而不是返回 HTTP 500
            error_payload = {
                "event_type": "error",
                "error_code": "AGENT_ERROR",
                "message": f"{type(e).__name__}: {e}",
                "recoverable": False,
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止 Nginx 缓冲 SSE
        },
    )
