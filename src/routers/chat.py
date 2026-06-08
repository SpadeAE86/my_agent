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
import uuid
import os
import logging
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, FileResponse
from openai import AsyncOpenAI

from models.pydantic.request import ChatRequest
from core.agent.agent import Agent
from core.agent.agent_loop import main_loop
from core.agent.event import TaskComplete, ErrorEvent
from core.tools.tool_manager import ToolManager
from infra.logging.logger import logger as log, log_agent_debug

# Set up a separate chat-related logger
chat_logger = logging.getLogger("agent_chat")

chat_router = APIRouter(prefix="/chat", tags=["chat"])

# ─── LLM 客户端 (后续可移到 config 层) ────────────────────────────
_llm_client = AsyncOpenAI(
    base_url="https://ai.comfly.chat/v1",
    api_key="sk-EZyThGS2JdkoxISCD7Dd64D625E94a8b9513D71aCfF6AcFc",
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
    # 确定会话 ID
    session_id = req.session_id or uuid.uuid4().hex[:8]
    chat_logger.info(f"\n==================== Start Chat Session: {session_id} ====================")
    chat_logger.info(f"User Input: {req.message}")

    # Log session start to structured debug log
    log_agent_debug(session_id, "session_start", {"message": req.message, "model": req.model, "user_id": req.user_id})

    async def event_generator():
        try:
            # 1. 优先推送 session_id，让前端捕获并保持会话
            yield f"data: {json.dumps({'event_type': 'session_id', 'session_id': session_id}, ensure_ascii=False)}\n\n"

            # 2. 创建 Agent
            agent = Agent(
                user_id=req.user_id,
                llm={"client": _llm_client, "model": req.model},
                tools=_tool_manager.list_names(),
                skills={},
                session_id=session_id,
                max_iteration=req.max_iterations,
                mode="swarm",
                is_base=True,
                max_token=1024,
                tool_manager=_tool_manager,
                language="中文",
            )

            # 3. 驱动 main_loop, 把每个 AgentEvent 序列化为 SSE data 行
            async for event in main_loop(
                agent,
                req.message,
                tool_manager=_tool_manager,
                reference_image_list=req.reference_image_list
            ):
                # Pydantic model → dict → JSON string
                payload = event.model_dump()
                evt_type = payload.get('event_type', '?')
                
                # 记录到独立的 agent_chat 日志文件
                if evt_type == 'status_update':
                    chat_logger.info(f"[Status] {payload.get('message')}")
                elif evt_type == 'agent_thought':
                    chat_logger.info(f"[Thought] {payload.get('content')}")
                    log_agent_debug(session_id, "agent_thought", payload)
                elif evt_type == 'tool_call':
                    chat_logger.info(f"[Tool Call] {payload.get('tool_name')}({payload.get('arguments')})")
                elif evt_type == 'tool_result':
                    status = "Success" if payload.get('success') else "Failed"
                    chat_logger.info(f"[Tool Result] {status}: {payload.get('output') or payload.get('error')}")
                elif evt_type == 'text_chunk':
                    chat_logger.info(f"[Response Chunk] {payload.get('content')}")
                elif evt_type == 'task_complete':
                    chat_logger.info(f"[Task Complete] {payload.get('summary')}")
                    log_agent_debug(session_id, "session_complete", payload)

                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                # 精简并增强系统主日志的输出，避免 text_chunk 的过度刷屏，且为关键事件提供充足上下文
                if evt_type == 'tool_call':
                    log.info(f"SSE → tool_call: {payload.get('tool_name')}({json.dumps(payload.get('arguments', {}), ensure_ascii=False)})")
                elif evt_type == 'tool_result':
                    status = "SUCCESS" if payload.get('success') else "FAILED"
                    out_err = payload.get('output') or payload.get('error') or ''
                    preview = out_err[:200] + "..." if len(out_err) > 200 else out_err
                    log.info(f"SSE → tool_result [{status}]: {payload.get('tool_name')} | {preview}")
                elif evt_type == 'status_update':
                    log.info(f"SSE → status_update: {payload.get('message')}")
                elif evt_type == 'agent_thought':
                    thought = payload.get('content') or ''
                    preview = thought[:100].replace('\n', ' ') + "..." if len(thought) > 100 else thought.replace('\n', ' ')
                    log.info(f"SSE → agent_thought: {preview}")
                elif evt_type == 'task_complete':
                    log.info(f"SSE → task_complete: {payload.get('summary')}")
                elif evt_type == 'error':
                    log.error(f"SSE → error: {payload.get('message') or payload.get('error', '?')}")
                elif evt_type != 'text_chunk':
                    log.info(f"SSE → {evt_type}")

                yield line

            # 保存对话历史
            is_first_turn = sum(1 for m in agent.messages if m.get("role") == "user") == 1
            if is_first_turn:
                abstract = req.message[:15]
                try:
                    summary_prompt = [
                        {"role": "system", "content": "你是一个会话标题生成器。请根据用户的第一个提问，生成一个极其简短、概括性强、没有标点符号的中文会话标题（不超过10个字）。直接输出标题，不要有任何前缀、解释或双引号。"},
                        {"role": "user", "content": f"用户第一个提问：{req.message}\n请生成对应标题。"}
                    ]
                    sum_resp = await _llm_client.chat.completions.create(
                        model=req.model,
                        messages=summary_prompt,
                        max_tokens=20,
                        temperature=0.3
                    )
                    content_str = sum_resp.choices[0].message.content.strip()
                    content_str = content_str.replace('"', '').replace('“', '').replace('”', '').replace("'", "").replace("'", "")
                    if content_str and len(content_str) <= 15:
                        abstract = content_str
                    elif content_str:
                        abstract = content_str[:15]
                except Exception as e:
                    chat_logger.error(f"Failed to generate session abstract: {e}")
                agent.save_history(abstract=abstract)
            else:
                agent.save_history()
            chat_logger.info(f"==================== End Chat Session: {session_id} ====================\n")

            # 4. 发送结束标记 [DONE] (OpenAI 的事实标准)
            yield "data: [DONE]\n\n"

        except Exception as e:
            chat_logger.error(f"Error in chat session {session_id}: {e}")
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


@chat_router.get("/sessions")
async def get_chat_sessions(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1)):
    """
    获取会话历史列表 (分页)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    memory_dir = project_root / "data" / "sessions"
    
    if not memory_dir.exists():
        return {"sessions": [], "has_more": False}
        
    # 获取所有的 .jsonl 文件及其修改时间
    files_with_mtime = []
    for item in memory_dir.iterdir():
        if item.is_file() and item.suffix == ".jsonl":
            files_with_mtime.append((item, item.stat().st_mtime))
            
    # 按最后修改时间倒序排列 (最新在最前)
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    total = len(files_with_mtime)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_files = files_with_mtime[start_idx:end_idx]
    
    sessions = []
    for item, mtime in page_files:
        session_id = item.stem
        title = "空会话"
        try:
            with open(item, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        meta = json.loads(first_line)
                        if "role" not in meta and "abstract" in meta:
                            title = meta["abstract"]
                        elif meta.get("role") == "user" and meta.get("content"):
                            title = meta["content"]
                    except json.JSONDecodeError:
                        pass
                
                if title == "空会话":
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            if msg.get("role") == "user" and msg.get("content"):
                                title = msg["content"]
                                break
                        except json.JSONDecodeError:
                            continue
            if len(title) > 30:
                title = title[:30] + "..."
        except Exception:
            pass
            
        sessions.append({
            "session_id": session_id,
            "title": title,
            "updated_at": mtime
        })
        
    has_more = end_idx < total
    return {"sessions": sessions, "has_more": has_more}


@chat_router.get("/sessions/{session_id}")
async def get_chat_history_events(session_id: str):
    """
    获取单个会话的完整历史事件流 (还原为前端 ChatEvent 格式)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    memory_file = project_root / "data" / "sessions" / f"{session_id}.jsonl"
    
    events = []
    if not memory_file.exists():
        return events
        
    messages = []
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return events
        
    base_time = int(time.time() * 1000) - len(messages) * 1000
    
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content") or ""
        ts = base_time + idx * 1000
        event_id = f"evt_hist_{session_id}_{idx}"
        
        if role == "user":
            if msg.get("is_summary"):
                # Clean up summary content wrapper "[Conversation Summary of previous turns:\n...]"
                clean_content = content
                if clean_content.startswith("[Conversation Summary of previous turns:\n"):
                    clean_content = clean_content[len("[Conversation Summary of previous turns:\n"):]
                if clean_content.endswith("]"):
                    clean_content = clean_content[:-1]
                
                events.append({
                    "id": event_id,
                    "type": "compressed",
                    "content": clean_content.strip(),
                    "timestamp": ts
                })
                continue

            text_content = ""
            ref_imgs = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_content = part.get("text", "")
                        elif part.get("type") == "image_url":
                            img_obj = part.get("image_url", {})
                            if isinstance(img_obj, dict) and img_obj.get("url"):
                                ref_imgs.append(img_obj["url"])
            else:
                text_content = content

            events.append({
                "id": event_id,
                "type": "user",
                "content": text_content,
                "timestamp": ts,
                "reference_image_list": ref_imgs
            })
        elif role == "assistant":
            # 如果包含历史思考内容，恢复为 thinking 事件，并赋予稍早的时间戳以保证前端渲染顺序
            reasoning = msg.get("reasoning_content")
            if reasoning and reasoning.strip():
                events.append({
                    "id": f"{event_id}_reasoning",
                    "type": "thinking",
                    "content": reasoning,
                    "timestamp": ts - 500,
                    "streaming": False
                })
                
            if content.strip():
                events.append({
                    "id": event_id,
                    "type": "assistant",
                    "content": content,
                    "timestamp": ts,
                    "streaming": False
                })
            
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for t_idx, tc in enumerate(tool_calls):
                    func = tc.get("function", {})
                    args_data = func.get("arguments", {})
                    if isinstance(args_data, str):
                        try:
                            args_data = json.loads(args_data)
                        except Exception:
                            pass
                    events.append({
                        "id": f"{event_id}_tc_{t_idx}",
                        "type": "tool_call",
                        "content": "",
                        "timestamp": ts,
                        "toolName": func.get("name", "unknown"),
                        "toolArgs": args_data
                    })
        elif role == "tool":
            tool_name = "unknown"
            tool_call_id = msg.get("tool_call_id")
            for prev_msg in reversed(messages[:idx]):
                if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                    for tc in prev_msg["tool_calls"]:
                        if tc.get("id") == tool_call_id:
                            tool_name = tc.get("function", {}).get("name", "unknown")
                            break
            
            success = not content.startswith("Error:")
            events.append({
                "id": event_id,
                "type": "tool_result",
                "content": content,
                "timestamp": ts,
                "toolName": tool_name,
                "toolSuccess": success
            })
            
    return events


@chat_router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """
    删除会话历史文件
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    memory_file = project_root / "data" / "sessions" / f"{session_id}.jsonl"
    if memory_file.exists():
        try:
            memory_file.unlink()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Session not found"}


@chat_router.get("/sessions/{session_id}/export")
async def export_chat_session(session_id: str):
    """
    导出并下载会话历史的 JSONL 文件
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    memory_file = project_root / "data" / "sessions" / f"{session_id}.jsonl"
    if memory_file.exists():
        return FileResponse(
            path=memory_file,
            filename=f"{session_id}.jsonl",
            media_type="application/json"
        )
    return {"error": "Session not found"}


@chat_router.get("/models")
async def get_supported_models():
    """
    获取支持的多模态模型列表
    """
    return [
        {"display_name": "Doubao Seed 2.0 Pro", "real_name": "doubao-seed-2-0-pro-260215"},
        {"display_name": "GPT 5.4 (默认)", "real_name": "gpt-5.4"},
        {"display_name": "Gemini 3.5 Flash", "real_name": "gemini-3.5-flash"},
        {"display_name": "Gemini 3.1 Pro", "real_name": "gemini-3.1-pro-preview"},
        {"display_name": "Qwen 3.5 Plus", "real_name": "qwen3.5-plus"},
        {"display_name": "Claude Sonnet 4.6", "real_name": "claude-sonnet-4-6"}
    ]
