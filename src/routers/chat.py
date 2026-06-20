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
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Query, UploadFile, File, Response
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from openai import AsyncOpenAI

from models.pydantic.request import ChatRequest, CreateRoleRequest
from core.agent.agent import Agent
from core.agent.agent_loop import main_loop
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

    # 读取角色信息以配置流式语音
    from core.roles.role_manager import role_manager
    role_meta = role_manager.get_role(req.role_id) if req.role_id else None
    voice_enabled = role_meta.get("voice_configured", False) if role_meta else False
    voice_character = (role_meta.get("voice_character") or "Vivi") if role_meta else "Vivi"

    async def event_generator():
        import asyncio
        import re
        import json

        sse_queue = asyncio.Queue()

        def push_event(payload):
            sse_queue.put_nowait(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

        try:
            # 1. 优先推送 session_id，让前端捕获并保持会话
            push_event({'event_type': 'session_id', 'session_id': session_id})

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
                active_workspace_id=req.active_workspace_id,
                active_graph_name=req.active_graph_name,
                role_id=req.role_id,
            )

            current_sentence = ""
            sentence_index = 0
            tts_tasks = []

            async def generate_voice_chunk(text_to_speak, idx):
                try:
                    import hashlib
                    from diskcache import Cache
                    from services.volcovoice_service import VolcoVoiceService
                    from config.config import MY_CONFIG, ENV
                    
                    segment_text = text_to_speak.strip()
                    if not segment_text:
                        return
                    
                    cache_dir = os.path.join(MY_CONFIG['cache_config'][ENV]['cache_dir'], 'tts_metadata')
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    cache_key = hashlib.sha256(f"{segment_text}_{voice_character}_1.0".encode("utf-8")).hexdigest()
                    
                    cached_url = None
                    with Cache(cache_dir) as tts_cache:
                        cached_url = tts_cache.get(cache_key)
                        
                    if cached_url:
                        log.info(f"SSE TTS Segment 缓存命中: {cache_key} -> {cached_url}")
                        audio_url = cached_url
                    else:
                        audio_url = await VolcoVoiceService.generate_voice(
                            text=segment_text,
                            voice_character=voice_character,
                            speed=1.0,
                            disable_segmentation=True
                        )
                        with Cache(cache_dir) as tts_cache:
                            tts_cache.set(cache_key, audio_url, expire=86400)
                        log.info(f"SSE TTS Segment 写入缓存: {cache_key} -> {audio_url}")
                        
                    push_event({
                        "event_type": "voice_chunk",
                        "url": audio_url,
                        "text": segment_text,
                        "index": idx
                    })
                except Exception as e:
                    log.error(f"Failed to generate voice chunk in SSE: {e}")

            async def run_agent_loop():
                nonlocal current_sentence, sentence_index
                try:
                    async for event in main_loop(
                        agent,
                        req.message,
                        tool_manager=_tool_manager,
                        reference_image_list=req.reference_image_list
                    ):
                        payload = event.model_dump()
                        evt_type = payload.get('event_type', '?')

                        # 记录日志
                        if evt_type == 'status_update':
                            chat_logger.info(f"[Status] {payload.get('message')}")
                            log.info(f"SSE → status_update: {payload.get('message')}")
                        elif evt_type == 'agent_thought':
                            chat_logger.info(f"[Thought] {payload.get('content')}")
                            log_agent_debug(session_id, "agent_thought", payload)
                            thought = payload.get('content') or ''
                            preview = thought[:100].replace('\n', ' ') + "..." if len(thought) > 100 else thought.replace('\n', ' ')
                            log.info(f"SSE → agent_thought: {preview}")
                        elif evt_type == 'tool_call':
                            chat_logger.info(f"[Tool Call] {payload.get('tool_name')}({payload.get('arguments')})")
                            log.info(f"SSE → tool_call: {payload.get('tool_name')}({json.dumps(payload.get('arguments', {}), ensure_ascii=False)})")
                        elif evt_type == 'tool_result':
                            status = "Success" if payload.get('success') else "Failed"
                            chat_logger.info(f"[Tool Result] {status}: {payload.get('output') or payload.get('error')}")
                            status_upper = "SUCCESS" if payload.get('success') else "FAILED"
                            out_err = payload.get('output') or payload.get('error') or ''
                            preview = out_err[:200] + "..." if len(out_err) > 200 else out_err
                            log.info(f"SSE → tool_result [{status_upper}]: {payload.get('tool_name')} | {preview}")
                        elif evt_type == 'text_chunk':
                            content = payload.get('content') or ''
                            chat_logger.info(f"[Response Chunk] {content}")
                            if voice_enabled:
                                current_sentence += content
                                parts = re.split(r'([。！？!?；;\n]+)', current_sentence)
                                if len(parts) > 1:
                                    for idx in range(0, len(parts) - 1, 2):
                                        sentence = parts[idx] + parts[idx+1]
                                        if sentence.strip():
                                            task = asyncio.create_task(generate_voice_chunk(sentence, sentence_index))
                                            tts_tasks.append(task)
                                            sentence_index += 1
                                    current_sentence = parts[-1]
                        elif evt_type == 'task_complete':
                            chat_logger.info(f"[Task Complete] {payload.get('summary')}")
                            log_agent_debug(session_id, "session_complete", payload)
                            log.info(f"SSE → task_complete: {payload.get('summary')}")
                        elif evt_type == 'error':
                            log.error(f"SSE → error: {payload.get('message') or payload.get('error', '?')}")
                        else:
                            log.info(f"SSE → {evt_type}")

                        push_event(payload)

                    # 提取最后一句
                    if voice_enabled and current_sentence.strip():
                        task = asyncio.create_task(generate_voice_chunk(current_sentence, sentence_index))
                        tts_tasks.append(task)
                        sentence_index += 1
                        current_sentence = ""

                    # 等待所有语音生成任务完成
                    if tts_tasks:
                        await asyncio.gather(*tts_tasks, return_exceptions=True)

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
                            content_str = content_str.replace('"', '').replace('“', '').replace('”', '').replace("'", "").replace('’', '')
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

                    # 4. 发送结束标记 [DONE]
                    sse_queue.put_nowait("data: [DONE]\n\n")

                except Exception as e:
                    log.error(f"Error in run_agent_loop: {e}")
                    push_event({"event_type": "error", "message": str(e)})
                finally:
                    await sse_queue.put(None)

            loop_task = asyncio.create_task(run_agent_loop())

            try:
                while True:
                    line = await sse_queue.get()
                    if line is None:
                        break
                    yield line
            finally:
                await loop_task
        except Exception as e:
            import traceback
            chat_logger.error(f"Error in chat session {session_id}: {e}\n{traceback.format_exc()}")
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
async def get_chat_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1),
    role_id: Optional[str] = Query(None)
):
    """
    获取会话历史列表 (分页)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    memory_dir = project_root / "data" / "sessions"
    
    from services.media_generate_services.workspace_db_service import workspace_db_service
    try:
        workspaces = await workspace_db_service.list_workspaces()
        workspace_ids = {ws.id for ws in workspaces}
    except Exception:
        workspace_ids = set()
    
    if not memory_dir.exists():
        return {"sessions": [], "has_more": False}
        
    # 获取所有的 .jsonl 文件及其修改时间
    files_with_mtime = []
    for item in memory_dir.iterdir():
        if item.is_file() and item.suffix == ".jsonl":
            files_with_mtime.append((item, item.stat().st_mtime))
            
    # 按最后修改时间倒序排列 (最新在最前)
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    # 根据 role_id 过滤
    filtered_files = []
    for item, mtime in files_with_mtime:
        if role_id:
            file_role_id = "default"
            try:
                with open(item, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        meta = json.loads(first_line)
                        if "role" not in meta:
                            file_role_id = meta.get("role_id", "default")
            except Exception:
                pass
            if file_role_id != role_id:
                continue
        filtered_files.append((item, mtime))
        
    total = len(filtered_files)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_files = filtered_files[start_idx:end_idx]
    
    sessions = []
    for item, mtime in page_files:
        session_id = item.stem
        title = "空会话"
        active_workspace_ids = set()
        active_graph_names = set()
        role_id = "default"
        try:
            with open(item, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        meta = json.loads(first_line)
                        if "role" not in meta:
                            role_id = meta.get("role_id", "default")
                            if "abstract" in meta:
                                title = meta["abstract"]
                            w_ids = meta.get("active_workspace_ids")
                            if w_ids:
                                active_workspace_ids.update(w_ids)
                            elif meta.get("active_workspace_id"):
                                active_workspace_ids.add(meta["active_workspace_id"])
                                
                            g_names = meta.get("active_graph_names")
                            if g_names:
                                active_graph_names.update(g_names)
                            elif meta.get("active_graph_name"):
                                active_graph_names.add(meta["active_graph_name"])
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
                            
                # Scan all lines to find active_workspace_ids and active_graph_names
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if msg.get("role") == "assistant" and "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                func = tc.get("function", {})
                                t_name = func.get("name")
                                args_str = func.get("arguments", "{}")
                                args = {}
                                if isinstance(args_str, str):
                                    try:
                                        args = json.loads(args_str)
                                    except Exception:
                                        pass
                                elif isinstance(args_str, dict):
                                    args = args_str
                                    
                                if t_name in ["read_graph", "make_graph", "update_graph"]:
                                    g_name = args.get("file_name")
                                    if g_name:
                                        active_graph_names.add(g_name)
                                elif t_name in ["get_canvas_graph", "create_canvas_node", "update_canvas_node", "link_canvas_nodes"]:
                                    w_id = args.get("workspace_id")
                                    if w_id:
                                        active_workspace_ids.add(w_id)
                    except Exception:
                        continue
            if len(title) > 30:
                title = title[:30] + "..."
        except Exception:
            pass
            
        # Context matching and fallbacks
        if session_id in workspace_ids:
            active_workspace_ids.add(session_id)
        if (project_root / "data" / "graphs" / f"{session_id}.json").exists():
            active_graph_names.add(session_id)
            
        if not active_workspace_ids and not active_graph_names:
            active_workspace_ids.add(session_id)

        sessions.append({
            "session_id": session_id,
            "title": title,
            "updated_at": mtime,
            "role_id": role_id,
            "active_workspace_id": list(active_workspace_ids)[0] if active_workspace_ids else None,
            "active_graph_name": list(active_graph_names)[0] if active_graph_names else None,
            "active_workspace_ids": list(active_workspace_ids),
            "active_graph_names": list(active_graph_names)
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


@chat_router.get("/roles")
async def get_roles(response: Response):
    """
    获取所有可用角色列表。
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    from core.roles.role_manager import role_manager
    roles = role_manager.list_roles()
    if not roles:
        # 保底: 返回默认 CC 角色
        roles = [{"id": "default", "mode": "default", "name": "CC", "description": "通用助手", "avatar_emoji": "⚡"}]
    
    # 动态构建带有修改时间戳的 avatar_url 和 portrait_url
    for r in roles:
        role_id = r.get("id")
        if not role_id or role_id == "default":
            continue
        workspace = role_manager.get_role_workspace(role_id)
        
        # 头像
        avatar_url = None
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                mtime = int(avatar_path.stat().st_mtime)
                avatar_url = f"/api/chat/roles/{role_id}/avatar?t={mtime}"
                break
        if avatar_url:
            r["avatar_url"] = avatar_url
            if "avatar_emoji" in r:
                del r["avatar_emoji"]
                
        # 立绘
        portrait_url = None
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            portrait_path = workspace / name
            if portrait_path.exists():
                mtime = int(portrait_path.stat().st_mtime)
                portrait_url = f"/api/chat/roles/{role_id}/portrait?t={mtime}"
                break
        if portrait_url:
            r["portrait_url"] = portrait_url
            
    return roles


@chat_router.post("/roles")
async def create_new_role(req: CreateRoleRequest):
    """
    新建一个自定义角色。
    """
    try:
        from core.roles.role_manager import role_manager
        meta = role_manager.create_role(req.name)
        return {"ok": True, "role": meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@chat_router.get("/roles/{role_id}/settings")
async def get_role_settings(role_id: str):
    """
    获取角色的自定义用户人设 (USER_SETTINGS.md)
    """
    try:
        from core.roles.role_manager import role_manager
        user_settings = role_manager.read_user_settings(role_id)
        return {"ok": True, "user_settings": user_settings}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@chat_router.post("/roles/{role_id}/settings")
async def save_role_settings(role_id: str, req: dict):
    """
    保存角色的自定义用户人设 (USER_SETTINGS.md)
    """
    try:
        from core.roles.role_manager import role_manager
        user_settings = req.get("user_settings", "")
        role_manager.save_user_settings(role_id, user_settings)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@chat_router.post("/roles/{role_id}/avatar")
async def upload_role_avatar(role_id: str, file: UploadFile = File(...)):
    """
    上传角色的头像图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return {"ok": False, "error": "角色工作区不存在"}
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return {"ok": False, "error": "不支持的图片格式"}
            
        # 清理已存在的头像
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                avatar_path.unlink()
                
        # 保存新头像
        target_path = workspace / f"avatar{ext}"
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            try:
                with open(role_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["avatar_url"] = f"/api/chat/roles/{role_id}/avatar"
                if "avatar_emoji" in meta:
                    del meta["avatar_emoji"]
                with open(role_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
                
        return {"ok": True, "avatar_url": f"/api/chat/roles/{role_id}/avatar?t={int(time.time())}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@chat_router.get("/roles/{role_id}/avatar")
async def get_role_avatar(role_id: str):
    """
    获取角色的头像图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                return FileResponse(path=avatar_path)
        # 默认返回一个 404 或小占位图
        return {"error": "Avatar not found"}
    except Exception as e:
        return {"error": str(e)}


@chat_router.get("/roles/{role_id}/files/{filename}")
async def get_role_file(role_id: str, filename: str):
    """
    获取角色工作区内的特定 Markdown 文件内容 (IDENTITY.md / SOUL.md 等)
    """
    try:
        from core.roles.role_manager import role_manager
        # 限制只能读取特定安全的文件，防止目录遍历漏洞
        if filename not in ["IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md", "USER_SETTINGS.md"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持读取该文件"})
            
        content = role_manager.read_md_file(role_id, filename)
        return Response(content=content, media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.post("/roles/{role_id}/files/{filename}")
async def save_role_file(role_id: str, filename: str, req: dict):
    """
    保存/覆盖角色工作区内的特定 Markdown 文件内容
    """
    try:
        from core.roles.role_manager import role_manager
        if filename not in ["IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md", "USER_SETTINGS.md"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持修改该文件"})
            
        content = req.get("content", "")
        role_manager.save_md_file(role_id, filename, content)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.get("/roles/{role_id}/gallery")
async def get_role_gallery(role_id: str):
    """
    获取角色画廊中的图片列表，以及当前使用的主形象和头像文件名
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        meta = role_manager.get_role(role_id) or {}
        active_portrait = meta.get("portrait_filename")
        active_avatar = meta.get("avatar_filename")
        
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载导入来源元数据
        meta_json_path = gallery_dir / "meta.json"
        meta_data = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}
        
        images = []
        for item in sorted(gallery_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                filename = item.name
                images.append({
                    "filename": filename,
                    "url": f"/api/chat/roles/{role_id}/gallery/{filename}?t={int(item.stat().st_mtime)}",
                    "is_portrait": filename == active_portrait,
                    "is_avatar": filename == active_avatar,
                    "source_url": meta_data.get(filename)
                })
                
        return {
            "ok": True,
            "images": images,
            "active_portrait_filename": active_portrait,
            "active_avatar_filename": active_avatar
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.post("/roles/{role_id}/gallery")
async def upload_role_gallery_file(role_id: str, file: UploadFile = File(...)):
    """
    上传一张图片到角色画廊
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持的图片格式"})
            
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # 保护文件名
        import re
        safe_name = re.sub(r"[^\w\-_.]", "_", file.filename)
        # 避免重名覆盖，自动加时间戳
        base_name, extension = os.path.splitext(safe_name)
        filename = f"{base_name}_{int(time.time())}{extension}"
        
        target_path = gallery_dir / filename
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return {"ok": True, "filename": filename, "url": f"/api/chat/roles/{role_id}/gallery/{filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.post("/roles/{role_id}/gallery/import")
async def import_role_gallery_file(role_id: str, req: dict):
    """
    从收藏空间导入图片到角色画廊
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        url = req.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 url 缺失"})
            
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        import httpx
        import re
        
        # 解析文件名
        clean_url = url.split("?")[0]
        original_filename = os.path.basename(clean_url)
        safe_name = re.sub(r"[^\w\-_.]", "_", original_filename)
        if not safe_name or "." not in safe_name:
            safe_name = f"imported_{int(time.time())}.png"
            
        base_name, extension = os.path.splitext(safe_name)
        filename = f"{base_name}_{int(time.time())}{extension}"
        target_path = gallery_dir / filename
        
        # 如果是 OBS 上的图片或外部 URL，下载它
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code != 200:
                return JSONResponse(status_code=400, content={"ok": False, "error": f"下载图片失败，HTTP {resp.status_code}"})
            with open(target_path, "wb") as f:
                f.write(resp.content)
        
        # 记录导入来源以去重
        meta_json_path = gallery_dir / "meta.json"
        meta_data = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}
        meta_data[filename] = url
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "filename": filename, "url": f"/api/chat/roles/{role_id}/gallery/{filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.post("/roles/{role_id}/portrait")
async def set_role_portrait(role_id: str, req: dict):
    """
    设置画廊里的某张图片为当前立绘 (复制到角色工作目录下的 portrait.png/jpg 并在 role.json 记录文件名)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        filename = req.get("filename")
        if not filename:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 filename 缺失"})
            
        gallery_dir = workspace / "gallery"
        source_path = gallery_dir / filename
        if not source_path.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "画廊中找不到该图片"})
            
        # 清除已存在的 portrait.*
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            p_path = workspace / name
            if p_path.exists():
                p_path.unlink()
                
        ext = os.path.splitext(filename)[1].lower()
        target_path = workspace / f"portrait{ext}"
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)
        
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["portrait_url"] = f"/api/chat/roles/{role_id}/portrait"
            meta["portrait_filename"] = filename
            with open(role_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "portrait_url": f"/api/chat/roles/{role_id}/portrait?t={int(time.time())}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.get("/roles/{role_id}/portrait")
async def get_role_portrait(role_id: str):
    """
    获取角色的主立绘图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            portrait_path = workspace / name
            if portrait_path.exists():
                return FileResponse(path=portrait_path)
        return JSONResponse(status_code=404, content={"error": "Portrait not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@chat_router.post("/roles/{role_id}/avatar-select")
async def set_role_avatar_select(role_id: str, req: dict):
    """
    从画廊选择图片设为头像 (复制到角色工作目录下的 avatar.png/jpg 并在 role.json 记录文件名)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        filename = req.get("filename")
        if not filename:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 filename 缺失"})
            
        gallery_dir = workspace / "gallery"
        source_path = gallery_dir / filename
        if not source_path.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "画廊中找不到该图片"})
            
        # 清除已存在的 avatar.*
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            a_path = workspace / name
            if a_path.exists():
                a_path.unlink()
                
        ext = os.path.splitext(filename)[1].lower()
        target_path = workspace / f"avatar{ext}"
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)
        
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["avatar_url"] = f"/api/chat/roles/{role_id}/avatar"
            meta["avatar_filename"] = filename
            if "avatar_emoji" in meta:
                del meta["avatar_emoji"]
            with open(role_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "avatar_url": f"/api/chat/roles/{role_id}/avatar?t={int(time.time())}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@chat_router.put("/roles/{role_id}/meta")
async def update_role_meta(role_id: str, req: dict):
    """
    修改角色元数据 (name, description, tags, voice_configured, voice_character)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        role_json_path = workspace / "role.json"
        meta = {}
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
        if "name" in req:
            meta["name"] = req["name"]
        if "description" in req:
            meta["description"] = req["description"]
        if "tags" in req:
            meta["tags"] = req["tags"]
        if "voice_configured" in req:
            meta["voice_configured"] = bool(req["voice_configured"])
        if "voice_character" in req:
            meta["voice_character"] = str(req["voice_character"])
        if "daily_message_enabled" in req:
            meta["daily_message_enabled"] = bool(req["daily_message_enabled"])
            
        with open(role_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        return {"ok": True, "role": meta}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "Vivi"
    speed: float = 1.0
    disable_segmentation: bool = True

@chat_router.post("/tts")
async def generate_tts(req: TTSRequest):
    """
    Volcano TTS 文本转语音接口，支持后端 1 天 (86400秒) 的 diskcache 缓存。
    """
    import hashlib
    import re
    from diskcache import Cache
    from config.config import MY_CONFIG, ENV
    from services.volcovoice_service import VolcoVoiceService

    text = req.text.strip()
    if not text:
        return {"ok": False, "error": "文本内容不能为空"}

    # Helper function for splitting text
    def split_text_by_punctuation(t: str, max_len: int = 200) -> list[str]:
        if len(t) <= max_len:
            return [t]
        sentences = re.split(r'([。！？!?；;\n]+)', t)
        chunks = []
        current_chunk = ""
        i = 0
        while i < len(sentences):
            part = sentences[i]
            punct = sentences[i+1] if i + 1 < len(sentences) else ""
            sentence = part + punct
            i += 2
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) <= max_len:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) > max_len:
                    sub_parts = re.split(r'([，,、：:]+)', sentence)
                    sub_chunk = ""
                    j = 0
                    while j < len(sub_parts):
                        sub_part = sub_parts[j]
                        sub_punct = sub_parts[j+1] if j + 1 < len(sub_parts) else ""
                        sub_sentence = sub_part + sub_punct
                        j += 2
                        if not sub_sentence.strip():
                            continue
                        if len(sub_chunk) + len(sub_sentence) <= max_len:
                            sub_chunk += sub_sentence
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            if len(sub_sentence) > max_len:
                                for k in range(0, len(sub_sentence), max_len):
                                    chunks.append(sub_sentence[k:k+max_len])
                                sub_chunk = ""
                            else:
                                sub_chunk = sub_sentence
                        current_chunk = sub_chunk
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    segments = split_text_by_punctuation(text, 200)
    urls = []
    all_cached = True

    # 获取缓存目录
    cache_dir = os.path.join(MY_CONFIG['cache_config'][ENV]['cache_dir'], 'tts_metadata')
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with Cache(cache_dir) as tts_cache:
            for segment in segments:
                segment_text = segment.strip()
                if not segment_text:
                    continue
                # 生成单个 segment 的缓存 key
                key_src = f"{segment_text}_{req.voice}_{req.speed}"
                cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

                cached_url = tts_cache.get(cache_key)
                if cached_url:
                    log.info(f"TTS Segment 缓存命中: {cache_key} -> {cached_url}")
                    urls.append(cached_url)
                else:
                    all_cached = False
                    # 缓存未命中，调用服务生成
                    audio_url = await VolcoVoiceService.generate_voice(
                        text=segment_text,
                        voice_character=req.voice,
                        speed=req.speed,
                        disable_segmentation=req.disable_segmentation
                    )
                    # 存入缓存，有效期 1 天 (86400 秒)
                    tts_cache.set(cache_key, audio_url, expire=86400)
                    log.info(f"TTS Segment 写入缓存: {cache_key} -> {audio_url}")
                    urls.append(audio_url)

            return {
                "ok": True, 
                "urls": urls, 
                "url": urls[0] if urls else "", 
                "cached": all_cached
            }
    except Exception as e:
        log.error(f"TTS 生成异常: {e}")
        return {"ok": False, "error": str(e)}


@chat_router.delete("/roles/{role_id}/gallery/{filename}")
async def delete_role_gallery_file(role_id: str, filename: str):
    """
    删除角色相册中的特定图片文件
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        gallery_dir = workspace / "gallery"
        file_path = gallery_dir / filename
        
        # 安全验证，防止路径跨目录遍历
        if not file_path.resolve().is_relative_to(gallery_dir.resolve()):
            return JSONResponse(status_code=400, content={"error": "非法文件请求"})
            
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            
            # 如果被删除的文件刚好是当前主立绘/头像，也把 role.json 中的配置清理掉
            role_json_path = workspace / "role.json"
            if role_json_path.exists():
                with open(role_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                updated = False
                if meta.get("portrait_filename") == filename:
                    meta["portrait_url"] = None
                    meta["portrait_filename"] = None
                    updated = True
                if meta.get("avatar_filename") == filename:
                    meta["avatar_url"] = None
                    meta["avatar_filename"] = None
                    updated = True
                if updated:
                    with open(role_json_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                        
            # 从 meta.json 中移除去重元数据映射
            meta_json_path = gallery_dir / "meta.json"
            if meta_json_path.exists():
                try:
                    with open(meta_json_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    if filename in meta_data:
                        del meta_data[filename]
                        with open(meta_json_path, "w", encoding="utf-8") as f:
                            json.dump(meta_data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            return {"ok": True}
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@chat_router.get("/roles/{role_id}/gallery/{filename}")
async def get_role_gallery_file(role_id: str, filename: str):
    """
    获取画廊中特定的图片文件
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        gallery_dir = workspace / "gallery"
        file_path = gallery_dir / filename
        
        # 安全验证，防止路径跨目录遍历
        if not file_path.resolve().is_relative_to(gallery_dir.resolve()):
            return JSONResponse(status_code=400, content={"error": "非法文件请求"})
            
        if file_path.exists() and file_path.is_file():
            return FileResponse(path=file_path)
            
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

