# Shared imports for sub-routers
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
from pydantic import BaseModel

router = APIRouter()
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "Vivi"
    speed: float = 1.0
    disable_segmentation: bool = True
    bubble_id: Optional[str] = None
    byte_stream: bool = True

@router.post("/tts")
async def generate_tts(req: TTSRequest):
    """
    Volcano / Qwen TTS 文本转语音流式接口，支持数据库句级缓存和整体 OBS 拼接。
    """
    import hashlib
    from services.tts_services.voice_tts_service import voice_tts_service

    text = req.text.strip()
    if not text:
        async def err_generator():
            yield f"data: {json.dumps({'event_type': 'error', 'message': '文本内容不能为空'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            err_generator(),
            media_type="text/event-stream"
        )

    bubble_id = req.bubble_id
    if not bubble_id:
        text_hash = hashlib.md5(f"{text}_{req.voice}".encode('utf-8')).hexdigest()
        bubble_id = f"gen_hash_{text_hash}"

    return StreamingResponse(
        voice_tts_service.generate_tts_stream(
            bubble_id=bubble_id,
            voice_character=req.voice,
            text=text,
            speed=req.speed,
            byte_stream=req.byte_stream
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止 Nginx 缓冲 SSE
        },
    )


