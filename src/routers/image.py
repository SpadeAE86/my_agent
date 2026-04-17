# routers/image.py — 图片和文本生成路由
# 端点:
#   POST /image          — 调用 Seedream 模型生成图片
#   POST /text           — 调用 Seed 文本模型生成文本

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

from models.pydantic.request import ImageGenerateRequest, TextGenerateRequest
from utils.call_model_utils import call_doubao_seedream, call_doubao_seedtext
from infra.logging.logger import logger as log

image_router = APIRouter(prefix="", tags=["image", "text"])


class ImageGenerateResponse(BaseModel):
    """图片生成响应"""
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


class TextGenerateResponse(BaseModel):
    """文本生成响应"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None


@image_router.post("/image", response_model=ImageGenerateResponse)
async def generate_image(req: ImageGenerateRequest):
    """
    调用豆包 Seedream 模型生成图片
    
    支持模型:
    - Seedream 4.0
    - Seedream 4.5
    - Seedream 5.0 (默认)
    
    尺寸支持:
    - 4.0: 1K, 2K, 4K, 或自定义宽高
    - 4.5: 2K, 4K, 或自定义宽高
    - 5.0: 2K, 3K, 或自定义宽高
    """
    try:
        log.info(f"收到图片生成请求: model={req.model}, size={req.size}")
        log.info(f"提示词: {req.prompt}")
        
        image_url = call_doubao_seedream(
            prompt=req.prompt,
            model=req.model.value,
            size=req.size
        )
        
        if image_url:
            log.info(f"图片生成成功: {image_url}")
            return ImageGenerateResponse(success=True, image_url=image_url)
        else:
            log.error("图片生成失败")
            return ImageGenerateResponse(success=False, error="图片生成失败，请检查提示词或重试")
            
    except Exception as e:
        log.error(f"图片生成异常: {e}")
        return ImageGenerateResponse(success=False, error=str(e))


@image_router.post("/text", response_model=TextGenerateResponse)
async def generate_text(req: TextGenerateRequest):
    """
    调用豆包 Seed 文本模型生成文本
    
    支持模型:
    - Seed 2.0 Pro (默认)
    - Seed 2.0 Lite
    - Seed 2.0 Mini
    """
    try:
        log.info(f"收到文本生成请求: model={req.model}")
        if req.system_prompt:
            log.info(f"系统提示词已提供")
        log.info(f"提示词: {req.prompt}")
        
        text = call_doubao_seedtext(
            prompt=req.prompt,
            model=req.model.value,
            system_prompt=req.system_prompt
        )
        
        if text:
            log.info(f"文本生成成功")
            return TextGenerateResponse(success=True, text=text)
        else:
            log.error("文本生成失败")
            return TextGenerateResponse(success=False, error="文本生成失败，请检查提示词或重试")
            
    except Exception as e:
        log.error(f"文本生成异常: {e}")
        return TextGenerateResponse(success=False, error=str(e))
