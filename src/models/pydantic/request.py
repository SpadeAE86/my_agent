# models/pydantic/request.py — API 请求体定义
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SeedreamModel(str, Enum):
    """豆包 Seedream 模型版本（用户可见名称）"""
    V4_0 = "Seedream 4.0"
    V4_5 = "Seedream 4.5"
    V5_0 = "Seedream 5.0"


class SeedTextModel(str, Enum):
    """豆包 Seed 文本模型版本（用户可见名称）"""
    V2_0_PRO = "Seed 2.0 Pro"
    V2_0_LITE = "Seed 2.0 Lite"
    V2_0_MINI = "Seed 2.0 Mini"


SEEDREAM_MODEL_MAP = {
    "Seedream 4.0": "doubao-seedream-4-0-250828",
    "Seedream 4.5": "doubao-seedream-4-5-251128",
    "Seedream 5.0": "doubao-seedream-5-0-260128",
}

SEEDTEXT_MODEL_MAP = {
    "Seed 2.0 Pro": "doubao-seed-2-0-pro-260215",
    "Seed 2.0 Lite": "doubao-seed-2-0-lite-260215",
    "Seed 2.0 Mini": "doubao-seed-2-0-mini-260215",
}


class ImageGenerateRequest(BaseModel):
    """POST /image 的请求体"""
    prompt: str = Field(..., description="图片生成提示词", min_length=1)
    size: str = Field(default="720x1280", description="图片尺寸，如 720x1280、2K、4K 等")
    model: SeedreamModel = Field(default=SeedreamModel.V5_0, description="使用的模型版本")


class TextGenerateRequest(BaseModel):
    """POST /text 的请求体"""
    prompt: str = Field(..., description="文本生成提示词", min_length=1)
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    model: SeedTextModel = Field(default=SeedTextModel.V2_0_PRO, description="使用的模型版本")


class PromptTemplateRequest(BaseModel):
    """保存提示词模板的请求体"""
    name: str = Field(..., description="模板名称", min_length=1)
    content: str = Field(..., description="模板内容", min_length=1)


class ChatRequest(BaseModel):
    """POST /chat 的请求体"""
    message: str = Field(..., description="用户消息内容", min_length=1)
    session_id: Optional[str] = Field(default=None, description="会话 ID, 不传则新建")
    user_id: str = Field(default="default_user", description="用户标识")
    model: str = Field(default="gemini-3-pro", description="指定模型")
    max_iterations: int = Field(default=10, description="Agent 最大迭代轮次", ge=1, le=50)
