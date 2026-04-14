# models/pydantic/request.py — API 请求体定义
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """POST /chat 的请求体"""
    message: str = Field(..., description="用户消息内容", min_length=1)
    session_id: Optional[str] = Field(default=None, description="会话 ID, 不传则新建")
    user_id: str = Field(default="default_user", description="用户标识")
    model: str = Field(default="gemini-3-pro", description="指定模型")
    max_iterations: int = Field(default=10, description="Agent 最大迭代轮次", ge=1, le=50)
