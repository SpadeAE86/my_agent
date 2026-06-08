import os
from typing import Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput

class ReadImageInput(ToolInput):
    """read_image 工具的入参。"""
    image_url: str = Field(
        ..., 
        description="要观察和分析的图片公网 URL 链接（支持 png, jpg, jpeg, webp 等格式）。"
    )

class ReadImageOutput(ToolOutput):
    """read_image 工具的出参。"""
    url: str = Field(default="", description="加载的图片公网 URL")

async def handle_read_image(
    params: ReadImageInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ReadImageOutput:
    url = params.image_url.strip()
    
    return ReadImageOutput(
        success=True,
        message=f"已成功将图片加载至你的视觉上下文中：{url}。在接下来的推理轮次中，你将可以直接‘看见’并分析该图。",
        data={"url": url},
        url=url
    )

# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="read_image",
    description=(
        "用于读取、观察并分析指定 URL 图像的视觉工具。调用此工具后，"
        "目标图像会被挂载到你的视觉输入（image_url）中，使你在下一轮推理中能够原生‘看见’它。"
        "当用户给你发送了图片链接、或者你想回看历史中某次生成的图片时，请调用此工具。"
    ),
    input_schema=ReadImageInput,
    output_schema=ReadImageOutput,
    handler=handle_read_image,
    tags=["image", "system"],
    timeout=10.0,
    is_concurrency_safe=True,
)
