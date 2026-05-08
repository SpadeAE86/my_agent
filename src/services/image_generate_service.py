# services/image_generate_service.py — 按模型路由图片生成
from __future__ import annotations

from typing import Optional

from utils.call_model_utils import call_doubao_seedream
from utils.call_gpt_image_utils import call_gpt_image_2


async def generate_image(
    *,
    prompt: str,
    model: str,
    size: str,
    reference_image_list: Optional[list[str]] = None,
) -> Optional[str]:
    """
    根据用户选择的模型名调用对应供应商。
    size 为前端计算的 WxH（已为 16 的倍数）或豆包支持的档位字符串。
    """
    if model == "gpt-image-2":
        return await call_gpt_image_2(
            prompt,
            model="gpt-image-2",
            size=size,
            reference_image_list=reference_image_list,
        )
    return await call_doubao_seedream(
        prompt=prompt,
        model=model,
        size=size,
        reference_image_list=reference_image_list,
    )
