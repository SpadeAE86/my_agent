# services/image_generate_service.py — 按模型路由图片生成
from __future__ import annotations

from typing import Optional
import math

from utils.call_model_utils import call_doubao_seedream
from utils.call_gpt_image_utils import call_gpt_image_2
from utils.call_gpt_image_wangsu_utils import call_gpt_image_wangsu_edge
from infra.logging.logger import logger as log


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
    # 对 Seedream 5.0 进行尺寸自动升阶，规避总像素低于 3,686,400 的 400 报错
    if "seedream-5-0" in model.lower() or "seedream 5.0" in model.lower():
        try:
            if "x" in size:
                w_str, h_str = size.split("x")
                w, h = int(w_str), int(h_str)
                if w * h < 3686400:
                    ratio = w / h
                    # 预设几种常见比例的升级尺寸
                    if 0.5 <= ratio <= 0.6:  # 9:16 (0.56)
                        size = "1440x2560"
                    elif 1.7 <= ratio <= 1.8: # 16:9 (1.77)
                        size = "2560x1440"
                    elif 0.9 <= ratio <= 1.1: # 1:1 (1.0)
                        size = "1920x1920"
                    else:
                        scale = math.sqrt(3686400 / (w * h))
                        new_w = int(math.ceil(w * scale / 16) * 16)
                        new_h = int(math.ceil(h * scale / 16) * 16)
                        size = f"{new_w}x{new_h}"
                    log.info(f"[Seedream 5.0] 尺寸自动升阶: {w}x{h} -> {size}")
            elif size in ["1K", "2K", "3K", "4K"]:
                if size == "1K":
                    size = "2K"
                    log.info(f"[Seedream 5.0] 尺寸档位升阶: 1K -> 2K")
        except Exception as e:
            log.warning(f"[Seedream 5.0] 尺寸自动升阶失败: {e}")

    if model == "gpt-image-2":
        return await call_gpt_image_2(
            prompt,
            model="gpt-image-2",
            size=size,
            reference_image_list=reference_image_list,
        )
    if model == "gpt-image-2-wangsu":
        return await call_gpt_image_wangsu_edge(
            prompt,
            size=size,
            reference_image_list=reference_image_list,
        )
    return await call_doubao_seedream(
        prompt=prompt,
        model=model,
        size=size,
        reference_image_list=reference_image_list,
    )
