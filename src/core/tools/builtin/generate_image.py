# core/tools/builtin/generate_image.py — AI 生图工具
from __future__ import annotations
import uuid
import datetime
import time
import json
from typing import Optional, List, Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from models.pydantic.request import SeedreamModel
from services.image_generate_service import generate_image as service_generate_image
from services.media_mirror_service import mirror_remote_url_to_obs
from services.image_history_db_service import image_history_db_service
from services.http_request_trace_service import http_request_trace_service
from infra.logging.logger import logger as log


class GenerateImageInput(ToolInput):
    """generate_image 工具的入参。"""
    prompt: str = Field(..., description="生图提示词，详细描述想要生成的画面内容")
    size: str = Field(default="1440x2560", description="图片尺寸。注意：当使用 Seedream 5.0 时，总像素必须达到 3,686,400 像素以上（如 '1440x2560' (9:16), '2560x1440' (16:9), 或 '1920x1920' (1:1)）。默认使用 '1440x2560'。")
    model: str = Field(default="Seedream 5.0", description="使用的生图模型，如 'Seedream 5.0' (要求使用 2K 级别的高分辨率尺寸，不支持 1K 尺寸), 'Seedream 4.5', 'gpt-image-2'")
    reference_image_list: Optional[List[str]] = Field(default=None, description="参考图URL列表")
    ratio: Optional[str] = Field(default=None, description="图片宽高比")


class GenerateImageOutput(ToolOutput):
    """generate_image 工具的出参。"""
    id: str = Field(..., description="任务 ID")
    url: Optional[str] = Field(None, description="生成的图片公网 URL")
    prompt: str = Field(..., description="提示词")
    model: str = Field(..., description="模型名")
    size: str = Field(..., description="尺寸")
    ratio: Optional[str] = Field(None, description="比例")
    time: str = Field(..., description="时间串")
    type: str = Field(..., description="生图类型: t2i (文生图) 或 i2i (图生图)")
    status: str = Field(..., description="状态: success 或 failed")
    error: Optional[str] = Field(None, description="错误信息")


async def handle_generate_image(
    params: GenerateImageInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ToolOutput:
    t0 = time.monotonic()
    task_id = str(uuid.uuid4())
    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d %H:%M")
    run_start = datetime.datetime.now(datetime.timezone.utc)
    
    # 区分 t2i 和 i2i
    gen_type = "i2i" if params.reference_image_list else "t2i"
    
    # 匹配模型枚举
    try:
        model_enum = SeedreamModel(params.model)
    except ValueError:
        # 降级
        model_enum = SeedreamModel.V5_0

    # 1. 创建 HTTP 追踪行 (看板详情联表)
    request_body = {
        "prompt": params.prompt,
        "model": params.model,
        "size": params.size,
        "reference_image_list": params.reference_image_list,
        "ratio": params.ratio,
        "type": gen_type,
    }
    trace_id = await http_request_trace_service.create_initial(
        request_url="/image (via agent tool)",
        http_method="POST",
        request_body=request_body,
        business_type="IMAGE_GEN",
        method_name="generate_image_tool",
        upstream_task_id=task_id,
    )
    
    # 2. 插入 running 状态到历史看板
    payload = {
        "id": task_id,
        "prompt": params.prompt,
        "model": params.model,
        "size": params.size,
        "ratio": params.ratio,
        "time": time_str,
        "type": gen_type,
        "status": "running",
        "request_id": trace_id,
        "referenceMedia": [{"url": m, "type": "image"} for m in params.reference_image_list] if params.reference_image_list else None,
        "current_run_started_at": run_start,
    }
    await image_history_db_service.upsert_many([payload])
    log.info(f"[generate_image_tool] 任务已创建: id={task_id}, status=running")
    
    # 3. 调用生图服务
    try:
        img_url = await service_generate_image(
            prompt=params.prompt,
            model=model_enum.value,
            size=params.size,
            reference_image_list=params.reference_image_list,
        )
        
        if img_url:
            prefix = "ai_picture/generated_image"
            obs_url = img_url
            try:
                # 镜像到 OBS
                obs_url = await mirror_remote_url_to_obs(img_url, obs_prefix=prefix)
                await image_history_db_service.upsert_many(
                    [
                        {
                            "id": task_id,
                            "obs_url": obs_url,
                            "doubao_url": img_url,
                            "status": "success",
                        }
                    ]
                )
            except Exception as obs_err:
                log.warning(f"[generate_image_tool] mirror to OBS failed: {obs_err}")
                await image_history_db_service.upsert_many(
                    [
                        {
                            "id": task_id,
                            "doubao_url": img_url,
                            "status": "success",
                        }
                    ]
                )
            
            # 成功结束 Trace
            duration_ms = int((time.monotonic() - t0) * 1000)
            await http_request_trace_service.finalize(
                trace_id,
                status_code=200,
                response_body={"success": True, "image_url": img_url, "obs_url": obs_url},
                duration_ms=duration_ms,
                business_success=True,
            )
            
            # 返回 JSON 字符串 (以便前端 JSON.parse 反序列化为 GeneratedItem)
            output_data = {
                "id": task_id,
                "url": obs_url,
                "prompt": params.prompt,
                "model": params.model,
                "size": params.size,
                "ratio": params.ratio,
                "time": time_str,
                "type": gen_type,
                "status": "success",
            }
            return ToolOutput(
                success=True,
                message=json.dumps(output_data, ensure_ascii=False),
                data=output_data,
            )
        else:
            raise ValueError("生成失败，未获取到图片 URL")
            
    except Exception as e:
        error_msg = str(e)
        log.error(f"[generate_image_tool] 生图异常: {error_msg}")
        await image_history_db_service.upsert_many(
            [{"id": task_id, "status": "failed", "error": error_msg}]
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        await http_request_trace_service.finalize(
            trace_id,
            status_code=500,
            error_message=error_msg,
            response_body={"success": False, "error": error_msg},
            duration_ms=duration_ms,
            business_success=False,
        )
        
        output_data = {
            "id": task_id,
            "url": None,
            "prompt": params.prompt,
            "model": params.model,
            "size": params.size,
            "ratio": params.ratio,
            "time": time_str,
            "type": gen_type,
            "status": "failed",
            "error": error_msg,
        }
        return ToolOutput(
            success=False,
            message=json.dumps(output_data, ensure_ascii=False),
            data=output_data,
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="generate_image",
    description=(
        "调用 AI 图像模型生成高质量创意图片。支持文生图(t2i)和图生图(i2i)。"
        "返回结果为 JSON 字符串，包含图片的任务 ID、OBS URL、状态等，可以直接被系统看板和聊天卡片渲染。"
    ),
    input_schema=GenerateImageInput,
    output_schema=GenerateImageOutput,
    handler=handle_generate_image,
    tags=["image", "generate", "ai"],
    timeout=60.0,
    is_concurrency_safe=True,
)
