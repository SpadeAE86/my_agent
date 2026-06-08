# core/tools/builtin/update_canvas_node.py — 更新画布节点工具
from __future__ import annotations
from typing import Optional, Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from services.workspace_db_service import workspace_db_service
from services.media_mirror_service import mirror_remote_url_to_obs, is_obs_url
from infra.logging.logger import logger as log


class UpdateCanvasNodeInput(ToolInput):
    """update_canvas_node 工具的入参。"""
    workspace_id: str = Field(..., description="画布工程 ID (session_id)")
    node_id: str = Field(..., description="要更新的节点 ID (例如 'node_xxxxxx'，或者 '#3' 这种在画布上可见的自增编号)")
    prompt: Optional[str] = Field(default=None, description="要更新的提示词。若是生图节点，则更新生图提示词；若是模板节点，则更新模板内容。")
    image_url: Optional[str] = Field(default=None, description="要更新的图片 URL，如果不需要更新则不填")
    status: Optional[str] = Field(default=None, description="更新节点状态: 'success', 'generating', 'failed'")
    error_message: Optional[str] = Field(default=None, description="失败时的错误提示信息")
    x: Optional[float] = Field(default=None, description="要更新的 X 坐标，如果位置不变则不填")
    y: Optional[float] = Field(default=None, description="要更新的 Y 坐标，如果位置不变则不填")
    template_name: Optional[str] = Field(default=None, description="要更新的模板名称，仅当节点为 prompt_template 类型时生效")
    template_text: Optional[str] = Field(default=None, description="要更新的模板提示词文本内容，仅当节点为 prompt_template 类型时生效")
    template_values: Optional[dict] = Field(default=None, description="要更新的槽位变量映射字典（例如 {'主体': '一只猫'}），仅当节点为 gen_node 时生效")


async def handle_update_canvas_node(
    params: UpdateCanvasNodeInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ToolOutput:
    # 验证工程和节点是否存在。我们首先支持使用自增display_id进行精准查询
    nodes = await workspace_db_service.list_nodes(params.workspace_id)
    target_node = None
    
    node_id_str = params.node_id.strip()
    if node_id_str.startswith("#"):
        try:
            display_id_val = int(node_id_str[1:])
            target_node = next((n for n in nodes if n.data and isinstance(n.data, dict) and n.data.get("display_id") == display_id_val), None)
        except ValueError:
            pass
            
    if not target_node:
        target_node = next((n for n in nodes if n.id == node_id_str), None)
        
    if not target_node:
        return ToolOutput(
            success=False,
            message=f"工程 {params.workspace_id} 下不存在节点 {params.node_id}",
            data={}
        )

    # 准备合并更新后的数据
    node_data = target_node.data or {}
    
    if target_node.type == "prompt_template":
        if params.prompt is not None:
            node_data["template_text"] = params.prompt
        if params.template_text is not None:
            node_data["template_text"] = params.template_text
        if params.template_name is not None:
            node_data["name"] = params.template_name
    else:
        if params.prompt is not None:
            node_data["prompt"] = params.prompt
        if params.image_url is not None:
            image_url = params.image_url.strip()
            # 自动修正检测：如果更新的目标链接非 OBS 链接，同步镜像之，防止失效
            if image_url and not is_obs_url(image_url):
                try:
                    log.info(f"[update_canvas_node] Detect non-OBS update image URL {image_url[:60]}..., mirroring synchronously.")
                    image_url = await mirror_remote_url_to_obs(image_url, obs_prefix="ai_picture/generated_image")
                    log.info(f"[update_canvas_node] Mirror success, new OBS URL: {image_url}")
                except Exception as e:
                    log.warning(f"[update_canvas_node] Synchronous mirror to OBS failed: {e}")
            node_data["image_url"] = image_url
        if params.status is not None:
            node_data["status"] = params.status
        if params.error_message is not None:
            node_data["error_message"] = params.error_message
        if params.template_values is not None:
            # 合并更新
            node_data["template_values"] = {
                **(node_data.get("template_values") or {}),
                **params.template_values
            }

    new_x = params.x if params.x is not None else target_node.x
    new_y = params.y if params.y is not None else target_node.y

    try:
        node = await workspace_db_service.upsert_node(
            workspace_id=params.workspace_id,
            node_id=target_node.id,
            node_type=target_node.type,
            x=new_x,
            y=new_y,
            data=node_data
        )
        return ToolOutput(
            success=True,
            message=f"成功更新节点 {target_node.id} ({node_id_str})",
            data={
                "id": node.id,
                "workspace_id": node.workspace_id,
                "type": node.type,
                "x": node.x,
                "y": node.y,
                "data": node.data
            }
        )
    except Exception as e:
        return ToolOutput(
            success=False,
            message=f"更新画布节点失败: {type(e).__name__}: {e}",
            data={}
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="update_canvas_node",
    description="更新指定节点式画布中某个已存在的图片卡片节点的状态、坐标、生图提示词或图片 URL。",
    input_schema=UpdateCanvasNodeInput,
    output_schema=ToolOutput,
    handler=handle_update_canvas_node,
    tags=["canvas", "workspace", "node"],
    timeout=30.0,
    is_concurrency_safe=True,
)
