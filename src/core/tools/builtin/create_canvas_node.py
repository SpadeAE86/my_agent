# core/tools/builtin/create_canvas_node.py — 创建画布节点工具
from __future__ import annotations
import uuid
from typing import Optional, Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from services.workspace_db_service import workspace_db_service
from services.media_mirror_service import mirror_remote_url_to_obs, is_obs_url
from infra.logging.logger import logger as log


class CreateCanvasNodeInput(ToolInput):
    """create_canvas_node 工具的入参。"""
    workspace_id: str = Field(..., description="当前激活的画布工程 ID (session_id)")
    node_id: Optional[str] = Field(default=None, description="自定义节点 ID，如果不填则自动生成，格式如 'node_xxxxxx'")
    x: float = Field(default=100.0, description="节点在画布上的 X 坐标")
    y: float = Field(default=100.0, description="节点在画布上的 Y 坐标")
    prompt: str = Field(..., description="该节点的提示词内容。如果是生图节点，则为生图提示词；如果是模板节点，则为模板文本内容（可包含 {变量名: 默认值} 插槽）。")
    image_url: Optional[str] = Field(default="", description="图片的 URL，如果是新生成的或参考图请填入")
    model: str = Field(default="gpt-5.4", description="生图模型，如 'gpt-5.4', 'gpt-image-2'")
    status: str = Field(default="success", description="节点状态: 'success' (成功), 'generating' (生成中), 'failed' (失败)")
    node_type: str = Field(default="gen_node", description="要创建的节点类型，可选: 'gen_node' (生成节点), 'prompt_template' (提示词模板节点)")
    template_name: Optional[str] = Field(default="未命名模板", description="提示词模板的名称。仅当 node_type 为 'prompt_template' 时生效。")


async def handle_create_canvas_node(
    params: CreateCanvasNodeInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ToolOutput:
    # 验证工程是否存在
    ws = await workspace_db_service.get_workspace(params.workspace_id)
    if not ws:
        return ToolOutput(
            success=False,
            message=f"工程 {params.workspace_id} 不存在，请先创建工程",
            data={}
        )

    node_id = params.node_id or f"node_{uuid.uuid4().hex[:6]}"
    
    # 自动计算自增 display_id
    nodes = await workspace_db_service.list_nodes(params.workspace_id)
    max_display_id = 0
    for n in nodes:
        if n.data and isinstance(n.data, dict):
            did = n.data.get("display_id")
            if did and isinstance(did, (int, float)):
                if int(did) > max_display_id:
                    max_display_id = int(did)
    
    image_url = (params.image_url or "").strip()
    # 自动修正检测：如果传入的图片链接非 OBS 链接，同步镜像之，防止火山引擎等具有时效性的链接过期失效
    if image_url and not is_obs_url(image_url):
        try:
            log.info(f"[create_canvas_node] Detect non-OBS image URL {image_url[:60]}..., mirroring synchronously.")
            image_url = await mirror_remote_url_to_obs(image_url, obs_prefix="ai_picture/generated_image")
            log.info(f"[create_canvas_node] Mirror success, new OBS URL: {image_url}")
        except Exception as e:
            log.warning(f"[create_canvas_node] Synchronous mirror to OBS failed: {e}")

    if params.node_type == "prompt_template":
        data = {
            "display_id": max_display_id + 1,
            "name": params.template_name or "未命名模板",
            "template_text": params.prompt
        }
        node_type = "prompt_template"
    else:
        data = {
            "display_id": max_display_id + 1,
            "prompt": params.prompt,
            "image_url": image_url,
            "model": params.model,
            "status": params.status,
            "error_message": "",
            "ratio": "9:16",
            "sizeLevel": "2K",
            "reference_images": []
        }
        node_type = "gen_node"

    try:
        node = await workspace_db_service.upsert_node(
            workspace_id=params.workspace_id,
            node_id=node_id,
            node_type=node_type,
            x=params.x,
            y=params.y,
            data=data
        )
        return ToolOutput(
            success=True,
            message=f"成功在坐标 ({params.x}, {params.y}) 创建 {node_type} 节点 {node_id}",
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
            message=f"创建画布节点失败: {type(e).__name__}: {e}",
            data={}
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="create_canvas_node",
    description="在指定的节点式画布工程 (workspace_id) 中，创建一个图片卡片节点。必须提供提示词 prompt 以及坐标 x 和 y。",
    input_schema=CreateCanvasNodeInput,
    output_schema=ToolOutput,
    handler=handle_create_canvas_node,
    tags=["canvas", "workspace", "node"],
    timeout=30.0,
    is_concurrency_safe=True,
)
