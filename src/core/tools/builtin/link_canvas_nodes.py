# core/tools/builtin/link_canvas_nodes.py — 画布连接节点工具
from __future__ import annotations
import uuid
from typing import Optional, Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from services.workspace_db_service import workspace_db_service


class LinkCanvasNodesInput(ToolInput):
    """link_canvas_nodes 工具的入参。"""
    workspace_id: str = Field(..., description="画布工程 ID (session_id)")
    source_node_id: str = Field(..., description="起始节点 ID (source node)")
    target_node_id: str = Field(..., description="目标节点 ID (target node)")
    label: Optional[str] = Field(default="", description="两个节点之间连线上的说明文字，通常是进行的演进操作/提示词")
    edge_id: Optional[str] = Field(default=None, description="自定义连线 ID，如果不填则自动生成，格式如 'edge_xxxxxx'")


async def handle_link_canvas_nodes(
    params: LinkCanvasNodesInput,
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

    # 验证起始节点和目标节点是否存在。支持 #3 这样 display_id 自增编号的查找
    nodes = await workspace_db_service.list_nodes(params.workspace_id)
    
    def find_node_by_any_id(any_id: str) -> Optional[Any]:
        any_id = any_id.strip()
        if any_id.startswith("#"):
            try:
                display_id_val = int(any_id[1:])
                return next((n for n in nodes if n.data and isinstance(n.data, dict) and n.data.get("display_id") == display_id_val), None)
            except ValueError:
                pass
        return next((n for n in nodes if n.id == any_id), None)

    source_node = find_node_by_any_id(params.source_node_id)
    target_node = find_node_by_any_id(params.target_node_id)
    
    if not source_node:
        return ToolOutput(
            success=False,
            message=f"起始节点 {params.source_node_id} 不在工程 {params.workspace_id} 中",
            data={}
        )
    if not target_node:
        return ToolOutput(
            success=False,
            message=f"目标节点 {params.target_node_id} 不在工程 {params.workspace_id} 中",
            data={}
        )

    # 提取数据库底层的真实 UUID
    source_uuid = source_node.id
    target_uuid = target_node.id

    edge_id = params.edge_id or f"edge_{uuid.uuid4().hex[:6]}"

    try:
        edge = await workspace_db_service.upsert_edge(
            workspace_id=params.workspace_id,
            edge_id=edge_id,
            source=source_uuid,
            target=target_uuid,
            label=params.label
        )
        return ToolOutput(
            success=True,
            message=f"成功创建连线 {edge_id} 从 {params.source_node_id} ({source_uuid}) 指向 {params.target_node_id} ({target_uuid})",
            data={
                "id": edge.id,
                "workspace_id": edge.workspace_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "label": edge.label
            }
        )
    except Exception as e:
        return ToolOutput(
            success=False,
            message=f"连接画布节点失败: {type(e).__name__}: {e}",
            data={}
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="link_canvas_nodes",
    description="在指定的画布工程中连接两个节点，指示图片之间的演进路径，并可添加演进说明标签 (label)。",
    input_schema=LinkCanvasNodesInput,
    output_schema=ToolOutput,
    handler=handle_link_canvas_nodes,
    tags=["canvas", "workspace", "edge"],
    timeout=30.0,
    is_concurrency_safe=True,
)
