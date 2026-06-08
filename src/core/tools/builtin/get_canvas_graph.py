# core/tools/builtin/get_canvas_graph.py — 获取画布图结构工具
from __future__ import annotations
from typing import Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from services.workspace_db_service import workspace_db_service


class GetCanvasGraphInput(ToolInput):
    """get_canvas_graph 工具的入参。"""
    workspace_id: str = Field(..., description="要读取的画布工程 ID (session_id)")


async def handle_get_canvas_graph(
    params: GetCanvasGraphInput,
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
            message=f"工程 {params.workspace_id} 不存在",
            data={}
        )

    try:
        nodes = await workspace_db_service.list_nodes(params.workspace_id)
        edges = await workspace_db_service.list_edges(params.workspace_id)

        serialized_nodes = []
        for n in nodes:
            serialized_nodes.append({
                "id": n.id,
                "type": n.type,
                "x": n.x,
                "y": n.y,
                "data": n.data or {}
            })

        serialized_edges = []
        for e in edges:
            serialized_edges.append({
                "id": e.id,
                "source": e.source_node_id,
                "target": e.target_node_id,
                "label": e.label or ""
            })

        graph_summary = (
            f"工程 '{ws.name}' ({ws.id}) 目前包含 {len(nodes)} 个节点 and {len(edges)} 条连线。\n"
            f"节点列表:\n"
        )
        display_map = {}
        for n in serialized_nodes:
            d = n["data"]
            display_id = d.get("display_id")
            display_id_str = f"#{display_id}" if display_id else n["id"]
            display_map[n["id"]] = display_id_str
            if n["type"] == "prompt_template":
                graph_summary += f"  - ID: {display_id_str} ({n['id']}) (位置: {n['x']:.1f}, {n['y']:.1f}) | 节点类型: prompt_template | 模板名称: '{d.get('name')}' | 模板内容: '{d.get('template_text')}'\n"
            else:
                graph_summary += f"  - ID: {display_id_str} ({n['id']}) (位置: {n['x']:.1f}, {n['y']:.1f}) | 节点类型: {n['type']} | 状态: {d.get('status')} | 提示词: '{d.get('prompt')}' | 图片: {d.get('image_url')}\n"
        
        graph_summary += "连线关系:\n"
        for e in serialized_edges:
            source_show = display_map.get(e['source'], e['source'])
            target_show = display_map.get(e['target'], e['target'])
            graph_summary += f"  - {source_show} ---> {target_show} (说明: '{e['label']}')\n"

        return ToolOutput(
            success=True,
            message=graph_summary,
            data={
                "workspace": {
                    "id": ws.id,
                    "name": ws.name,
                    "created_at": ws.created_at.isoformat() if ws.created_at else None,
                    "updated_at": ws.updated_at.isoformat() if ws.updated_at else None,
                },
                "nodes": serialized_nodes,
                "edges": serialized_edges
            }
        )
    except Exception as e:
        return ToolOutput(
            success=False,
            message=f"读取画布工程图数据失败: {type(e).__name__}: {e}",
            data={}
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="get_canvas_graph",
    description="获取指定工程画布 (workspace_id) 中的所有图片卡片节点列表及它们之间的连线演进关系。让你能在对话中知晓用户当前的画板图网状态。",
    input_schema=GetCanvasGraphInput,
    output_schema=ToolOutput,
    handler=handle_get_canvas_graph,
    tags=["canvas", "workspace", "graph"],
    timeout=30.0,
    is_concurrency_safe=True,
)
