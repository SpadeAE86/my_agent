import json
import os
from typing import Any

from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput


class ReadGraphInput(ToolInput):
    """read_graph 工具的入参。"""

    file_name: str = Field(
        ...,
        description="要读取的力导图 ID (即文件名，不含 .json 后缀)，例如 'tech_stack_graph'",
    )


class ReadGraphOutput(ToolOutput):
    """read_graph 工具的出参。"""

    graph_name: str = Field(default="", description="力导图的名称")
    graph_id: str = Field(default="", description="力导图的 ID / 文件名")
    nodes: list[dict] = Field(default_factory=list, description="反序列化后的扁平节点数据列表")
    edges: list[dict] = Field(default_factory=list, description="反序列化后的扁平边数据列表")


async def handle_read_graph(
    params: ReadGraphInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ReadGraphOutput:
    """
    1. 从 data/graphs/{file_name}.json 中读取嵌套格式的力导图数据
    2. 反序列化为扁平格式的 nodes 和 edges，方便 Agent 阅读和二次编辑
    """
    from pathlib import Path
    current = Path(__file__).resolve()
    while current.name != "src" and current.parent != current:
        current = current.parent
    project_root = current.parent
    base_dir = os.path.join(project_root, "data", "graphs")
    file_path = os.path.join(base_dir, f"{params.file_name}.json")

    if not os.path.exists(file_path):
        return ReadGraphOutput(
            success=False,
            message=f"未找到力导图: {params.file_name}",
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 反序列化 nodes 嵌套格式到扁平格式
        flat_nodes = []
        for n in data.get("nodes", []):
            flat_nodes.append({
                "id": n["id"],
                "label": n.get("data", {}).get("label", n["id"]),
                "group": n.get("data", {}).get("group", "default"),
            })

        # 反序列化 edges 嵌套格式到扁平格式
        flat_edges = []
        for e in data.get("edges", []):
            flat_edges.append({
                "source": e["source"],
                "target": e["target"],
                "label": e.get("data", {}).get("label", ""),
            })

        return ReadGraphOutput(
            success=True,
            message=f"成功读取并反序列化力导图: {params.file_name}",
            graph_name=params.file_name,
            graph_id=params.file_name,
            nodes=flat_nodes,
            edges=flat_edges,
        )
    except Exception as e:
        return ReadGraphOutput(
            success=False,
            message=f"读取力导图失败: {type(e).__name__}: {e}",
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="read_graph",
    description="读取并反序列化指定的力导图 ID，将其解析为干净扁平的节点和边结构，以便你可以理解现有关系、进行分析或二次修改。",
    input_schema=ReadGraphInput,
    output_schema=ReadGraphOutput,
    handler=handle_read_graph,
    tags=["graph", "visualization", "data"],
    timeout=30.0,
    is_concurrency_safe=True,
)
