import os
import json
from typing import Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput

class ListGraphsInput(ToolInput):
    """list_graphs 工具的入参。"""
    pass

class ListGraphsOutput(ToolOutput):
    """list_graphs 工具的出参。"""
    graphs: list[dict] = Field(default_factory=list, description="保存的所有力导向图列表")

async def handle_list_graphs(
    params: ListGraphsInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ListGraphsOutput:
    
    from pathlib import Path
    current = Path(__file__).resolve()
    while current.name != "src" and current.parent != current:
        current = current.parent
    project_root = current.parent
    base_dir = os.path.join(project_root, "data", "graphs")
    
    if not os.path.exists(base_dir):
        return ListGraphsOutput(success=True, graphs=[])
        
    graphs = []
    try:
        for f in os.listdir(base_dir):
            if f.endswith(".json"):
                name = f[:-5]
                file_path = os.path.join(base_dir, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        graphs.append({
                            "graph_id": name,
                            "graph_name": name,
                            "node_count": len(data.get("nodes", [])),
                            "edge_count": len(data.get("edges", [])),
                        })
                except Exception:
                    pass
        return ListGraphsOutput(
            success=True,
            message=f"成功获取了 {len(graphs)} 个保存的力导图",
            graphs=graphs
        )
    except Exception as e:
        return ListGraphsOutput(
            success=False,
            message=f"获取力导图列表失败: {type(e).__name__}: {e}",
            graphs=[]
        )

tool_def = ToolDef(
    name="list_graphs",
    description="列出系统内已保存的所有力导图列表。返回简要信息包括图ID、名称、节点数和连线数。",
    input_schema=ListGraphsInput,
    output_schema=ListGraphsOutput,
    handler=handle_list_graphs,
    tags=["graph", "visualization", "data"],
    timeout=10.0,
    is_concurrency_safe=True,
)
