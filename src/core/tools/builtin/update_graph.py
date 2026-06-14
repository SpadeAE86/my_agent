import json
import os
from typing import Any

from pydantic import Field, field_validator

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput


class UpdateGraphInput(ToolInput):
    """update_graph 工具的入参。"""

    file_name: str = Field(
        ...,
        description="要编辑更新的力导图 ID (即文件名，不含 .json 后缀)，例如 'tech_stack_graph'",
    )
    nodes: list[dict] = Field(
        ...,
        description=(
            '更新后的节点数组。每个节点包含 id, label, group（必填），以及可选的 x, y 坐标。'
            '填写 x/y 后前端支持"原始结构"静态渲染模式（适合时序图/层级图）。'
            '格式: [{"id": "Vue", "label": "Vue", "group": "前端框架", "x": 200, "y": 300}]'
        ),
    )
    edges: list[dict] = Field(
        ...,
        description=(
            '更新后的边数组。每条边包含 source, target, label 三个字段。'
            '格式: [{"source": "Vite", "target": "Vue", "label": "构建工具"}]'
        ),
    )

    @field_validator("nodes", "edges", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list:
        if isinstance(v, str):
            return json.loads(v)
        return v


class UpdateGraphOutput(ToolOutput):
    """update_graph 工具的出参。"""

    saved_path: str = Field(default="", description="JSON 文件的保存路径")
    node_count: int = Field(default=0, description="保存的节点数量")
    edge_count: int = Field(default=0, description="保存的边数量")
    graph_name: str = Field(default="", description="力导图的名称")
    graph_id: str = Field(default="", description="力导图的 ID / 文件名")
    nodes: list[dict] = Field(default_factory=list, description="组装后的嵌套节点数据")
    edges: list[dict] = Field(default_factory=list, description="组装后的嵌套连线数据")


async def handle_update_graph(
    params: UpdateGraphInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> UpdateGraphOutput:
    """
    1. 解析 LLM 传来的更新后的扁平 JSON 结构
    2. 重组为前端 G6 需要的嵌套 data 格式
    3. 覆盖写入至 data/graphs/{file_name}.json
    """
    # 重组为前端结构
    try:
        from core.tools.builtin.make_graph import _assemble_graph
        graph_data = _assemble_graph(params.nodes, params.edges)
    except (KeyError, TypeError) as e:
        return UpdateGraphOutput(
            success=False,
            message=f"图数据重组失败: {type(e).__name__}: {e}。请确保每个 node 有 id/label/group, 每条 edge 有 source/target。",
            saved_path="",
        )

    # 保存/覆写
    from pathlib import Path
    current = Path(__file__).resolve()
    while current.name != "src" and current.parent != current:
        current = current.parent
    project_root = current.parent
    base_dir = os.path.join(project_root, "data", "graphs")
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{params.file_name}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        return UpdateGraphOutput(
            success=True,
            message=f"成功更新力导图并保存至 {file_path}",
            data={"file_path": file_path, "graph_id": params.file_name, "graph_name": params.file_name},
            saved_path=file_path,
            node_count=len(graph_data["nodes"]),
            edge_count=len(graph_data["edges"]),
            graph_name=params.file_name,
            graph_id=params.file_name,
            nodes=graph_data["nodes"],
            edges=graph_data["edges"]
        )
    except Exception as e:
        return UpdateGraphOutput(
            success=False,
            message=f"保存更新后的力导图失败: {type(e).__name__}: {e}",
            saved_path="",
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="update_graph",
    description=(
        "覆盖写入或更新指定的力导图 ID。传入扁平的节点(nodes)和边(edges)结构，工具会自动组装为前端所需格式并存储。\n"
        "节点可携带可选的 x/y 坐标（参考 make_graph 的坐标系说明），填写后前端支持'原始结构'静态模式预览。"
    ),
    input_schema=UpdateGraphInput,
    output_schema=UpdateGraphOutput,
    handler=handle_update_graph,
    tags=["graph", "visualization", "data"],
    timeout=30.0,
    is_concurrency_safe=True,
)
