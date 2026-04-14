import json
import os
from typing import Any

from pydantic import Field, field_validator

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput


# ─── 入参定义 (平铺, 无自定义嵌套类型) ────────────────────────────
class MakeGraphInput(ToolInput):
    """make_graph 工具的入参。nodes 和 edges 为 dict 数组。"""

    file_name: str = Field(
        ...,
        description="保存的 json 文件名 (不含 .json 后缀), 例如 'tech_stack_graph'",
    )
    nodes: list[dict] = Field(
        ...,
        description=(
            '节点数组。每个节点包含 id, label, group 三个字段。'
            '格式: [{"id": "Vue", "label": "Vue", "group": "前端框架"}, '
            '{"id": "Vite", "label": "Vite", "group": "构建工具"}]'
        ),
    )
    edges: list[dict] = Field(
        ...,
        description=(
            '边数组。每条边包含 source, target, label 三个字段。'
            '格式: [{"source": "Vite", "target": "Vue", "label": "构建工具"}, '
            '{"source": "Vue", "target": "Pinia", "label": "状态管理"}]'
        ),
    )

    @field_validator("nodes", "edges", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list:
        """兜底: 万一 LLM 传了 JSON 字符串, 也能解析"""
        if isinstance(v, str):
            return json.loads(v)
        return v


# ─── 出参定义 ─────────────────────────────────────────────────────
class MakeGraphOutput(ToolOutput):
    """make_graph 工具的出参。"""

    saved_path: str = Field(default="", description="JSON 文件的保存路径")
    node_count: int = Field(default=0, description="保存的节点数量")
    edge_count: int = Field(default=0, description="保存的边数量")


# ─── 内部: 将平铺格式重组为前端需要的嵌套结构 ──────────────────────
def _assemble_graph(raw_nodes: list, raw_edges: list) -> dict:
    """
    LLM 输入:  {"id": "Vue", "label": "Vue", "group": "框架"}
    前端需要:  {"id": "Vue", "data": {"label": "Vue", "group": "框架"}}

    validator 已将 str/list 统一为 list, 这里只做结构重组。
    """

    # 重组 nodes: 把 label/group 塞进 data 字段
    assembled_nodes = []
    for n in raw_nodes:
        assembled_nodes.append({
            "id": n["id"],
            "data": {
                "label": n.get("label", n["id"]),
                "group": n.get("group", "default"),
            },
        })

    # 重组 edges: 把 label 塞进 data 字段
    assembled_edges = []
    for e in raw_edges:
        assembled_edges.append({
            "source": e["source"],
            "target": e["target"],
            "data": {
                "label": e.get("label", ""),
            },
        })

    return {"nodes": assembled_nodes, "edges": assembled_edges}


# ─── 执行函数 ─────────────────────────────────────────────────────
async def handle_make_graph(
    params: MakeGraphInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> MakeGraphOutput:
    """
    1. 解析 LLM 传来的平铺 JSON 字符串
    2. 重组为前端 GraphView 需要的嵌套结构
    3. 保存到 data/graphs/
    """

    # 重组为前端结构
    try:
        graph_data = _assemble_graph(params.nodes, params.edges)
    except (KeyError, TypeError) as e:
        return MakeGraphOutput(
            success=False,
            message=f"图数据重组失败: {type(e).__name__}: {e}。请确保每个 node 有 id/label/group, 每条 edge 有 source/target。",
            saved_path="",
        )

    # 保存
    base_dir = os.path.join(os.getcwd(), "data", "graphs")
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{params.file_name}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        return MakeGraphOutput(
            success=True,
            message=f"成功生成力导向图并保存至 {file_path}",
            data={"file_path": file_path},
            saved_path=file_path,
            node_count=len(graph_data["nodes"]),
            edge_count=len(graph_data["edges"]),
        )
    except Exception as e:
        return MakeGraphOutput(
            success=False,
            message=f"保存图数据失败: {type(e).__name__}: {e}",
            saved_path="",
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="make_graph",
    description=(
        "将你想要表达的复杂关系、实体及层级结构，组装成标准的节点(nodes)和边(edges)的力导向图 JSON 格式并保存到 workspace。"
        "如果用户要求展示某个生态链、关系网、或者技术图谱，使用这个工具将它们结构化输出。"
        "生成后将可以直接在前端视图中以高大上的结构图可视化显示。"
    ),
    input_schema=MakeGraphInput,
    output_schema=MakeGraphOutput,
    handler=handle_make_graph,
    tags=["graph", "visualization", "data"],
    timeout=30.0,
    is_concurrency_safe=True,
)
