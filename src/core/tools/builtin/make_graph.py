import json
import os
from typing import Any, List, Dict

from pydantic import Field, BaseModel

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput

# ─── 数据结构定义 ─────────────────────────────────────────────────────
class NodeData(BaseModel):
    label: str = Field(..., description="节点的展示名称")
    group: str = Field(..., description="节点所属的分类组（用于颜色区分等）")

class GraphNode(BaseModel):
    id: str = Field(..., description="节点的维一标识符ID")
    data: NodeData

class EdgeData(BaseModel):
    label: str = Field(default="", description="边上的文本标签")

class GraphEdge(BaseModel):
    source: str = Field(..., description="起始节点的ID")
    target: str = Field(..., description="目标节点的ID")
    data: EdgeData = Field(default_factory=EdgeData)


# ─── 入参定义 ─────────────────────────────────────────────────────
class MakeGraphInput(ToolInput):
    """make_graph 工具的入参。定义图的节点和边。"""
    
    file_name: str = Field(
        ...,
        description="保存的 json 文件的名称 (不需要 .json 后缀)，例如 'tech_stack_graph'",
    )
    nodes: List[GraphNode] = Field(
        ...,
        description="图的节点列表",
    )
    edges: List[GraphEdge] = Field(
        ...,
        description="图的边(关系)列表",
    )


# ─── 出参定义 ─────────────────────────────────────────────────────
class MakeGraphOutput(ToolOutput):
    """make_graph 工具的出参。"""
    
    saved_path: str = Field(default="", description="JSON 文件的保存路径")
    node_count: int = Field(default=0, description="保存的节点数量")
    edge_count: int = Field(default=0, description="保存的边数量")


# ─── 执行函数 ─────────────────────────────────────────────────────
async def handle_make_graph(
    params: MakeGraphInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> MakeGraphOutput:
    """
    接收图的节点和边，将其保存为统一的 JSON 格式，供前端 GraphView 等渲染。
    """
    
    # 构造标准图数据
    graph_data = {
        "nodes": [node.model_dump() for node in params.nodes],
        "edges": [edge.model_dump() for edge in params.edges],
    }
    
    # 确定保存路径 (保存到当前用户的 workspace/data 目录下)
    # 暂时默认保存到项目根目录的 data/graphs/ 下
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
            node_count=len(params.nodes),
            edge_count=len(params.edges),
        )
    except Exception as e:
        return MakeGraphOutput(
            success=False,
            message=f"保存图数据失败: {type(e).__name__}: {e}",
            saved_path="",
            node_count=0,
            edge_count=0,
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
