import json
import os
from typing import Any

from pydantic import Field, field_validator

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput


# ─── 坐标系说明（供 AI 参考）─────────────────────────────────────────
# 画布坐标系: x 向右为正，y 向下为正，原点在画布左上角
# 建议画布尺寸: 宽 1000px, 高 700px (中心约 500, 350)
# 布局建议:
#   左→右时序/流程图: 每列 x 间距 160-200, y 保持 300-400
#   上→下层级图:      每层 y 间距 120-150, x 居中对齐同层节点
#   环形/自由关系图:  不填 x/y，交给力导布局自动摆放
#   示例（3步时序图）: step1 (100,350) → step2 (300,350) → step3 (500,350)


# ─── 入参定义 ─────────────────────────────────────────────────────────
class MakeGraphInput(ToolInput):
    """make_graph 工具的入参。nodes 和 edges 为 dict 数组。"""

    file_name: str = Field(
        ...,
        description="保存的 json 文件名 (不含 .json 后缀), 例如 'tech_stack_graph'",
    )
    nodes: list[dict] = Field(
        ...,
        description=(
            '节点数组。每个节点包含 id, label, group 三个必填字段，以及可选的 x, y 初始坐标。\n'
            '• id/label/group: 必填\n'
            '• x, y: 可选。填写后前端支持"原始结构"模式，按坐标静态渲染（适合时序图/层级图）。\n'
            '  不填则由力导布局自动摆放。\n'
            '坐标系: x 向右，y 向下，建议范围 x: 50-950, y: 50-650\n'
            '  左→右时序图示例: 每步 x += 180, y 固定在 300\n'
            '  上→下层级图示例: 每层 y += 140, 同层节点均匀分配 x\n'
            '格式: [{"id": "A", "label": "开始", "group": "流程", "x": 100, "y": 300}, '
            '{"id": "B", "label": "处理", "group": "流程", "x": 300, "y": 300}]'
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
    graph_name: str = Field(default="", description="力导图的名称")
    graph_id: str = Field(default="", description="力导图的 ID / 文件名")
    nodes: list[dict] = Field(default_factory=list, description="组装后的节点数据")
    edges: list[dict] = Field(default_factory=list, description="组装后的连线数据")
    has_positions: bool = Field(default=False, description="节点是否包含自定义坐标（支持原始结构模式）")


# ─── 内部: 将平铺格式重组为前端需要的嵌套结构 ──────────────────────
def _assemble_graph(raw_nodes: list, raw_edges: list) -> dict:
    """
    LLM 输入:  {"id": "Vue", "label": "Vue", "group": "框架", "x": 100, "y": 200}
    前端需要:  {"id": "Vue", "data": {"label": "Vue", "group": "框架"}, "x": 100, "y": 200}

    x, y 是可选字段，存在时保留在节点顶层（G6 preset layout 直接读取）。
    """

    assembled_nodes = []
    has_positions = False
    for n in raw_nodes:
        node: dict = {
            "id": n["id"],
            "data": {
                "label": n.get("label", n["id"]),
                "group": n.get("group", "default"),
            },
        }
        # 保留可选坐标：G6 preset layout 从节点顶层读 x/y
        if "x" in n and n["x"] is not None:
            node["x"] = float(n["x"])
            has_positions = True
        if "y" in n and n["y"] is not None:
            node["y"] = float(n["y"])
        assembled_nodes.append(node)

    assembled_edges = []
    for e in raw_edges:
        assembled_edges.append({
            "source": e["source"],
            "target": e["target"],
            "data": {
                "label": e.get("label", ""),
            },
        })

    return {"nodes": assembled_nodes, "edges": assembled_edges, "has_positions": has_positions}


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
    2. 重组为前端 GraphView 需要的嵌套结构（保留 x/y 坐标）
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

        has_positions = graph_data.get("has_positions", False)
        msg = f"成功生成图并保存至 {file_path}。"
        if has_positions:
            msg += "节点含自定义坐标，前端支持'原始结构'静态模式。"
        else:
            msg += "节点无坐标，将使用力导布局。"

        return MakeGraphOutput(
            success=True,
            message=msg,
            data={"file_path": file_path, "graph_id": params.file_name, "graph_name": params.file_name},
            saved_path=file_path,
            node_count=len(graph_data["nodes"]),
            edge_count=len(graph_data["edges"]),
            graph_name=params.file_name,
            graph_id=params.file_name,
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
            has_positions=has_positions,
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
        "将你想要表达的复杂关系、实体及层级结构，组装成节点(nodes)和边(edges)并保存到 workspace，"
        "可在前端力导图视图中可视化。\n\n"
        "【布局模式选择指引】\n"
        "• 关系网/生态图（节点关系是网状的）→ 不填 x/y，前端用力导布局自动收束，分组着色\n"
        "• 时序图/流程图（从左到右线性推进）→ 填写 x/y，AI 按 x 递增排列，前端支持'原始结构'静态预览\n"
        "• 层级图/树形图（从上到下父子关系）→ 填写 x/y，AI 按 y 递增分层，前端支持'原始结构'静态预览\n\n"
        "【坐标填写规则（当你决定填 x/y 时）】\n"
        "• 画布宽约 1000，高约 700，坐标原点在左上角\n"
        "• 左→右时序图: 每步 x += 160~200，y 统一取 300~350\n"
        "• 上→下层级图: 每层 y += 130~160，同层节点均匀分配 x（例如 3 个节点分配到 200/500/800）\n"
        "• 混合分支图: 主干节点按主方向排，分支节点在垂直方向偏移 ±120\n"
        "• 不需要精确像素，AI 给出合理的相对间距即可，前端会自动缩放适配视口"
    ),
    input_schema=MakeGraphInput,
    output_schema=MakeGraphOutput,
    handler=handle_make_graph,
    tags=["graph", "visualization", "data"],
    timeout=30.0,
    is_concurrency_safe=True,
)
