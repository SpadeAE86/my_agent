# core/planning/execution_graph.py — DAG / 并行执行图
# 职责:
#   1. 将 plan 的步骤列表构建为有向无环图 (DAG)
#   2. 自动检测步骤间的依赖关系 (输入/输出)
#   3. 识别可并行执行的独立任务组
#   4. 提供拓扑排序用于顺序调度
#   5. 支持动态更新 (运行时插入/移除节点)
# 数据结构:
#   TaskNode  — 单个任务节点 (id, deps, status, result)
#   ExecGraph — 管理所有 TaskNode 的图结构
