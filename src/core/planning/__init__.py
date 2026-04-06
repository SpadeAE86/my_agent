# core/planning/ — 任务规划与调度
# 子模块:
#   planner          — 将用户需求拆解为结构化的 plan.md
#   task_dispatcher   — 将 plan 中的步骤转为 task.json 并分发
#   execution_graph   — 构建任务依赖 DAG, 识别可并行的任务组
#   plan_validator    — 校验 plan 的完整性、可行性与依赖闭环
