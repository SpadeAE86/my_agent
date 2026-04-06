# core/state/task_state.py — 任务全局状态
# 字段:
#   plan_id         — 当前 plan 的唯一标识
#   plan_status     — "draft" | "confirmed" | "executing" | "completed" | "failed"
#   tasks           — 子任务列表 [{task_id, agent_id, status, result}]
#   execution_graph — 关联的 DAG 实例引用
#   progress        — 完成百分比
