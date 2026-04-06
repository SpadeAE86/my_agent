# core/planning/task_dispatcher.py — 任务调度与分发
# 职责:
#   1. 根据 execution_graph 的拓扑排序确定执行顺序
#   2. 为每个任务节点生成 task.json (目标、约束、工具列表、上下文)
#   3. 调用 agent_service 创建并启动子 Agent
#   4. 监听子 Agent 完成事件, 触发后续依赖任务
#   5. 无依赖的任务并行派发
