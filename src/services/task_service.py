# services/task_service.py — 任务下发与聚合
# 职责:
#   1. 将主 Agent 产出的 plan.md 解析为结构化任务列表
#   2. 用户确认后, 将任务拆分为 task.json 并分发给子 Agent
#   3. 收集子 Agent 的执行结果, 更新任务状态
#   4. 所有子任务完成后, 汇总结果返回给主 Agent
