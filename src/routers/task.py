# routers/task.py — 任务状态路由
# 端点:
#   GET  /task/plan          — 获取当前 plan.md 的内容与状态
#   GET  /task/list          — 列出所有进行中的子任务
#   GET  /task/{task_id}     — 查看单个任务详情及其子 Agent 状态
#   POST /task/plan/confirm  — 用户确认 plan, 触发任务拆分与下发
# 依赖: services.task_service
