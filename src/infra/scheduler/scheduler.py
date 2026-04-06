# infra/scheduler/scheduler.py — APScheduler 封装
# 职责:
#   1. 封装 APScheduler, 提供 add_job / remove_job / pause_job 接口
#   2. 支持 Cron 表达式 (如: 每天凌晨 3 点执行记忆压缩)
#   3. 支持 Interval 触发 (如: 每 30 分钟心跳)
#   4. 持久化 job store (重启后恢复已注册的任务)
#   5. 与 FastAPI 生命周期集成 (startup/shutdown)
