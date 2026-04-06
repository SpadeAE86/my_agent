# infra/scheduler/jobs/memory_summary.py — 定时记忆压缩任务
# 触发: Cron (如每天凌晨 3:00)
# 职责:
#   1. 扫描所有用户的中期记忆 (logs/YYYY/MM/*.md)
#   2. 调用 core.memory.summarizer 进行深度压缩
#   3. 将压缩结果合并到 MEMORY.md (增量更新)
#   4. 记录压缩日志 (处理了哪些日期、消耗了多少 token)
