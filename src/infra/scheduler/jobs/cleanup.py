# infra/scheduler/jobs/cleanup.py — 过期数据清理任务
# 触发: Cron (如每周日凌晨) 或 queue_worker 按需触发
# 职责:
#   1. 清理已完成子 Agent 的短期记忆 JSONL (超过 TTL)
#   2. 清理 diskcache 中过期的缓存条目
#   3. 清理沙箱临时工作目录
#   4. 压缩/归档过旧的中期记忆日志
