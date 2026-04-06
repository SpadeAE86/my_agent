# infra/mq/message_schema.py — 消息体 Schema
# 定义队列中流转的消息模型:
#   TaskMessage      — 后台任务消息 (task_type, payload, priority, max_tokens)
#   MemorySyncMsg    — 记忆同步消息 (user_id, sync_direction, data)
#   CleanupMsg       — 清理任务消息 (target_path, ttl_days)
