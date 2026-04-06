# core/memory/short_term.py — 短期记忆 (JSONL)
# 存储路径:
#   主 Agent: data/memory/{user_id}/{session_id}.jsonl
#   子 Agent: data/memory/{user_id}/{session_id}/subagents/{agent_id}.jsonl
# 职责:
#   1. 追加写入每一轮对话 (role, content, timestamp, tool_calls)
#   2. 按 session_id 读取完整对话历史
#   3. 通过 diskcache 保证并发写入的原子性
#   4. 提供滑动窗口截断 (超出 token 限制时丢弃最早轮次)
