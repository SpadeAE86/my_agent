# core/memory/mid_term.py — 中期记忆 (Daily Markdown)
# 存储路径: data/memory/{user_id}/logs/YYYY/MM/YYYY-MM-DD.md
# 职责:
#   1. 会话结束时, 调用 summarizer 提炼当次对话的关键结论
#   2. 按日期追加到对应的 Markdown 文件中
#   3. 每条记录包含: 时间戳、会话摘要、关键决策、待办事项
#   4. 支持按日期范围查询
