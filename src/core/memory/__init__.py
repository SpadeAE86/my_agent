# core/memory/ — 记忆系统
# 三层记忆架构: 短期 (JSONL) → 中期 (Daily MD) → 长期 (MEMORY.md)
# 子模块:
#   memory_manager — 统一调度: 根据查询类型选择合适的记忆层
#   short_term     — JSONL 读写: 实时追加对话轮次
#   mid_term       — Markdown 日志: 会话结束后提炼关键信息
#   long_term      — 压缩记忆: Cron 任务周期性总结中期记忆
#   summarizer     — LLM 驱动的摘要生成器
#   retriever      — 基于关键词/语义的记忆检索 (RAG)
