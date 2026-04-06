# core/memory/summarizer.py — LLM 驱动的摘要生成器
# 职责:
#   1. 接收原始对话记录, 生成结构化摘要
#   2. 支持多种粒度: 单轮摘要、会话摘要、跨天摘要
#   3. 提供有损压缩: 在保留关键信息的前提下控制 token 消耗
#   4. 被 mid_term 和 long_term 模块调用
