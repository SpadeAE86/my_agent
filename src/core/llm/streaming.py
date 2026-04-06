# core/llm/streaming.py — 流式响应处理
# 职责:
#   1. 将不同 Provider 的 stream chunk 统一为内部格式
#   2. 实时解析 delta content 和 tool_call delta
#   3. 边流式输出边累积完整响应 (用于记忆存储)
#   4. Token 计数 (input_tokens, output_tokens)
