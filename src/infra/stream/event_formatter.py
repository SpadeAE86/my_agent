# infra/stream/event_formatter.py — 内部事件 → SSE 格式化
# 职责:
#   1. 将 core.agent.event 中的事件对象转为 SSE 文本格式
#   2. 格式: "event: {event_type}\ndata: {json_payload}\n\n"
#   3. 支持自定义事件类型 (text_chunk, tool_call, plan_update, error...)
#   4. 处理特殊字符转义与多行 data
