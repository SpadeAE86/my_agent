# services/stream_service.py — SSE 流式响应封装
# 职责:
#   1. 将 core.llm 的流式输出转为 SSE (Server-Sent Events) 格式
#   2. 在流中插入工具调用事件、状态变更事件
#   3. 处理流的中断与重连机制
#   4. 提供统一的 StreamingResponse 工厂方法
