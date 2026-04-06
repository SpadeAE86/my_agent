# infra/stream/sse.py — SSE StreamingResponse 封装
# 职责:
#   1. 将 async generator 包装为 FastAPI 的 StreamingResponse
#   2. 设置正确的 Content-Type: text/event-stream
#   3. 处理客户端断开连接 (graceful shutdown)
#   4. 心跳保活 (防止代理/负载均衡器超时断开)
