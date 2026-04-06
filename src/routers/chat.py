# routers/chat.py — 基础对话路由
# 端点:
#   POST /chat          — 发送消息, 获取回复 (支持流式 SSE)
#   GET  /chat/history  — 获取当前会话的对话历史
# 依赖: services.chat_service
