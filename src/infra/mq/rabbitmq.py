# infra/mq/rabbitmq.py — RabbitMQ 实现
# 职责:
#   1. 实现 BaseMQ 接口, 底层使用 pika / aio-pika
#   2. 连接管理: 自动重连、心跳保活
#   3. 消息持久化配置
#   4. 可选: 本地开发可用内存队列替代
