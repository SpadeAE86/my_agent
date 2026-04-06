# infra/mq/base.py — 消息队列抽象接口
# 定义:
#   class BaseMQ(ABC):
#       publish(queue, message) -> None
#       subscribe(queue, callback) -> None
#       ack(delivery_tag) -> None
#       close() -> None
# 所有 MQ 实现必须继承此接口
