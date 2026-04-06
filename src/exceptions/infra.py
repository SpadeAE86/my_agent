# exceptions/infra.py — 基础设施异常
# CacheError      — 缓存读写异常 (Redis 连接失败等)
# MQError         — 消息队列异常 (RabbitMQ 连接失败等)
# StorageError    — 文件/Diskcache 存储异常
# ConfigError     — 配置缺失或格式错误
# SchedulerError  — 定时任务注册/执行异常
from fastapi import HTTPException

class ServiceException(HTTPException):
    def __init__(self, code: int = 400, message: str = "业务异常", data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(status_code=400, detail=message)