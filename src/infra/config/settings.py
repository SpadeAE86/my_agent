# infra/config/settings.py — 应用配置
# 基于 Pydantic BaseSettings, 自动从 .env 文件和环境变量加载
# 配置项分组:
#   AppSettings      — 应用名、版本、调试模式
#   LLMSettings      — API Key、模型名、温度、max_tokens
#   MemorySettings   — 数据目录路径、记忆 TTL、压缩阈值
#   CacheSettings    — Redis URL、本地缓存大小
#   MQSettings       — RabbitMQ 连接串
#   SchedulerSettings— 心跳间隔、队列每次处理数、token 预算
