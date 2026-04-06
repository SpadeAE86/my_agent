# services/ — 业务逻辑编排层
# 职责: 组合 core 层与 infra 层能力, 为 routers 提供可复用的用例方法
# 原则: 每个 service 对应一个领域概念, 不直接操作底层存储
# 子模块: chat_service, agent_service, memory_service, task_service, stream_service
