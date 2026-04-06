# exceptions/ — 统一异常体系
# 子模块:
#   base  — 基础异常类 (AppException, 所有自定义异常的父类)
#   agent — Agent 相关异常 (AgentLoopError, MaxTurnsExceeded, PlanValidationError)
#   tool  — 工具相关异常 (ToolNotFound, ToolExecutionError, ToolTimeout)
#   infra — 基础设施异常 (CacheError, MQError, StorageError, ConfigError)
