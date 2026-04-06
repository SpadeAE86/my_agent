# routers/ — API 路由层
# 职责: 定义 HTTP 端点, 参数校验, 调用 services 层
# 不包含业务逻辑, 仅做请求转发与响应格式化
# 子模块: chat, agent, memory, task
from routers.stack import stack_router as stack_router
all_router = [stack_router]