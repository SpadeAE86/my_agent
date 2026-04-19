# routers/ — API 路由层
# 职责: 定义 HTTP 端点, 参数校验, 调用 services 层
# 不包含业务逻辑, 仅做请求转发与响应格式化
# 子模块: chat, agent, memory, task, image, prompt_template
from routers.stack import stack_router as stack_router
from routers.chat import chat_router as chat_router
from routers.image import image_router as image_router
from routers.prompt_template import prompt_router as prompt_router
from routers.video import video_router as video_router
all_router = [stack_router, chat_router, image_router, prompt_router, video_router]