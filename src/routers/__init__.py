# routers/ — API 路由层
# 职责: 定义 HTTP 端点, 参数校验, 调用 services 层
# 不包含业务逻辑, 仅做请求转发与响应格式化
# 子模块: chat, agent, memory, task, image, prompt_template
from routers.stack import stack_router as stack_router
from routers.chat import chat_router as chat_router
from routers.image import image_router as image_router
from routers.management.prompt_template import prompt_router as prompt_router
from routers.video import video_router as video_router
from routers.video_analysis import video_analysis_router as video_analysis_router
from routers.script_match import script_match_router as script_match_router
from routers.video_match import video_match_router as video_match_router
from routers.video_mix import video_mix_router as video_mix_router
from routers.workspace import workspace_router as workspace_router
from routers.collections import collections_router as collections_router
from routers.scheduler import router as scheduler_router
from routers.agent import router as agent_router
from routers.audio import audio_router as audio_router
all_router = [stack_router, chat_router, image_router, prompt_router, video_router, video_analysis_router, script_match_router, video_match_router, video_mix_router, workspace_router, collections_router, scheduler_router, agent_router, audio_router]