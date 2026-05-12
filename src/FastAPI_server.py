# FastAPI_server.py — 统一服务入口
# 职责:
#   1. 初始化 FastAPI 应用实例
#   2. 注册所有 routers (chat, agent, memory, task)
#   3. 挂载中间件 (CORS, 日志, 异常处理)
#   4. 启动时初始化 infra 层 (scheduler, mq, cache)
#   5. 关闭时优雅释放资源
import uvicorn, asyncio, os, json, contextlib
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import *
from infra.logging.logger import logger as log
from services.analysis_video import get_embedding_model
# from utils.obs_utils import *

from config.config import *
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from exceptions.infra import ServiceException
# init connectors and tables
from infra.connector_loader import connector_loader
from infra.storage.sqlmodel_init import create_tables_if_not_exists
# from database import *
# from core.health_monitor.lifespan import start_health_monitor, stop_health_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_task: asyncio.Task[None] | None = None
    # --- 环境预设 ---
    # 使用国内 HF 镜像加速模型下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 开启加速下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    log.info("FastAPI started")
    # 不要用 loop.set_default_executor 替换 Uvicorn/asyncio 的默认线程池：
    # 在 Windows 上曾出现「Application startup complete 后仍像完全收不到 HTTP」的现象，
    # 可能与默认执行器被替换后部分 IO/回调无法调度有关。向量模型改由独立池加载（见下）。

    try:
        # Initialize infra connectors (mysql/redis/rabbitmq/opensearch)
        await connector_loader.startup()
        # Create SQLModel tables if missing
        await create_tables_if_not_exists()

        # # --- 模型预热（后台）：须先 yield 后才开始接 HTTP；原先在 yield 前 await 会卡住整条事件循环，
        # # 导致 /health、/docs 在预热完成前一律无响应（本地常需 30–90s，看起来像「服务挂死」）。
        # async def _warmup_embedding() -> None:
        #     log.info("开始预热向量模型 (SentenceTransformer)，后台任务...")
        #     try:
        #         loop = asyncio.get_running_loop()
        #         with ThreadPoolExecutor(max_workers=1, thread_name_prefix="st_embed") as pool:
        #             await loop.run_in_executor(pool, get_embedding_model)
        #         log.info("向量模型预热完成。")
        #     except Exception as e:
        #         log.error(f"向量模型预热失败: {e}")
        #
        # warmup_task = asyncio.create_task(_warmup_embedding())
        # log.info("Lifespan 核心初始化完成，即将对外接受 HTTP（向量模型仍在后台加载）。")
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup_task
        # 停止健康监控服务
        try:
            await connector_loader.shutdown()
        except Exception as e:
            log.warning(f"connector shutdown failed: {e}")
        log.info("shutting down...")
        log.info("exit")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _log_incoming_http(request: Request, call_next):
    """确认 ASGI 层是否收到请求（与 Uvicorn access log 互补）。"""
    log.info("http in  %s %s", request.method, request.url.path)
    resp: Response = await call_next(request)
    log.info("http out %s %s -> %s", request.method, request.url.path, resp.status_code)
    return resp


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in all_router:
    app.include_router(r)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# 全局兜底异常处理
@app.exception_handler(ServiceException)
async def business_exception_handler(request: Request, exc: ServiceException):
    log.info(f"[Service Exception] {exc.code}: {exc.message}, extra info: {exc.data}")
    return JSONResponse(
        status_code=200,  # 可以统一返回 200，code 自定义区分错误类型
        content={
            "code": exc.code,
            "message": exc.message,
            "data": exc.data,
        },
    )


if __name__ == "__main__":
    uvicorn.run("FastAPI_server:app", host="0.0.0.0", port=8001)
