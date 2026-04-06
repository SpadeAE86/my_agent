# FastAPI_server.py — 统一服务入口
# 职责:
#   1. 初始化 FastAPI 应用实例
#   2. 注册所有 routers (chat, agent, memory, task)
#   3. 挂载中间件 (CORS, 日志, 异常处理)
#   4. 启动时初始化 infra 层 (scheduler, mq, cache)
#   5. 关闭时优雅释放资源
import uvicorn, asyncio, concurrent, os, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import *
from infra.logging.logger import logger as log
# from utils.obs_utils import *

from config.config import *
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from exceptions.infra import ServiceException
# from database import *
# from core.health_monitor.lifespan import start_health_monitor, stop_health_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI started")
    log.info("inject main loop")
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    default_pool = getattr(loop, "_default_executor", None)

    if default_pool:
        pool_size = getattr(default_pool, "_max_workers", "Unknown")
        log.info(f"当前默认线程池大小 (Max Workers): {pool_size}")

    else:
        log.info("无法获取默认线程池")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(executor)
    # log.info("Default thread pool executor set to max_workers=50")
    # log.info("memory loaded")
    # log.info(f"established {len(db_manager.engines)} connections to mysql database")

    try:
        yield
    finally:
        # 停止健康监控服务
        log.info("shutting down...")
        log.info("exit")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in all_router:
    app.include_router(r)


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
