# FastAPI_server.py — 统一服务入口
# 职责:
#   1. 初始化 FastAPI 应用实例
#   2. 注册所有 routers (chat, agent, memory, task)
#   3. 挂载中间件 (CORS, 日志, 异常处理)
#   4. 启动时初始化 infra 层 (scheduler, mq, cache)
#   5. 关闭时优雅释放资源
import uvicorn, asyncio, os, json, contextlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import *
from infra.logging.logger import logger as log
from services.analysis_video import start_embedding_warmup_background
# from utils.obs_utils import *

from config.config import *
from contextlib import asynccontextmanager
from exceptions.infra import ServiceException
# init connectors and tables
from infra.connector_loader import connector_loader
from infra.storage.sqlmodel_init import create_tables_if_not_exists
# from database import *
# from core.health_monitor.lifespan import start_health_monitor, stop_health_monitor


def _apply_hf_env_presets() -> None:
    """
    默认为国内镜像；本地/离线仅加载已下载模型时不要强制 HF_ENDPOINT，否则会走代理去拉 modules.json。
    参见环境变量：SENTENCE_TRANSFORMER_MODEL、HF_HUB_OFFLINE、SKIP_HF_MIRROR。
    """
    st = (os.environ.get("SENTENCE_TRANSFORMER_MODEL") or "").strip()
    local_dir = bool(st) and os.path.isdir(st)
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    skip_mirror = os.environ.get("SKIP_HF_MIRROR", "").strip().lower() in ("1", "true", "yes")
    if offline or skip_mirror or local_dir:
        if offline:
            log.info("HF_HUB_OFFLINE 已开启：不设置 HF_ENDPOINT，避免 Hugging Face Hub 网络请求")
        elif skip_mirror:
            log.info("SKIP_HF_MIRROR 已开启：不设置 HF_ENDPOINT")
        elif local_dir:
            log.info(
                "SENTENCE_TRANSFORMER_MODEL 指向本地目录 ({})：不设置 HF_ENDPOINT；请确保目录内为完整模型快照",
                st,
            )
        return
    if (os.environ.get("HF_ENDPOINT") or "").strip():
        return
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    log.info("已预设 HF_ENDPOINT=%s（仅下载场景；本地模型请设 SENTENCE_TRANSFORMER_MODEL 为目录或设 HF_HUB_OFFLINE=1）", os.environ["HF_ENDPOINT"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_task: asyncio.Task[None] | None = None
    _apply_hf_env_presets()

    log.info("FastAPI started")
    # 不要用 loop.set_default_executor 替换 Uvicorn/asyncio 的默认线程池：
    # 在 Windows 上曾出现「Application startup complete 后仍像完全收不到 HTTP」的现象，
    # 可能与默认执行器被替换后部分 IO/回调无法调度有关。向量模型改由独立池加载（见 analysis_video._EMBED_EXECUTOR）。

    try:
        # Initialize infra connectors (mysql/redis/rabbitmq/opensearch)
        await connector_loader.startup()
        
        # Start background job scheduler
        from infra.scheduler.scheduler import scheduler_manager
        from infra.scheduler.jobs.memory_summary import run_job_sync
        scheduler_manager.start()
        scheduler_manager.add_cron_job(run_job_sync, "daily_memory_summary", hour=0, minute=0)
        
        # Create SQLModel tables if missing
        await create_tables_if_not_exists()

        try:
            from services.interrupted_tasks_recovery import mark_interrupted_tasks_on_startup

            await mark_interrupted_tasks_on_startup("服务重启或进程中断，任务未完成")
        except Exception as _e:
            log.warning("启动时标记中断任务失败（可忽略若表未就绪）: %s", _e)

        if os.environ.get("SKIP_FRAME_ORIENTATION_BACKFILL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            log.info("已设置 SKIP_FRAME_ORIENTATION_BACKFILL，跳过 frame_orientation 索引回填")
        else:

            async def _frame_orientation_backfill_bg() -> None:
                try:
                    await asyncio.sleep(2)
                    from services.frame_orientation_os_backfill import run_frame_orientation_backfill

                    n = await run_frame_orientation_backfill()
                    log.info(f"OpenSearch frame_orientation 回填完成，更新文档数: {n}")
                except Exception as _fo:
                    log.warning(
                        f"frame_orientation 回填未执行或失败（可稍后手动: python -m services.frame_orientation_os_backfill）: {_fo}"
                    )

            asyncio.create_task(_frame_orientation_backfill_bg())

        # 模型预热（后台 task，不 await）：yield 后 HTTP 立即可用；OpenSearch 入库前会 await ensure_embedding_model_ready 等待同一加载任务。
        warmup_task = start_embedding_warmup_background()
        log.info("已向后台派发向量模型预热；HTTP 即将就绪（向量化入库前会等待预热完成）。")
        yield
    finally:
        # Shutdown background job scheduler
        try:
            from infra.scheduler.scheduler import scheduler_manager
            scheduler_manager.shutdown()
        except Exception as se:
            log.warning(f"scheduler shutdown failed: {se}")

        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup_task
        # 停止健康监控服务
        try:
            await connector_loader.shutdown()
        except Exception as e:
            log.warning(f"connector shutdown failed: {e}")
        try:
            from infra.storage.mix_overall_time_mysql import dispose_mix_overall_time_engine

            await dispose_mix_overall_time_engine()
        except Exception:
            pass
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
