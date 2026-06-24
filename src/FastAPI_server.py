# FastAPI_server.py — 统一服务入口
# 职责:
#   1. 初始化 FastAPI 应用实例
#   2. 注册所有 routers (chat, agent, memory, task)
#   3. 挂载中间件 (CORS, 日志, 异常处理)
#   4. 启动时初始化 infra 层 (scheduler, mq, cache)
#   5. 关闭时优雅释放资源
import uvicorn, asyncio, contextlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import *
from infra.logging.logger import logger as log
from services.video_match_services.analysis_video import start_embedding_warmup_background
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

        # Create SQLModel tables if missing
        await create_tables_if_not_exists()

        # Seed Volcano & Qwen voice timbres if missing
        try:
            from services.volcovoice_service import VolcoVoiceService
            await VolcoVoiceService.seed_default_timbres_if_empty()
            await VolcoVoiceService.seed_volcano_timbres_if_missing()
            await VolcoVoiceService.seed_qwen_timbres_if_missing()
        except Exception as _seeder_err:
            log.warning(f"Failed to seed voice timbres: {_seeder_err}")

        # Seed default database-configured jobs if missing
        from services.scheduler_service import seed_default_jobs_if_empty
        await seed_default_jobs_if_empty()


        # Start background job scheduler
        from infra.scheduler.scheduler import scheduler_manager
        scheduler_manager.start()
        
        # Load and synchronize all database-configured jobs
        await scheduler_manager.init_jobs()

        try:
            from services.taskboard_services.interrupted_tasks_recovery import mark_interrupted_tasks_on_startup

            await mark_interrupted_tasks_on_startup("服务重启或进程中断，任务未完成")
        except Exception as _e:
            log.warning("启动时标记中断任务失败（可忽略若表未就绪）: {}", _e)

        try:
            from services.tts_services.voice_tts_service import voice_tts_service
            asyncio.create_task(voice_tts_service.heal_pending_tasks_on_startup())
            log.info("已向后台派发 TTS 任务启动时修复任务。")
        except Exception as _e:
            log.warning("启动时派发 TTS 任务修复失败: {}", _e)

        try:
            from services.tts_services.voice_clone_service import voice_clone_service
            async def heal_qwen_voice_demos():
                await asyncio.sleep(5)  # 等待系统组件及数据库完全就绪
                log.info("开始扫描并自动补全缺失演示音频的阿里云自定义音色...")
                from sqlmodel import select
                from models.sqlmodel.voice_cloned import VoiceCloned
                from models.sqlmodel.voice_designed import VoiceDesigned
                from infra.storage.mysql_connector import mysql_connector
                
                async with mysql_connector.session_scope() as session:
                    cloned_res = await session.execute(
                        select(VoiceCloned)
                        .where(VoiceCloned.provider == "qwen")
                        .where(VoiceCloned.status != 3)
                        .where((VoiceCloned.demo_audio_url == None) | (VoiceCloned.demo_audio_url == ""))
                    )
                    cloned_voices = cloned_res.scalars().all()
                    
                    designed_res = await session.execute(
                        select(VoiceDesigned)
                        .where(VoiceDesigned.provider == "qwen")
                        .where(VoiceDesigned.status != 3)
                        .where((VoiceDesigned.demo_audio_url == None) | (VoiceDesigned.demo_audio_url == ""))
                    )
                    designed_voices = designed_res.scalars().all()
                    
                for v in cloned_voices:
                    log.info("自动补全 Qwen 克隆音色 demo 预览音频: {} ({})", v.voice_character, v.custom_speaker_id)
                    await voice_clone_service._generate_qwen_demo_audio(v.id, v.custom_speaker_id)
                    
                for v in designed_voices:
                    log.info("自动补全 Qwen 设计音色 demo 预览音频: {} ({})", v.voice_character, v.custom_speaker_id)
                    await voice_clone_service._generate_qwen_design_demo_audio(v.id, v.custom_speaker_id)
                    
                log.info("阿里云自定义音色缺失演示音频自动补全任务处理完毕。")

            asyncio.create_task(heal_qwen_voice_demos())
        except Exception as _heal_err:
            log.warning("启动时派发阿里云演示音频补全任务失败: %s", _heal_err)

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
                    from services.video_match_services.frame_orientation_os_backfill import run_frame_orientation_backfill

                    n = await run_frame_orientation_backfill()
                    log.info(f"OpenSearch frame_orientation 回填完成，更新文档数: {n}")
                except Exception as _fo:
                    log.warning(
                        f"frame_orientation 回填未执行或失败（可稍后手动: python -m services.frame_orientation_os_backfill）: {_fo}"
                    )

            asyncio.create_task(_frame_orientation_backfill_bg())

        # 同步数据库已有标签到自组织标签库
        async def _sync_tags_bg() -> None:
            try:
                from routers.collections import sync_existing_tags_to_library
                await sync_existing_tags_to_library()
            except Exception as _sync_err:
                log.warning(f"标签库后台自动同步失败: {_sync_err}")

        asyncio.create_task(_sync_tags_bg())

        # 定时每 15 分钟将标签库同步到数据库镜像表
        async def _sync_library_to_db_loop() -> None:
            try:
                from routers.collections import start_tag_library_db_sync_task
                await start_tag_library_db_sync_task()
            except Exception as _sync_err:
                log.warning(f"标签库同步写入数据库后台循环异常: {_sync_err}")

        asyncio.create_task(_sync_library_to_db_loop())

        # 定时每 10 分钟同步火山和阿里云音色状态
        async def _sync_custom_voices_loop() -> None:
            # 启动后先等待 15 秒，避免和启动初始化阶段冲突，并给网络和DB连接充足的时间
            await asyncio.sleep(15)
            while True:
                try:
                    from services.tts_services.voice_clone_service import voice_clone_service
                    log.info("定时任务启动：开始同步火山引擎和阿里云自定义音色状态...")
                    await asyncio.gather(
                        voice_clone_service.sync_volcano_voices(),
                        voice_clone_service.sync_qwen_voices(),
                        return_exceptions=True
                    )
                    log.info("定时任务：自定义音色状态同步完成")
                except Exception as _sync_err:
                    log.warning(f"定时同步自定义音色状态异常: {_sync_err}")
                await asyncio.sleep(600)

        asyncio.create_task(_sync_custom_voices_loop())



        # 启动时后台预热所有已配置的角色音色
        async def _prewarm_all_roles_bg() -> None:
            try:
                # 稍微等待 6 秒，给数据库和连接就绪时间
                await asyncio.sleep(6)
                from core.roles.role_manager import role_manager
                from services.tts_services.voice_tts_service import voice_tts_service
                
                roles = role_manager.list_roles()
                for r in roles:
                    if r.get("voice_configured"):
                        voice_char = r.get("voice_character")
                        if voice_char:
                            log.info(f"启动后台预热角色 '{r.get('name')}' 的音色 '{voice_char}'...")
                            await voice_tts_service.warm_up_voice(voice_char)
                log.info("启动时角色音色后台预热完毕。")
            except Exception as _pe:
                log.warning(f"启动后台预热角色音色异常: {_pe}")

        asyncio.create_task(_prewarm_all_roles_bg())



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
