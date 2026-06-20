from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_match import VideoMatchJob
from services.taskboard_services.http_request_trace_service import http_request_trace_service
from services.video_match_services.video_match_service import (
    create_job_and_parse,
    get_job_payload,
    get_material_match_board_detail,
    get_shot_match_detail,
    list_material_match_histories,
    list_video_match_jobs,
    rematch_video_match_shot,
    run_job_search,
    run_video_match_retry_background,
    schedule_video_match_job_retry,
    synthesize_shot_obs_audio,
)
from services.video_compose_services import start_mix_compose_for_job


class VideoMatchCreateJobBody(BaseModel):
    script: str = Field(..., description="口播脚本")
    topic: Optional[str] = None
    title: Optional[str] = None
    car_model: Optional[str] = None
    frame_size: Optional[str] = Field(
        default=None,
        description="画面比例约束，与索引 frame_size 一致：横版16:9 / 竖版9:16；写入每镜检索标签 must",
    )
    frame_orientation: Optional[str] = Field(
        default=None,
        description="横竖屏约束，与索引 frame_orientation 一致：横屏 / 竖屏；可不选具体比例；写入每镜 tags_json",
    )
    workspace: Optional[str] = Field(default="v1", description="与视频分析 workspace 对齐")
    mock: bool = Field(default=False, description="true 时返回固定分镜，不写库")


class VideoMatchSearchBody(BaseModel):
    strategy_name: str = Field(..., min_length=1, description="与 video_analysis_search_strategy.name 一致")
    mode: str = Field(default="field_aligned_hybrid", description="match_script_tags_segments mode")
    top_k: int = Field(default=5, ge=1, le=50)
    enable_road_run_fallback: bool = Field(default=True, description="开启路跑兜底")


class MixComposeFromJobBody(BaseModel):
    mock: Optional[bool] = Field(
        default=None,
        description="true=仅占位写入 mix 表；false=POST 真实混剪；null=使用 config mix_compose.mock",
    )
    prefer_srt: bool = Field(
        default=False,
        description="true 时请求体不含 cap_config，由服务端生成 SRT，轮询完成后见 result_srt_text",
    )


video_match_router = APIRouter(prefix="/video-match", tags=["video-match"])


@video_match_router.get("/material-matches")
async def list_material_matches_route(
    workspace: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    ids: Optional[str] = None,
    limit: int = 100,
):
    return await list_material_match_histories(
        workspace=workspace, source=source, status=status, ids=ids, limit=limit
    )


@video_match_router.get("/material-matches/{match_id}/detail")
async def get_material_match_board_detail_route(match_id: str):
    detail = await get_material_match_board_detail(match_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="material match not found")
    return {"success": True, "detail": detail}


@video_match_router.get("/jobs")
async def list_video_match_jobs_route(
    parse_status: Optional[str] = None,
    workspace: Optional[str] = None,
    ids: Optional[str] = None,
    limit: int = 50,
):
    return await list_video_match_jobs(parse_status=parse_status, workspace=workspace, ids=ids, limit=limit)


@video_match_router.post("/jobs")
async def create_video_match_job(body: VideoMatchCreateJobBody, background_tasks: BackgroundTasks):
    return await create_job_and_parse(
        script=body.script,
        topic=body.topic,
        title=body.title,
        car_model=body.car_model,
        frame_size=body.frame_size,
        frame_orientation=body.frame_orientation,
        workspace=body.workspace,
        mock=body.mock,
        background_tasks=background_tasks,
    )


@video_match_router.get("/jobs/{job_id}/detail")
async def get_video_match_job_board_detail(job_id: str):
    """任务看板：整 job 口播转写 / 解析阶段的 HTTP 详情（联表 request_id）。"""
    from services.video_match_services.task_detail_service import build_video_match_job_task_detail, merge_http_trace_into_detail

    jid = (job_id or "").strip()
    if not jid:
        raise HTTPException(status_code=400, detail="invalid job_id")
    async with mysql_connector.session_scope() as session:
        job = await session.get(VideoMatchJob, jid)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    row = {
        "id": job.id,
        "script": job.script,
        "topic": job.topic,
        "title": job.title,
        "car_model": job.car_model,
        "frame_size": job.frame_size,
        "frame_orientation": job.frame_orientation,
        "workspace": job.workspace,
        "parse_status": job.parse_status,
        "parse_error": job.parse_error,
        "search_status": job.search_status,
        "search_error": job.search_error,
        "request_id": job.request_id,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    base = build_video_match_job_task_detail(row)
    rid = (job.request_id or "").strip()
    trace_dict = await http_request_trace_service.get_dict(str(rid)) if rid else None
    return {"success": True, "detail": merge_http_trace_into_detail(base, trace_dict)}


@video_match_router.get("/jobs/{job_id}")
async def get_video_match_job(job_id: str, verbose: int = Query(0, ge=0, le=1)):
    """
    verbose=1 时在服务日志中打印每条分镜的 top1 / 命中数 / 首条 hit 字段，便于排查「匹配成功但 URL 空」。
    """
    data = await get_job_payload(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="job not found")
    if verbose:
        for s in data.get("shots") or []:
            if not isinstance(s, dict):
                continue
            hits = s.get("match_top_hits_json")
            fk: list[str] = []
            if isinstance(hits, list) and hits and isinstance(hits[0], dict):
                fk = list(hits[0].keys())
            log.info(
                "video_match_services GET job={} shot_order={} search={} top1_non_empty={} hit_count={} first_hit_keys={}",
                job_id,
                s.get("shot_order"),
                s.get("search_status"),
                bool(str(s.get("top1_obs_url") or "").strip()),
                s.get("match_hit_count"),
                fk,
            )
    return data


@video_match_router.post("/jobs/{job_id}/extract")
async def extract_tags_for_job_route(job_id: str, background_tasks: BackgroundTasks):
    """批量提取标签 (Stage 2)"""
    from services.video_match_services.video_match_service import run_job_extract_tags
    background_tasks.add_task(run_job_extract_tags, job_id)
    return {"success": True, "message": "extracting tags in background"}


@video_match_router.post("/jobs/{job_id}/retry")
async def retry_video_match_job(job_id: str, background_tasks: BackgroundTasks):
    """失败任务重试：转写失败则重新解析；仅检索失败则依赖 job 内记录的检索策略重新跑匹配（需曾成功发起过匹配）。"""
    body = await schedule_video_match_job_retry(job_id)
    if not body.get("success"):
        raise HTTPException(status_code=400, detail=str(body.get("error") or "retry not allowed"))
    kind = str(body.get("kind") or "").strip()
    strat = body.get("strategy_name")
    strategy_name = str(strat).strip() if strat else None
    background_tasks.add_task(run_video_match_retry_background, job_id, kind, strategy_name)
    return {"success": True, "retry": kind}



@video_match_router.post("/jobs/{job_id}/shots/{shot_row_id}/extract")
async def extract_tags_for_shot_route(job_id: str, shot_row_id: int, background_tasks: BackgroundTasks):
    """单条分镜提取标签"""
    from services.video_match_services.video_match_service import run_shot_extract_tags
    background_tasks.add_task(run_shot_extract_tags, job_id, shot_row_id)
    return {"success": True, "message": "extracting shot tags in background"}

@video_match_router.post("/jobs/{job_id}/shots/{shot_row_id}/rematch")
async def rematch_video_match_shot_route(job_id: str, shot_row_id: int):
    """对单条分镜重新执行素材检索（与整 job 匹配共用策略快照）。"""
    return await rematch_video_match_shot(job_id, shot_row_id)

@video_match_router.put("/jobs/{job_id}/shots/{shot_row_id}/tokens")
async def update_shot_tokens_route(job_id: str, shot_row_id: int, body: dict = Body(...)):
    """更新分镜的结构化搜索标签（AND/OR/NOT 节点）"""
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise HTTPException(status_code=400, detail="tokens must be a list")
    from services.video_match_services.video_match_service import update_shot_tokens
    return await update_shot_tokens(job_id, shot_row_id, tokens)


@video_match_router.put("/jobs/{job_id}/shots/{shot_row_id}/top1")
async def update_shot_top1_route(job_id: str, shot_row_id: int, body: dict = Body(...)):
    """手动切换/更新分镜的 Top1 视频"""
    top1_obs_url = body.get("top1_obs_url")
    if not isinstance(top1_obs_url, str):
        raise HTTPException(status_code=400, detail="top1_obs_url must be a string")
    from services.video_match_services.video_match_service import update_shot_top1_url
    return await update_shot_top1_url(job_id, shot_row_id, top1_obs_url)


@video_match_router.get("/jobs/{job_id}/shots/{shot_row_id}/detail")
async def get_video_match_shot_detail(job_id: str, shot_row_id: int):
    data = await get_shot_match_detail(job_id, shot_row_id)
    if data is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return data


@video_match_router.post("/jobs/{job_id}/shots/{shot_row_id}/audio")
async def synthesize_shot_audio_route(job_id: str, shot_row_id: int):
    return await synthesize_shot_obs_audio(job_id, shot_row_id)


@video_match_router.post("/jobs/{job_id}/mix-compose")
async def start_mix_compose_from_match_job(
    job_id: str,
    body: MixComposeFromJobBody = Body(default_factory=MixComposeFromJobBody),
):
    """与 ``POST /video-mix/compose`` 等价，仅路径绑定在匹配 job 上。可选 body: ``{ \"mock\": true|false|null }``。"""
    try:
        return await start_mix_compose_for_job(job_id, mix_mock=body.mock, prefer_srt=body.prefer_srt)
    except ValueError as e:
        msg = str(e)
        if msg == "job not found":
            raise HTTPException(status_code=404, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e


@video_match_router.post("/jobs/{job_id}/search")
async def search_video_match_job(job_id: str, body: VideoMatchSearchBody):
    return await run_job_search(
        job_id,
        strategy_name=body.strategy_name,
        mode=body.mode,
        top_k=body.top_k,
        enable_road_run_fallback=body.enable_road_run_fallback,
    )
