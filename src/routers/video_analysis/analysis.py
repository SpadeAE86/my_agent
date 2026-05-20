# routers/video_analysis.py — 视频分析路由
# 端点:
#   POST /video-analysis           — 上传视频并执行分析流水线, 返回分镜卡片列表
#   GET  /video-analysis/history   — 获取所有历史分析记录
#   POST /video-analysis/history   — 覆盖写入全部历史记录
#   POST /video-analysis/history/update — 追加/更新单条历史记录

import asyncio
import functools
import hashlib
import os
import sys
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_material_match import VideoMaterialMatchHistory

from models.pydantic.video_analysis_request import (
    HistorySaveRequest,
    HistoryUpdateRequest,
    VideoAnalysisHistoryItem,
    ShotCard as PydShotCard,
)
from models.pydantic.model_output_schema.seedtext_script_segments_schema import SeedtextIndexTagsEnvelope
from services.analysis_video import analyze_video, index_shotcards_to_opensearch
from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags
from utils.frame_orientation import infer_frame_orientation
from services.video_analysis_db_service import video_analysis_db_service
from services.video_match_service import _load_strategy_by_name
from services.http_request_trace_service import http_request_trace_service
from infra.logging.logger import logger as log
from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.query_builder import query_builder
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from models.pydantic.opensearch_index.base_index import (
    get_index_name, get_vector_fields, get_searchable_fields, get_field_weights, get_vector_weights,
)
from services.script_match_recall import ensure_hybrid_pipeline, ensure_rrf_pipeline
from services.video_match_http_trace import trace_response_top_hits_with_explain
from services.token_join_template_service import (
    TOKEN_JOIN_TERM_FIELDS_V2,
    normalize_v2_term_filter_value,
    create_template,
    delete_template,
    get_default_and_fields,
    list_templates,
    set_default_template,
    update_template,
)
from utils.search_utils import reciprocal_rank_fuse
from core.workspace import list_workspaces, DEFAULT_WORKSPACE_KEY, get_workspace


UPLOAD_TMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "video_analysis_uploads",
)

# 全局并发：超出时在后台任务内排队，配合 DB 中 PENDING → RUNNING
_video_analysis_max_concurrent = max(1, min(32, int(os.getenv("VIDEO_ANALYSIS_MAX_CONCURRENT", "3"))))
_video_analysis_slot = asyncio.BoundedSemaphore(_video_analysis_max_concurrent)

# 内存任务状态（与 DB 并存；上传/重试入口均会写入 PENDING）
video_task_status: Dict[str, Dict[str, Any]] = {}

analysis_router = APIRouter()

def _append_chunk_to_disk(f: Any, md5_hash: Any, chunk: bytes) -> None:
    """大块同步磁盘写入；须经 asyncio.to_thread 调用，避免阻塞事件循环。"""
    f.write(chunk)
    md5_hash.update(chunk)


def _parse_retry_params_from_trace_body(body: Any) -> Dict[str, Any]:
    """从首次分析的 http_request_traces.request_body 还原参数；缺省与 POST /video-analysis 表单默认一致。"""
    out: Dict[str, Any] = {
        "frame_interval": 2.0,
        "threshold": 30.0,
        "custom_prompt": None,
        "split_scenes": True,
    }
    if not isinstance(body, dict):
        return out
    fi = body.get("frame_interval")
    if isinstance(fi, (int, float)):
        out["frame_interval"] = float(fi)
    th = body.get("threshold")
    if isinstance(th, (int, float)):
        out["threshold"] = float(th)
    cp = body.get("custom_prompt")
    if isinstance(cp, str):
        t = cp.strip()
        out["custom_prompt"] = t or None
    elif cp is not None:
        out["custom_prompt"] = str(cp)
    ss = body.get("split_scenes")
    if isinstance(ss, bool):
        out["split_scenes"] = ss
    return out


async def _local_video_path_for_retry(video_url: str, file_name: str) -> str:
    """从 OBS/CDN 等 URL 拉取到临时路径，供 ``_bg_analyze_video`` 使用（结束后由该函数删除）。"""
    from utils.obs_utils import download_url_to_file

    safe_base = (file_name or "video.mp4").replace("\\", "/").split("/")[-1] or "video.mp4"
    if "." not in safe_base:
        safe_base = f"{safe_base}.mp4"
    local_path = os.path.join(UPLOAD_TMP_DIR, f"retry_{uuid.uuid4().hex[:12]}_{safe_base}")
    os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
    await download_url_to_file(str(video_url).strip(), local_path)
    return local_path


async def _bg_retry_video_analysis(
    history_id: str,
    video_url: str,
    file_name: str,
    frame_interval: float,
    threshold: float,
    custom_prompt: Optional[str],
    split_scenes: bool,
    workspace: str,
    car_model: Optional[str],
    http_trace_id: str,
) -> None:
    local_path = await _local_video_path_for_retry(video_url, file_name)
    await _bg_analyze_video(
        history_id,
        local_path,
        file_name,
        frame_interval,
        threshold,
        custom_prompt,
        split_scenes,
        workspace,
        car_model,
        video_url,
        http_trace_id,
        remove_local_after=True,
    )


@analysis_router.post("/history/{history_id}/retry")
async def retry_failed_video_analysis(history_id: str, background_tasks: BackgroundTasks):
    """仅 FAILED：同一 history id，从 ``video_url`` 拉源并重跑分析（参数优先来自首次 ``request_id`` 追踪体）。"""
    hid = (history_id or "").strip()
    if not hid:
        raise HTTPException(status_code=400, detail="invalid id")
    row = await video_analysis_db_service.get_history_row(hid)
    if row is None:
        raise HTTPException(status_code=404, detail="history not found")
    st = str(row.get("status") or "").strip().upper()
    if st != "FAILED":
        raise HTTPException(status_code=400, detail="仅失败任务可重试")
    video_url = str(row.get("video_url") or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="记录缺少视频地址，无法重试")

    workspace = str(row.get("workspace") or "v1").strip() or "v1"
    car_model_raw = row.get("car_model")
    car_model = str(car_model_raw).strip() if car_model_raw is not None else None
    if car_model == "":
        car_model = None
    file_name = str(row.get("name") or "video.mp4").strip() or "video.mp4"

    params = {
        "frame_interval": 2.0,
        "threshold": 30.0,
        "custom_prompt": None,
        "split_scenes": True,
    }
    old_rid = row.get("request_id")
    if old_rid:
        tr = await http_request_trace_service.get_dict(str(old_rid))
        if tr:
            params.update(_parse_retry_params_from_trace_body(tr.get("request_body")))

    retry_trace_id = await http_request_trace_service.create_initial(
        request_url=f"/video-analysis/history/{hid}/retry",
        http_method="POST",
        request_body={
            "retry_of": hid,
            "video_url": video_url,
            "workspace": workspace,
            "car_model": car_model,
            **params,
        },
        business_type="VIDEO_ANALYSIS",
        method_name="POST /video-analysis/history/retry",
        upstream_task_id=hid,
    )

    await video_analysis_db_service.upsert_history_item(
        {
            "id": hid,
            "name": file_name,
            "time": datetime.now().isoformat(timespec="seconds"),
            "video_url": video_url,
            "workspace": workspace,
            "status": "PENDING",
            "request_id": retry_trace_id,
            "car_model": car_model,
            "error_msg": None,
        },
        shot_cards_version=workspace,
    )
    video_task_status[hid] = {"status": "PENDING"}

    background_tasks.add_task(
        _bg_retry_video_analysis,
        hid,
        video_url,
        file_name,
        float(params["frame_interval"]),
        float(params["threshold"]),
        params.get("custom_prompt"),
        bool(params.get("split_scenes", True)),
        workspace,
        car_model,
        retry_trace_id,
    )
    return {"success": True, "task_id": hid}



class RewriteScriptRequest(BaseModel):
    script: str = Field(..., description="需要提取的口播脚本或自然语言描述")
    topic: Optional[str] = None
    title: Optional[str] = None
    car_model: Optional[str] = None
    frame_size: Optional[str] = Field(
        default=None,
        description="与索引 frame_size 一致：横版16:9 / 竖版9:16；写入每段标签并参与搜索 must",
    )
    frame_orientation: Optional[str] = Field(
        default=None,
        description="与索引 frame_orientation 一致：横屏 / 竖屏；可不指定具体比例",
    )

@analysis_router.post("/rewrite-script")
async def rewrite_script_endpoint(req: RewriteScriptRequest):
    """
    调用大模型将自然语言脚本提取为结构化的检索标签。
    直接返回 SeedtextIndexTagsEnvelope 格式的字典。
    """
    log.info(
        f"[rewrite-script] received request: script={req.script!r} topic={req.topic!r} "
        f"title={req.title!r} car_model={req.car_model!r} frame_size={req.frame_size!r} "
        f"frame_orientation={req.frame_orientation!r}"
    )
    if not req.script.strip():
        return {"success": False, "error": "script cannot be empty"}
    
    try:
        # 调用现有的 service 逻辑
        storyboard, tags = await rewrite_script_to_storyboard_and_tags(
            script=req.script,
            topic=req.topic,
            title=req.title,
            car_model=req.car_model,
            frame_size=(req.frame_size or "").strip() or None,
            frame_orientation=(req.frame_orientation or "").strip() or None,
            index=0,
        )
        tags_dump = tags.model_dump(exclude_none=True)
        for item in tags_dump.get("segment_result") or []:
            if not isinstance(item, dict):
                continue
            cm = (req.car_model or "").strip()
            if cm:
                item["car_model"] = cm
            fs = (req.frame_size or "").strip()
            if fs and fs != "未知":
                item["frame_size"] = fs
            fo = (req.frame_orientation or "").strip()
            if fo in ("横屏", "竖屏"):
                item["frame_orientation"] = fo
            elif not fo and fs:
                inf = infer_frame_orientation(fs)
                if inf and inf != "未知":
                    item["frame_orientation"] = inf
        return {"success": True, "tags": tags_dump}
    except Exception as e:
        log.error(f"rewrite_script failed: {e}")
        return {"success": False, "error": str(e)}


@analysis_router.get("/status/{task_id}")
async def get_video_status(task_id: str):
    """查询视频分析任务状态"""
    status = video_task_status.get(task_id)
    if not status:
        row = await video_analysis_db_service.get_history_row(task_id)
        if not row:
            return {"success": False, "error": "Task not found"}
        st = str(row.get("status") or "SUCCESS").upper()
        if st == "SUCCESS":
            item = await video_analysis_db_service.get_history_item(task_id)
            return {"success": True, "status": "SUCCESS", "item": item or row}
        return {"success": True, "status": st, "item": row}
    return {"success": True, **status}

async def _bg_analyze_video(
    project_id: str,
    local_path: str,
    file_name: str,
    frame_interval: float,
    threshold: float,
    custom_prompt: Optional[str],
    split_scenes: bool,
    workspace: str,
    car_model: Optional[str],
    obs_video_url: str,
    http_trace_id: str,
    *,
    remove_local_after: bool = True,
):
    """后台分析：先排队等全局并发槽，再 RUNNING 并执行 analyze_video。"""
    await _video_analysis_slot.acquire()
    t0 = time.monotonic()
    video_task_status[project_id] = {"status": "RUNNING", "progress": 0}
    await video_analysis_db_service.upsert_history_item(
        {
            "id": project_id,
            "name": file_name,
            "time": datetime.now().isoformat(timespec="seconds"),
            "video_url": obs_video_url,
            "workspace": workspace,
            "status": "RUNNING",
            "request_id": http_trace_id,
            "car_model": car_model,
        },
        shot_cards_version=workspace,
    )
    try:
        try:
            cards = await analyze_video(
                local_video_path=local_path,
                project_id=project_id,
                frame_interval=frame_interval,
                threshold=threshold,
                custom_prompt=custom_prompt,
                split_scenes=split_scenes,
                cleanup_workspace=True,
                workspace=workspace,
                car_model=car_model,
            )

            history_item = VideoAnalysisHistoryItem(
                id=project_id,
                name=file_name,
                time=datetime.now().isoformat(timespec="seconds"),
                video_url=obs_video_url,
                workspace=workspace,
                car_model=car_model,
                cards=cards,
                request_id=http_trace_id,
            )

            await video_analysis_db_service.upsert_history_item(
                {
                    **history_item.model_dump(exclude_none=True),
                    "status": "SUCCESS",
                },
                shot_cards_version=workspace,
            )

            keys_ok = [(project_id, c.scene_id) for c in cards if not c.error]
            if keys_ok:
                try:
                    await index_shotcards_to_opensearch(cards, id_prefix=project_id, refresh=True, workspace=workspace)
                    await video_analysis_db_service.update_cards_index_status(
                        keys_ok, status="OK", error=None, shot_cards_version=workspace
                    )
                    log.info(f"[{project_id}] OpenSearch 入库完成")
                except Exception as _e:
                    await video_analysis_db_service.update_cards_index_status(
                        keys_ok, status="FAILED", error=str(_e), shot_cards_version=workspace
                    )
                    log.error(f"[{project_id}] OpenSearch 入库失败: {_e}")

            final_item = await video_analysis_db_service.get_history_item(project_id)
            video_task_status[project_id] = {"status": "SUCCESS", "item": final_item or history_item.model_dump()}

            duration_ms = int((time.monotonic() - t0) * 1000)
            await http_request_trace_service.finalize(
                http_trace_id,
                status_code=200,
                response_body={
                    "status": "SUCCESS",
                    "project_id": project_id,
                    "scene_count": len(cards),
                },
                duration_ms=duration_ms,
                business_success=True,
            )

        except Exception as e:
            log.error(f"[{project_id}] 视频分析任务异常: {e}")
            video_task_status[project_id] = {"status": "FAILED", "error": str(e)}
            duration_ms = int((time.monotonic() - t0) * 1000)
            try:
                await video_analysis_db_service.upsert_history_item(
                    {
                        "id": project_id,
                        "name": file_name,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "video_url": obs_video_url,
                        "workspace": workspace,
                        "status": "FAILED",
                        "error_msg": str(e),
                        "request_id": http_trace_id,
                        "car_model": car_model,
                    },
                    shot_cards_version=workspace,
                )
            except Exception as _db_e:
                log.warning(f"persist FAILED video history: {_db_e}")
            try:
                await http_request_trace_service.finalize(
                    http_trace_id,
                    status_code=500,
                    error_message=str(e),
                    response_body={"status": "FAILED", "project_id": project_id, "error": str(e)},
                    duration_ms=duration_ms,
                    business_success=False,
                )
            except Exception as _fe:
                log.warning(f"finalize http trace: {_fe}")
    finally:
        _video_analysis_slot.release()
        if remove_local_after and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

@analysis_router.post("/")
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="待分析的视频文件"),
    frame_interval: float = Form(2.0),
    threshold: float = Form(30.0),
    custom_prompt: Optional[str] = Form(None),
    split_scenes: bool = Form(True),
    workspace: str = Form("v1"),
    car_model: Optional[str] = Form(None),
    async_mode: bool = Form(True),
):
    """接收上传视频并执行分析流水线"""
    log.info(f"[analyze_video_endpoint] received request: filename={file.filename}, async={async_mode}")
    
    # 计算 MD5
    temp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
    local_path = os.path.join(UPLOAD_TMP_DIR, f"{temp_id}_{file.filename}")
    
    try:
        md5_hash = hashlib.md5()
        with open(local_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await asyncio.to_thread(_append_chunk_to_disk, f, md5_hash, chunk)
        project_id = md5_hash.hexdigest()[:16]
    except Exception as e:
        if os.path.exists(local_path): os.remove(local_path)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 上传源视频
    from services.analysis_video import _get_or_upload_source_video
    obs_video_url = await _get_or_upload_source_video(local_path, project_id, car_model)

    va_trace_id = await http_request_trace_service.create_initial(
        request_url="/video-analysis",
        http_method="POST",
        request_body={
            "project_id": project_id,
            "filename": file.filename,
            "frame_interval": frame_interval,
            "threshold": threshold,
            "custom_prompt": custom_prompt,
            "split_scenes": split_scenes,
            "workspace": workspace,
            "car_model": car_model,
            "obs_video_url": obs_video_url,
            "async_mode": async_mode,
        },
        business_type="VIDEO_ANALYSIS",
        method_name="POST /video-analysis",
        upstream_task_id=project_id,
    )

    # 先入库 PENDING：批量提交时任务看板/角标可立即看到排队项
    await video_analysis_db_service.upsert_history_item(
        {
            "id": project_id,
            "name": file.filename or "video",
            "time": datetime.now().isoformat(timespec="seconds"),
            "video_url": obs_video_url or "",
            "workspace": workspace,
            "status": "PENDING",
            "request_id": va_trace_id,
            "car_model": car_model,
        },
        shot_cards_version=workspace,
    )
    video_task_status[project_id] = {"status": "PENDING"}

    if async_mode:
        background_tasks.add_task(
            _bg_analyze_video,
            project_id, local_path, file.filename,
            frame_interval, threshold, custom_prompt,
            split_scenes, workspace, car_model, obs_video_url,
            va_trace_id,
            remove_local_after=True,
        )
        return {"success": True, "task_id": project_id, "status": "PENDING"}
    else:
        # 兼容同步模式
        try:
            await _bg_analyze_video(
                project_id, local_path, file.filename,
                frame_interval, threshold, custom_prompt,
                split_scenes, workspace, car_model, obs_video_url,
                va_trace_id,
                remove_local_after=True,
            )
            item = await video_analysis_db_service.get_history_item(project_id)
            return {"success": True, "item": item}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

