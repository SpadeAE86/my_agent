from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json

from infra.logging.logger import logger as log
from models.sqlmodel.scheduler_job import SchedulerJob
from services.scheduler_service import (
    get_all_jobs_from_db,
    get_job_from_db_by_id,
    create_job_in_db,
    update_job_in_db,
    delete_job_from_db,
    get_job_logs,
    JOB_TYPE_INFO
)
from infra.scheduler.scheduler import scheduler_manager

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

class JobCreateBody(BaseModel):
    job_id: str = Field(..., description="唯一的任务ID标识")
    name: str = Field(..., description="任务展示名称")
    job_type: str = Field(..., description="任务类型，如 buy_ticket, cleanup")
    trigger_type: str = Field(..., description="触发类型: cron, date, interval")
    schedule_expr: str = Field(..., description="调度表达式（cron表达式、时间戳或间隔秒数）")
    args_json: Optional[str] = Field("[]", description="JSON格式的参数数组")
    is_enabled: Optional[bool] = Field(True, description="是否启用任务")

class JobUpdateBody(BaseModel):
    name: Optional[str] = None
    job_type: Optional[str] = None
    trigger_type: Optional[str] = None
    schedule_expr: Optional[str] = None
    args_json: Optional[str] = None
    is_enabled: Optional[bool] = None

@router.get("/jobs")
async def list_jobs():
    """获取所有调度任务列表，合并了运行时的下一次执行时间"""
    try:
        db_jobs = await get_all_jobs_from_db()
        jobs_payload = []
        for db_job in db_jobs:
            next_run = None
            active_job = scheduler_manager._scheduler.get_job(db_job.job_id)
            if active_job and active_job.next_run_time:
                next_run = active_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
                
            job_dict = db_job.model_dump()
            job_dict["next_run_time"] = next_run
            job_dict["is_active"] = active_job is not None
            jobs_payload.append(job_dict)
        return {"success": True, "data": jobs_payload}
    except Exception as e:
        log.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs")
async def create_job(body: JobCreateBody):
    """创建定时任务配置，并注册进调度器"""
    try:
        # Check if already exists
        existing = await get_job_from_db_by_id(body.job_id)
        if existing:
            raise HTTPException(status_code=400, detail=f"Job ID '{body.job_id}' already exists.")
            
        # Verify args is valid JSON
        if body.args_json:
            try:
                json.loads(body.args_json)
            except Exception:
                raise HTTPException(status_code=400, detail="args_json must be a valid JSON array string.")
                
        new_job = SchedulerJob(
            job_id=body.job_id,
            name=body.name,
            job_type=body.job_type,
            trigger_type=body.trigger_type,
            schedule_expr=body.schedule_expr,
            args_json=body.args_json,
            is_enabled=body.is_enabled if body.is_enabled is not None else True
        )
        created = await create_job_in_db(new_job)
        
        # Sync to active scheduler
        scheduler_manager.sync_job_to_scheduler(created)
        
        return {"success": True, "data": created.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to create job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/jobs/{job_id}")
async def update_job(job_id: str, body: JobUpdateBody):
    """更新定时任务配置，并重新同步到调度器"""
    try:
        # Check if exists
        existing = await get_job_from_db_by_id(job_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        # Verify args is valid JSON
        if body.args_json:
            try:
                json.loads(body.args_json)
            except Exception:
                raise HTTPException(status_code=400, detail="args_json must be a valid JSON array string.")
                
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        updated = await update_job_in_db(job_id, updates)
        
        if updated:
            # Sync change to active scheduler
            scheduler_manager.sync_job_to_scheduler(updated)
            return {"success": True, "data": updated.model_dump()}
        return {"success": False, "message": "Failed to update"}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to update job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """从数据库和调度器中删除该定时任务"""
    try:
        success = await delete_job_from_db(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found in database.")
            
        # Remove from running scheduler
        scheduler_manager.unschedule_job(job_id)
        return {"success": True, "message": "Job deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to delete job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/{job_id}/run")
async def trigger_job_once(job_id: str):
    """立即手动执行一次目标任务，不影响其现有的调度配置"""
    try:
        await scheduler_manager.run_job_once_now(job_id)
        return {"success": True, "message": f"Job '{job_id}' triggered successfully."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Failed to trigger job once: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def list_logs(
    page: int = Query(default=1, ge=1, description="页码"),
    limit: int = Query(default=20, ge=1, le=100, description="每页限制数量")
):
    """获取分页的调度历史执行日志"""
    try:
        data = await get_job_logs(page=page, limit=limit)
        return {"success": True, "data": data}
    except Exception as e:
        log.error(f"Failed to list logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upcoming")
async def list_upcoming():
    """获取未来7天的所有定时任务规划执行时间线，用于日历排程展示"""
    try:
        db_jobs = await get_all_jobs_from_db()
        upcoming_events = []
        for db_job in db_jobs:
            if not db_job.is_enabled:
                continue
            run_times = scheduler_manager.get_upcoming_run_times(db_job.job_id)
            for run_time in run_times:
                upcoming_events.append({
                    "job_id": db_job.job_id,
                    "name": db_job.name,
                    "job_type": db_job.job_type,
                    "run_time": run_time
                })
        # Sort chronologically
        upcoming_events.sort(key=lambda e: e["run_time"])
        return {"success": True, "data": upcoming_events}
    except Exception as e:
        log.error(f"Failed to fetch upcoming tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job-types")
async def get_job_types():
    """获取系统支持调度的任务类型元数据"""
    return {"success": True, "data": JOB_TYPE_INFO}
