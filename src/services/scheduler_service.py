import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlmodel import select, desc
from sqlalchemy import func
from models.sqlmodel.scheduler_job import SchedulerJob, SchedulerJobLog
from infra.storage.mysql_connector import mysql_connector
from infra.logging.logger import logger as log

# Import actual background tasks
from infra.scheduler.jobs.memory_summary import run_daily_memory_consolidation_and_dream
from infra.scheduler.jobs.cleanup import run_cleanup
from infra.scheduler.jobs.buy_ticket import run_buy_ticket_job
from infra.scheduler.jobs.get_showcase import run_get_showcase_job

JOB_TYPE_MAP = {
    "memory_summary": run_daily_memory_consolidation_and_dream,
    "cleanup": run_cleanup,
    "buy_ticket": run_buy_ticket_job,
    "get_showcase": run_get_showcase_job
}

JOB_TYPE_INFO = {
    "memory_summary": {
        "name": "每日记忆压缩归档",
        "description": "整理短期对话 Session 日志并合并为长期记忆（MEMORY.md）。",
        "default_args": "[]"
    },
    "cleanup": {
        "name": "临时文件与缓存清理",
        "description": "清理沙箱生成的临时文件以及清理已过期的数据文件缓存。",
        "default_args": "[]"
    },
    "buy_ticket": {
        "name": "B站自动抢票任务",
        "description": "自动查询目标票务并在有票时发起自动订单购买流程。",
        "default_args": "[1002142]"
    },
    "get_showcase": {
        "name": "B站漫展列表同步",
        "description": "同步最新的漫展展览数据至后端，提供给前端做为资讯卡片展示。",
        "default_args": "[1, 20]"
    }
}

async def get_all_jobs_from_db() -> List[SchedulerJob]:
    """获取所有数据库中的定时任务配置"""
    async with mysql_connector.session_scope() as session:
        statement = select(SchedulerJob)
        results = await session.execute(statement)
        return list(results.scalars().all())

async def get_job_from_db_by_id(job_id: str) -> Optional[SchedulerJob]:
    """根据 job_id 获取定时任务配置"""
    async with mysql_connector.session_scope() as session:
        statement = select(SchedulerJob).where(SchedulerJob.job_id == job_id)
        results = await session.execute(statement)
        return results.scalar_one_or_none()

async def create_job_in_db(job: SchedulerJob) -> SchedulerJob:
    """在数据库中创建定时任务"""
    async with mysql_connector.session_scope() as session:
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

async def update_job_in_db(job_id: str, updates: Dict[str, Any]) -> Optional[SchedulerJob]:
    """更新数据库中的定时任务"""
    async with mysql_connector.session_scope() as session:
        statement = select(SchedulerJob).where(SchedulerJob.job_id == job_id)
        results = await session.execute(statement)
        db_job = results.scalar_one_or_none()
        if not db_job:
            return None
        
        for key, val in updates.items():
            if hasattr(db_job, key):
                setattr(db_job, key, val)
                
        session.add(db_job)
        await session.commit()
        await session.refresh(db_job)
        return db_job

async def delete_job_from_db(job_id: str) -> bool:
    """删除数据库中的定时任务"""
    async with mysql_connector.session_scope() as session:
        statement = select(SchedulerJob).where(SchedulerJob.job_id == job_id)
        results = await session.execute(statement)
        db_job = results.scalar_one_or_none()
        if not db_job:
            return False
        await session.delete(db_job)
        await session.commit()
        return True

async def get_job_logs(page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """获取分页的定时任务执行日志"""
    async with mysql_connector.session_scope() as session:
        # Get total count
        count_stmt = select(func.count()).select_from(SchedulerJobLog)
        count_res = await session.execute(count_stmt)
        total = count_res.scalar() or 0
        
        # Get logs
        offset = (page - 1) * limit
        stmt = select(SchedulerJobLog).order_by(desc(SchedulerJobLog.start_time)).offset(offset).limit(limit)
        results = await session.execute(stmt)
        logs = list(results.scalars().all())
        
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "logs": logs
        }

async def add_job_log(
    job_id: str,
    name: str,
    job_type: str,
    status: str,
    scheduled_run_time: Optional[datetime] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    duration_ms: Optional[float] = None,
    error_message: Optional[str] = None
) -> SchedulerJobLog:
    """写入一条任务执行记录"""
    log_entry = SchedulerJobLog(
        job_id=job_id,
        name=name,
        job_type=job_type,
        status=status,
        scheduled_run_time=scheduled_run_time,
        start_time=start_time or datetime.now(),
        end_time=end_time,
        duration_ms=duration_ms,
        error_message=error_message
    )
    async with mysql_connector.session_scope() as session:
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return log_entry

async def update_job_log(
    log_id: int,
    status: str,
    end_time: datetime,
    duration_ms: float,
    error_message: Optional[str] = None
) -> None:
    """更新一条已有的任务执行记录状态"""
    async with mysql_connector.session_scope() as session:
        log_entry = await session.get(SchedulerJobLog, log_id)
        if log_entry:
            log_entry.status = status
            log_entry.end_time = end_time
            log_entry.duration_ms = duration_ms
            log_entry.error_message = error_message
            session.add(log_entry)
            await session.commit()

async def seed_default_jobs_if_empty() -> None:
    """如果定时任务表为空，则注入默认的系统级定时任务"""
    async with mysql_connector.session_scope() as session:
        statement = select(SchedulerJob)
        results = await session.execute(statement)
        if len(results.scalars().all()) > 0:
            log.info("scheduler_jobs table is not empty, skipping seeding.")
            return

        log.info("scheduler_jobs table is empty. Seeding default system jobs...")
        default_jobs = [
            SchedulerJob(
                job_id="daily_memory_summary",
                name="每日记忆压缩归档",
                job_type="memory_summary",
                trigger_type="cron",
                schedule_expr="0 0 * * *",
                args_json="[]",
                is_enabled=True
            ),
            SchedulerJob(
                job_id="cleanup_job",
                name="系统临时文件清理",
                job_type="cleanup",
                trigger_type="cron",
                schedule_expr="0 2 * * 0",
                args_json="[]",
                is_enabled=True
            ),
            SchedulerJob(
                job_id="showcase_sync",
                name="B站漫展资讯每日同步",
                job_type="get_showcase",
                trigger_type="cron",
                schedule_expr="0 9 * * *",
                args_json="[1, 20]",
                is_enabled=True
            ),
            SchedulerJob(
                job_id="buy_ticket_real_job",
                name="B站自动抢票(正式)",
                job_type="buy_ticket",
                trigger_type="date",
                schedule_expr="2026-06-20 12:00:00",
                args_json="[1002142]",
                is_enabled=True
            )
        ]
        for job in default_jobs:
            session.add(job)
        await session.commit()
        log.info("Default system jobs successfully seeded to database.")

