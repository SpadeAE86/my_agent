# infra/scheduler/scheduler.py — AsyncIOScheduler Wrapper with Database Sync & Event Logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler import events
from datetime import datetime
import time
import json
from typing import Dict, Any, Optional

from infra.logging.logger import logger as log

class AsyncIOSchedulerManager:
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._is_running = False
        self._job_start_times = {}  # key: (job_id, scheduled_run_time), value: (log_id, start_time)

    def start(self):
        if not self._is_running:
            self._setup_event_listeners()
            self._scheduler.start()
            self._is_running = True
            log.info("AsyncIOScheduler started successfully")

    def shutdown(self):
        if self._is_running:
            self._scheduler.shutdown()
            self._is_running = False
            log.info("AsyncIOScheduler shutdown successfully")

    def _setup_event_listeners(self):
        # We listen for EVENT_JOB_SUBMITTED to log 'running' state and capture start_time,
        # and EVENT_JOB_EXECUTED / EVENT_JOB_ERROR for final outcomes.
        self._scheduler.add_listener(
            self._handle_scheduler_event,
            events.EVENT_JOB_SUBMITTED | events.EVENT_JOB_EXECUTED | events.EVENT_JOB_ERROR
        )
        self._job_start_times = {}
        log.info("AsyncIOScheduler event listeners configured")

    async def _handle_scheduler_event(self, event):
        # Extract base job_id (handling immediate executions with temporary once ids)
        base_job_id = event.job_id
        is_once_run = "_once_" in base_job_id
        if is_once_run:
            base_job_id = base_job_id.split("_once_")[0]

        # Ignore standard system jobs that we don't track in DB log, or fallback names
        if base_job_id in ("daily_memory_summary",):
            name = "每日记忆压缩归档"
            job_type = "memory_summary"
        else:
            from services.scheduler_service import get_job_from_db_by_id
            db_job = await get_job_from_db_by_id(base_job_id)
            if db_job:
                name = db_job.name
                job_type = db_job.job_type
            else:
                name = base_job_id
                job_type = "unknown"

        # Unique key to track this specific execution run
        key = (event.job_id, str(event.scheduled_run_time))

        try:
            if event.code == events.EVENT_JOB_SUBMITTED:
                # 1. Job started running
                start_time = datetime.now()
                from services.scheduler_service import add_job_log
                log_entry = await add_job_log(
                    job_id=base_job_id,
                    name=name,
                    job_type=job_type,
                    status="running",
                    scheduled_run_time=event.scheduled_run_time,
                    start_time=start_time
                )
                self._job_start_times[key] = (log_entry.id, start_time)
                
            elif event.code in (events.EVENT_JOB_EXECUTED, events.EVENT_JOB_ERROR):
                # 2. Job finished or failed
                log_info = self._job_start_times.pop(key, None)
                end_time = datetime.now()
                duration_ms = 0.0
                log_id = None
                
                if log_info:
                    log_id, start_time = log_info
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                
                status = "success" if event.code == events.EVENT_JOB_EXECUTED else "failed"
                error_message = None
                if event.code == events.EVENT_JOB_ERROR:
                    error_message = str(event.exception) if event.exception else "Unknown execution error"
                
                if log_id:
                    from services.scheduler_service import update_job_log
                    await update_job_log(
                        log_id=log_id,
                        status=status,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        error_message=error_message
                    )
                else:
                    # Fallback to creating a log if we missed the SUBMITTED event
                    from services.scheduler_service import add_job_log
                    await add_job_log(
                        job_id=base_job_id,
                        name=name,
                        job_type=job_type,
                        status=status,
                        scheduled_run_time=event.scheduled_run_time,
                        start_time=end_time,
                        end_time=end_time,
                        duration_ms=0.0,
                        error_message=error_message
                    )
        except Exception as e:
            log.error(f"Error handling scheduler event: {e}", exc_info=True)

    def add_cron_job(self, func, job_id: str, hour: int = 0, minute: int = 0, args: list = None):
        """Adds a daily cron job at the specified hour and minute (Static/default jobs)."""
        self._scheduler.add_job(
            func,
            trigger="cron",
            hour=hour,
            minute=minute,
            id=job_id,
            args=args or [],
            replace_existing=True
        )
        log.info(f"Scheduled daily cron job '{job_id}' at {hour:02d}:{minute:02d}")

    def add_date_job(self, func, job_id: str, run_date, args: list = None):
        """Adds a one-shot job to run at a specific date/time (datetime object or ISO string)."""
        self._scheduler.add_job(
            func,
            trigger="date",
            run_date=run_date,
            id=job_id,
            args=args or [],
            replace_existing=True
        )
        log.info(f"Scheduled one-shot job '{job_id}' at {run_date}")

    def _add_or_update_active_job(self, func, job_id: str, trigger_type: str, schedule_expr: str, args: list = None):
        """Helper to add or update an active job inside the APScheduler instance."""
        trigger_args = {}
        if trigger_type == "cron":
            parts = schedule_expr.strip().split()
            if len(parts) == 5:
                trigger_args = {
                    "minute": parts[0],
                    "hour": parts[1],
                    "day": parts[2],
                    "month": parts[3],
                    "day_of_week": parts[4]
                }
            elif ":" in schedule_expr:
                time_parts = schedule_expr.split(":")
                trigger_args = {
                    "hour": time_parts[0],
                    "minute": time_parts[1]
                }
            else:
                trigger_args = {"hour": schedule_expr}
        elif trigger_type == "date":
            # Can parse standard ISO string
            trigger_args = {"run_date": schedule_expr}
        elif trigger_type == "interval":
            try:
                trigger_args = {"seconds": int(schedule_expr)}
            except ValueError:
                trigger_args = {"minutes": 15}  # default fallback

        self._scheduler.add_job(
            func,
            trigger=trigger_type,
            id=job_id,
            args=args or [],
            replace_existing=True,
            **trigger_args
        )
        log.info(f"Active scheduler job '{job_id}' synchronized. Trigger={trigger_type}, expr='{schedule_expr}'")

    async def init_jobs(self):
        """Initialize and sync all enabled jobs from database on startup."""
        from services.scheduler_service import get_all_jobs_from_db, JOB_TYPE_MAP
        
        log.info("AsyncIOScheduler: Syncing jobs from database...")
        db_jobs = await get_all_jobs_from_db()
        db_job_ids = set()
        
        for db_job in db_jobs:
            if not db_job.is_enabled:
                continue
                
            db_job_ids.add(db_job.job_id)
            func = JOB_TYPE_MAP.get(db_job.job_type)
            if not func:
                log.warning(f"Unknown job type '{db_job.job_type}' for job '{db_job.job_id}', skipping.")
                continue
                
            args = []
            if db_job.args_json:
                try:
                    args = json.loads(db_job.args_json)
                except Exception as e:
                    log.error(f"Failed to parse args for job '{db_job.job_id}': {e}")
                    
            try:
                self._add_or_update_active_job(
                    func=func,
                    job_id=db_job.job_id,
                    trigger_type=db_job.trigger_type,
                    schedule_expr=db_job.schedule_expr,
                    args=args
                )
            except Exception as e:
                log.error(f"Failed to schedule job '{db_job.job_id}' on startup: {e}")

        # Cleanup active jobs that are no longer enabled/exist in DB
        # Keep internal daily consolidation job
        for active_job in self._scheduler.get_jobs():
            if active_job.id not in db_job_ids and active_job.id not in (
                "daily_memory_summary",
                "buy_ticket_test_job",
                "buy_ticket_real_job"
            ) and not active_job.id.endswith("_once"):
                self._scheduler.remove_job(active_job.id)
                log.info(f"Removed active scheduler job '{active_job.id}' (not active in DB)")

    def sync_job_to_scheduler(self, db_job):
        """Syncs a single database job configuration to the running scheduler."""
        from services.scheduler_service import JOB_TYPE_MAP
        
        if not db_job.is_enabled:
            if self._scheduler.get_job(db_job.job_id):
                self._scheduler.remove_job(db_job.job_id)
                log.info(f"Removed disabled job '{db_job.job_id}' from active scheduler")
            return
            
        func = JOB_TYPE_MAP.get(db_job.job_type)
        if not func:
            log.warning(f"Unknown job type '{db_job.job_type}' for job '{db_job.job_id}'")
            return
            
        args = []
        if db_job.args_json:
            try:
                args = json.loads(db_job.args_json)
            except Exception as e:
                log.error(f"Failed to parse args for job '{db_job.job_id}': {e}")
                
        self._add_or_update_active_job(
            func=func,
            job_id=db_job.job_id,
            trigger_type=db_job.trigger_type,
            schedule_expr=db_job.schedule_expr,
            args=args
        )

    def unschedule_job(self, job_id: str):
        """Removes a job from the active scheduler."""
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            log.info(f"Unscheduled job '{job_id}' from active scheduler")

    async def run_job_once_now(self, job_id: str):
        """Triggers a job to run immediately once without modifying its schedule."""
        from services.scheduler_service import get_job_from_db_by_id, JOB_TYPE_MAP
        
        db_job = await get_job_from_db_by_id(job_id)
        if not db_job:
            raise ValueError(f"Job '{job_id}' not found in database.")
            
        func = JOB_TYPE_MAP.get(db_job.job_type)
        if not func:
            raise ValueError(f"Function mapping not found for job type '{db_job.job_type}'")
            
        args = []
        if db_job.args_json:
            try:
                args = json.loads(db_job.args_json)
            except Exception:
                pass
                
        # Run immediately by adding a one-shot job with trigger='date' running 'now'
        temp_id = f"{job_id}_once_{int(time.time())}"
        self._scheduler.add_job(
            func,
            trigger="date",
            run_date=datetime.now(),
            id=temp_id,
            args=args
        )
        log.info(f"Triggered immediate execution of job '{job_id}' (temp_id={temp_id})")

    def get_upcoming_run_times(self, job_id: str, limit: int = 50) -> list[str]:
        """Calculates upcoming fire times for a specific job in the next 7 days."""
        from datetime import datetime, timedelta
        
        active_job = self._scheduler.get_job(job_id)
        if not active_job:
            return []
            
        trigger = active_job.trigger
        tz = getattr(trigger, "timezone", None)
        
        # now_param for trigger: aware if tz exists, else naive
        now_param = datetime.now(tz) if tz else datetime.now()
        
        # naive local datetimes for timezone-safe comparison
        now_local = datetime.now()
        end_time_local = now_local + timedelta(days=7)
        
        def make_naive_local(dt):
            if dt is None:
                return None
            if dt.tzinfo is not None:
                return dt.astimezone().replace(tzinfo=None)
            return dt

        runs = []
        try:
            current_fire = trigger.get_next_fire_time(None, now_param)
            while current_fire and len(runs) < limit:
                current_fire_local = make_naive_local(current_fire)
                if current_fire_local > end_time_local:
                    break
                    
                runs.append(current_fire_local.strftime("%Y-%m-%d %H:%M:%S"))
                
                # Compute next check (preserving timezone-awareness of current_fire)
                next_check = current_fire + timedelta(seconds=1)
                current_fire = trigger.get_next_fire_time(current_fire, next_check)
        except Exception as e:
            log.error(f"Error calculating upcoming fire times for job {job_id}: {e}", exc_info=True)
            
        return runs

# Singleton instance
scheduler_manager = AsyncIOSchedulerManager()
