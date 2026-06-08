# infra/scheduler/scheduler.py — AsyncIOScheduler Wrapper
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from infra.logging.logger import logger as log

class AsyncIOSchedulerManager:
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._scheduler.start()
            self._is_running = True
            log.info("AsyncIOScheduler started successfully")

    def shutdown(self):
        if self._is_running:
            self._scheduler.shutdown()
            self._is_running = False
            log.info("AsyncIOScheduler shutdown successfully")

    def add_cron_job(self, func, job_id: str, hour: int = 0, minute: int = 0, args: list = None):
        """Adds a daily cron job at the specified hour and minute."""
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

# Singleton instance
scheduler_manager = AsyncIOSchedulerManager()
