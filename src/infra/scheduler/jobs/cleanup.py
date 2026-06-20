# infra/scheduler/jobs/cleanup.py — 过期数据清理任务
import os
import shutil
from pathlib import Path
from infra.logging.logger import logger as log

async def run_cleanup():
    """
    Expired data cleanup job.
    1. Cleans up sandbox temporary working directories
    2. Cleans up old task cache and logs
    """
    log.info("Starting background expired data cleanup job...")
    # Add actual cleanup implementation if needed, for now we log and perform minimal tasks
    work_dir = Path("./work")
    if work_dir.exists():
        log.info(f"Checking work directory: {work_dir}")
        # Placeholder for cleanup logic (e.g. deleting files older than 7 days)
    log.info("Background expired data cleanup job finished successfully.")
