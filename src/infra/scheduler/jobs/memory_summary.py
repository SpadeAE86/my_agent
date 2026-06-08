# infra/scheduler/jobs/memory_summary.py — Daily memory consolidation and dream job
import os
import asyncio
from pathlib import Path
from core.memory import mid_term
from core.memory import long_term
from infra.logging.logger import logger as log

MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "memory"

async def run_daily_memory_consolidation_and_dream() -> None:
    """
    Daily background memory consolidation job.
    1. Discovers all users under data/memory/
    2. For each user, discovers active session directories
    3. Runs session-to-daily-log consolidation (using session.md + jsonl)
    4. Runs dream consolidation (updating MEMORY.md)
    """
    log.info("Starting daily memory consolidation and dream job...")
    
    if not MEMORY_ROOT.exists():
        log.warning(f"Memory root directory {MEMORY_ROOT} does not exist. Skipping job.")
        return
        
    # Discover users
    user_ids = []
    try:
        for p in MEMORY_ROOT.iterdir():
            if p.is_dir():
                user_ids.append(p.name)
    except Exception as e:
        log.error(f"Failed to scan memory root: {e}")
        return
        
    for user_id in user_ids:
        log.info(f"Processing memory for user: {user_id}")
        user_dir = MEMORY_ROOT / user_id
        
        # Discover sessions
        session_ids = []
        try:
            for p in user_dir.iterdir():
                if p.is_dir() and p.name != "logs":
                    session_ids.append(p.name)
        except Exception as e:
            log.warning(f"Failed to scan sessions for user {user_id}: {e}")
            continue
            
        # 1. Consolidate sessions (first part of memory lifecycle)
        for session_id in session_ids:
            log.info(f"Consolidating session {session_id} for user {user_id}...")
            try:
                # pass None client and model to use summarizer defaults
                success = await mid_term.consolidate_session_to_daily_log(
                    user_id=user_id,
                    session_id=session_id,
                    client=None,
                    model=""
                )
                if success:
                    log.info(f"Session {session_id} consolidated successfully.")
                else:
                    log.debug(f"Session {session_id} had no logs to consolidate.")
            except Exception as e:
                log.error(f"Failed to consolidate session {session_id}: {e}")
                
        # 2. Dream (second part of memory lifecycle)
        log.info(f"Running dream consolidation for user {user_id}...")
        try:
            success = await long_term.dream_consolidation(
                user_id=user_id,
                client=None,
                model=""
            )
            if success:
                log.info(f"Dream consolidation completed successfully for user {user_id}.")
            else:
                log.info(f"No recent logs to dream for user {user_id}.")
        except Exception as e:
            log.error(f"Dream consolidation failed for user {user_id}: {e}")
            
    log.info("Daily memory consolidation and dream job finished.")

def run_job_sync() -> None:
    """Wrapper to run the async job synchronously in the scheduler."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(run_daily_memory_consolidation_and_dream())
    else:
        loop.run_until_complete(run_daily_memory_consolidation_and_dream())
