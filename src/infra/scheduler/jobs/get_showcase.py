from infra.logging.logger import logger as log
from services.ticket_services.bilibili_ticket_service import get_showcase_main_flow

async def run_get_showcase_job(page: int = 1, pagesize: int = 20):
    log.info(f"[Scheduler Job] Starting Bilibili showcase sync (page={page}, pagesize={pagesize})")
    try:
        shows = await get_showcase_main_flow(page=page, pagesize=pagesize)
        log.info(f"[Scheduler Job] Bilibili showcase sync completed. Fetched {len(shows)} items.")
    except Exception as e:
        log.error(f"[Scheduler Job] Bilibili showcase sync failed: {str(e)}", exc_info=True)
