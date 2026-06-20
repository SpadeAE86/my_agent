from infra.logging.logger import logger as log
from test.ticket_demo import buy_ticket_main_flow

async def run_buy_ticket_job(ticket_id: int):
    log.info(f"[Scheduler Job] Starting automatic ticket buying for ticket_id={ticket_id}")
    try:
        success = await buy_ticket_main_flow(ticket_id)
        log.info(f"[Scheduler Job] Finished automatic ticket buying. Success: {success}")
    except Exception as e:
        log.error(f"[Scheduler Job] Failed to run ticket buying job: {str(e)}", exc_info=True)
