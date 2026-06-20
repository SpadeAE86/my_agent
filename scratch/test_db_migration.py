import asyncio
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from infra.connector_loader import connector_loader
from infra.storage.sqlmodel_init import create_tables_if_not_exists
from services.scheduler_service import get_all_jobs_from_db, add_job_log, get_job_logs

async def test_db():
    print("Connecting to database...")
    await connector_loader.startup()
    
    print("Running create_tables_if_not_exists...")
    await create_tables_if_not_exists()
    
    print("Verifying SchedulerJob table query...")
    jobs = await get_all_jobs_from_db()
    print(f"Query successful. Found {len(jobs)} jobs in database.")
    
    print("Verifying SchedulerJobLog table query...")
    logs_data = await get_job_logs(page=1, limit=5)
    print(f"Query successful. Logs count in DB: {logs_data['total']}")
    
    print("Shutting down connectors...")
    await connector_loader.shutdown()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(test_db())
