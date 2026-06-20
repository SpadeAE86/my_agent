import asyncio
import sys
import os
import json

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from infra.connector_loader import connector_loader
from infra.storage.sqlmodel_init import create_tables_if_not_exists
from services.ticket_services.bilibili_ticket_service import get_show_list, add_buyer, list_buyers

async def test():
    # 1. Start resources and ensure table is created
    await connector_loader.startup()
    await create_tables_if_not_exists()
    
    # 2. Add a test buyer to the DB to verify SQLModel mapping works
    print("--- Adding Test Buyer to Database ---")
    session_str = "35ca2942%2C1795675962%2Cc1803%2A52CjCWJ2B1N9IeYqBkfEoFD-ol2h9s02d6UDtXl7CypyM2AEXU3aejtZLbS3wBfRpaWFUSVkV1dmpyTmZiOG1meDBvSEd6aWhwZ0V5SXd3Wmxsa0U4MFd0anFwZ3UtekpfSGJNbUxKd19pM0FTVUJ2eEtmV016Ykk5MEp6bE8wTi0zWEMxZzNvSmxnIIEC"
    buyer = await add_buyer(
        buyer_name="林诺诚",
        tel="15618435583",
        sessdata=session_str,
        device_id="b91800ce78f2a77070a3ab35c8f3227c"
    )
    print(f"Added buyer: ID={buyer.id}, Name={buyer.buyer_name}, Tel={buyer.tel}")
    
    # List all buyers to verify retrieval
    buyers = await list_buyers()
    print(f"Current buyers in DB: {len(buyers)}")
    for b in buyers:
        print(f"  - Buyer ID: {b.id}, Name: {b.buyer_name}")
        
    # 3. Test Bilibili show list fetching & simplification
    print("\n--- Testing get_show_list ---")
    shows = await get_show_list(page=1, pagesize=3)  # Fetch top 3 shows
    print(f"Successfully fetched {len(shows)} shows.")
    print("First show item simplified structure:")
    if shows:
        print(json.dumps(shows[0], indent=4, ensure_ascii=False))
        
    # 4. Shutdown resources
    await connector_loader.shutdown()

if __name__ == "__main__":
    asyncio.run(test())
