import sys
import os
import json
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infra.storage.mysql_connector import mysql_connector
from services.video_match_services.video_history_db_service import video_history_db_service

async def main():
    # 1. Initialize DB connector
    await mysql_connector.init()
    
    # 2. Read old history JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "video_history.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    print(f"Loaded {len(items)} items from JSON.")
    
    # 3. Format fields to align with DB models
    formatted = []
    for item in items:
        # Ensure type is t2v or i2v
        t = item.get("type") or "t2v"
        if t not in ("t2v", "i2v"):
            t = "t2v"
        item["type"] = t
        
        # Keep URL fields mapped
        if "url" in item and "doubao_url" not in item:
            item["doubao_url"] = item["url"]
        
        # Try parsing date display format to datetime for DB insert
        time_str = item.get("time") or ""
        if time_str:
            try:
                # "5-26 16:04" -> 2026-05-26 16:04:00
                parts = time_str.split(" ")
                month_day = parts[0].split("-")
                hour_minute = parts[1].split(":")
                from datetime import datetime
                dt = datetime(2026, int(month_day[0]), int(month_day[1]), int(hour_minute[0]), int(hour_minute[1]))
                item["created_at"] = dt
            except Exception:
                pass
                
        formatted.append(item)
        
    # 4. Upsert into database
    await video_history_db_service.upsert_many(formatted)
    print(f"Successfully migrated {len(formatted)} video history records to MySQL database!")
    
    # 5. Clean up connector
    await mysql_connector.close()

if __name__ == "__main__":
    asyncio.run(main())
