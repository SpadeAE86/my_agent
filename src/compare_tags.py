import asyncio
import os
import sys
import pickle

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.collections import CollectionItem
from sqlmodel import select

async def check_tag_diff():
    # 1. Initialize MySQL Connector
    print("Connecting to MySQL Database...")
    await mysql_connector.init()
    
    # 2. Query all tags in MySQL
    db_tags = set()
    async with mysql_connector.session_scope() as session:
        res = await session.execute(select(CollectionItem))
        items = res.scalars().all()
        for item in items:
            if item.tags:
                for t in item.tags:
                    db_tags.add(t.strip())
                    
    # 3. Load tag library (AutoClusterOrchestrator pickle)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
    if not os.path.exists(state_file):
        state_file = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
        
    lib_tags = set()
    if os.path.exists(state_file):
        print(f"Loading tag library from: {state_file}")
        try:
            with open(state_file, "rb") as f:
                state = pickle.load(f)
                history = state.get("history", {})
                for path, items in history.items():
                    for item in items:
                        lib_tags.add(item["tag"].strip())
        except Exception as e:
            print(f"Error loading pickle state: {e}")
    else:
        print(f"Warning: Pickle state file not found at {state_file}")

    # 4. Compare
    in_db_but_not_lib = db_tags - lib_tags
    in_lib_but_not_db = lib_tags - db_tags
    in_both = db_tags & lib_tags

    # 5. Check last sync time
    import json
    import time
    sync_time_file = os.path.join(base_dir, "test", "tag_library_sync_time.json")
    if not os.path.exists(sync_time_file):
        sync_time_file = os.path.join(base_dir, "core", "auto_cluster", "tag_library_sync_time.json")
        
    last_sync_time_str = "Unknown / Never synced"
    if os.path.exists(sync_time_file):
        try:
            with open(sync_time_file, "r") as f:
                sync_data = json.load(f)
                last_ts = sync_data.get("last_sync_timestamp", 0)
                if last_ts > 0:
                    diff_seconds = int(time.time() - last_ts)
                    minutes, seconds = divmod(diff_seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    last_sync_time_str = f"{hours}h {minutes}m {seconds}s ago"
        except Exception as e:
            last_sync_time_str = f"Error: {e}"

    print("\n" + "="*50)
    print(" DATABASE vs TAG LIBRARY SYNCHRONIZATION REPORT ")
    print("="*50)
    print(f"Last Mirror Sync to DB Table: {last_sync_time_str}")
    print(f"Total Unique Tags in MySQL: {len(db_tags)}")
    print(f"Total Unique Tags in Library: {len(lib_tags)}")
    print(f"Tags fully synchronized: {len(in_both)}")
    print("-"*50)
    
    if in_db_but_not_lib:
        print(f"\n[WARNING] Found {len(in_db_but_not_lib)} tags in MySQL but NOT in Tag Library (auto_cluster_state.pkl):")
        for idx, tag in enumerate(sorted(in_db_but_not_lib), 1):
            print(f"  {idx}. {tag}")
    else:
        print("\n[SUCCESS] No missing tags! All MySQL tags are present in the Tag Library.")
        
    if in_lib_but_not_db:
        print(f"\n[INFO] Found {len(in_lib_but_not_db)} tags in Tag Library but NOT currently used in MySQL:")
        for idx, tag in enumerate(sorted(in_lib_but_not_db), 1):
            print(f"  {idx}. {tag}")
            
    print("="*50)
    
    # Close connections
    await mysql_connector.close()

if __name__ == "__main__":
    asyncio.run(check_tag_diff())
