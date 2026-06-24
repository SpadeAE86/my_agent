import sys
import os
import asyncio
from sqlalchemy import inspect, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.storage.mysql_connector import mysql_connector
from sqlmodel import SQLModel
import models.sqlmodel  # registers all models in SQLModel.metadata

async def compare_schema():
    await mysql_connector.init()
    engine = await mysql_connector.get_engine()
    
    # Reflect current DB schema
    async with engine.connect() as conn:
        def get_db_info(connection):
            inspector = inspect(connection)
            db_tables = inspector.get_table_names()
            table_columns = {}
            for table in db_tables:
                columns = inspector.get_columns(table)
                table_columns[table] = {col['name']: col for col in columns}
            return db_tables, table_columns
        
        db_tables, db_columns = await conn.run_sync(get_db_info)

    # Get models from SQLModel.metadata
    model_tables = SQLModel.metadata.tables
    
    print("=== DB SCHEMA COMPARISON ===")
    
    missing_tables = []
    missing_columns = {}
    ddl_statements = []

    # 1. Compare tables
    for table_name, table_obj in model_tables.items():
        if table_name not in db_tables:
            missing_tables.append(table_name)
            continue
        
        # 2. Compare columns
        for col_name, col_obj in table_obj.columns.items():
            if col_name not in db_columns[table_name]:
                if table_name not in missing_columns:
                    missing_columns[table_name] = []
                missing_columns[table_name].append(col_obj)

    # 3. Print findings and DDL
    if missing_tables:
        print(f"\n[!] Missing Tables in DB ({len(missing_tables)}):")
        for t in missing_tables:
            print(f"  - {t}")
        print("  -> Fix: Restarting the FastAPI server will automatically run SQLModel.metadata.create_all() to create them.")
        
    if missing_columns:
        print(f"\n[!] Missing Columns in Existing Tables:")
        for t, cols in missing_columns.items():
            print(f"  Table: {t}")
            for col in cols:
                # Generate ALTER TABLE statement
                col_type = col.type.compile(dialect=engine.dialect)
                nullable_str = "NULL" if col.nullable else "NOT NULL"
                default_str = ""
                # Simple check for defaults
                if col.default is not None and hasattr(col.default, 'arg'):
                    default_str = f" DEFAULT {col.default.arg}"
                alter_sql = f"ALTER TABLE {t} ADD COLUMN {col.name} {col_type} {nullable_str}{default_str};"
                ddl_statements.append(alter_sql)
                print(f"    - Column: {col.name} ({col_type}) -> Fix DDL: {alter_sql}")
                
    # 4. Check auto-increment offset for image_history_cards
    if 'image_history_cards' in db_tables:
        async with engine.connect() as conn:
            # Query AUTO_INCREMENT status
            query = """
            SELECT AUTO_INCREMENT 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE() AND table_name = 'image_history_cards'
            """
            res = await conn.execute(text(query))
            auto_inc = res.scalar()
            
            # Also get max id
            query_max = "SELECT MAX(numeric_id) FROM image_history_cards"
            res_max = await conn.execute(text(query_max))
            max_id = res_max.scalar() or 0
            
            print(f"\n=== Auto-increment Check for image_history_cards ===")
            print(f"  Current AUTO_INCREMENT: {auto_inc}")
            print(f"  Max numeric_id: {max_id}")
            
            if auto_inc is not None and auto_inc < 10000 and max_id < 10000:
                alter_inc = "ALTER TABLE image_history_cards AUTO_INCREMENT = 10000;"
                ddl_statements.append(alter_inc)
                print(f"  [!] AUTO_INCREMENT is {auto_inc} (expected >= 10000 to prevent ID conflicts with legacy systems).")
                print(f"    -> Fix DDL: {alter_inc}")
            else:
                print("  AUTO_INCREMENT is aligned (> 10000 or table has large IDs).")

    if ddl_statements:
        print("\n=== SQL FIX SCRIPTS ===")
        print("You can run the following SQL commands in your MySQL client to sync the schema:")
        print("\n" + "\n".join(ddl_statements))
    else:
        print("\n[+] Database schema is fully in sync with SQLModel schemas!")

    await mysql_connector.close()

if __name__ == '__main__':
    asyncio.run(compare_schema())
