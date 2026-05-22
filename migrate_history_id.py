import sqlalchemy as sa

def migrate_db(db_url):
    engine = sa.create_engine(db_url)
    try:
        with engine.begin() as conn:
            # check if id is already int
            res = conn.execute(sa.text("SHOW COLUMNS FROM video_material_match_history LIKE 'id'"))
            row = res.fetchone()
            if b"int" in row[1].lower().encode('utf-8') or "int" in row[1].lower():
                print(f"[{db_url}] already migrated.")
                return
            
            print(f"Migrating [{db_url}]...")
            conn.execute(sa.text("ALTER TABLE video_material_match_history DROP PRIMARY KEY"))
            conn.execute(sa.text("ALTER TABLE video_material_match_history DROP COLUMN id"))
            conn.execute(sa.text("ALTER TABLE video_material_match_history ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY FIRST"))
            print(f"[{db_url}] migration successful!")
    except Exception as e:
        print(f"Error on [{db_url}]: {e}")

if __name__ == "__main__":
    # Local
    migrate_db("mysql+pymysql://root:RootDev123@127.0.0.1:3306/mark")
    # Test
    migrate_db("mysql+pymysql://freeu:RootDev123@1.94.143.208:3306/ai_recommend_test")

