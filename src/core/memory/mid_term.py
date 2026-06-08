# core/memory/mid_term.py — Mid-term memory (Daily Markdown Logs)
import os
import json
from pathlib import Path
from datetime import datetime
from core.memory import short_term
from core.memory import summarizer

MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "memory"
SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sessions"

def get_daily_log_path(user_id: str, date_str: str) -> Path:
    """Gets the path to the daily log markdown file: logs/YYYY/MM/YYYY-MM-DD.md"""
    # Parse date_str (YYYY-MM-DD)
    parts = date_str.split("-")
    if len(parts) != 3:
        # fallback
        now = datetime.now()
        year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%Y-%m-%d")
    else:
        year, month, day = parts[0], parts[1], date_str
        
    path = MEMORY_ROOT / user_id / "logs" / year / month / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def append_to_daily_log(user_id: str, date_str: str, text: str) -> None:
    """Appends summary text to the daily log file."""
    log_path = get_daily_log_path(user_id, date_str)
    now_time = datetime.now().strftime("%H:%M:%S")
    
    # Check if we need a heading
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Daily Log: {date_str}\n")
        f.write(f"\n## Entry [{now_time}]\n{text}\n")

async def consolidate_session_to_daily_log(
    user_id: str,
    session_id: str,
    client,
    model: str
) -> bool:
    """
    Consolidates session diary (session.md) and chat transcript (.jsonl) into the daily log.
    Returns True if consolidation was performed, False otherwise.
    """
    # 1. Load session diary
    session_diary = short_term.load_session_diary(user_id, session_id)
    
    # 2. Load JSONL transcript
    jsonl_path = SESSIONS_DIR / f"{session_id}.jsonl"
    jsonl_lines = []
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        # Read raw json structure, simplify it for summary context
                        try:
                            item = json.loads(line.strip())
                            if "role" in item:
                                role = item.get("role")
                                content = item.get("content", "")
                                if isinstance(content, list):
                                    text = " ".join([i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text"])
                                else:
                                    text = str(content)
                                jsonl_lines.append(f"{role}: {text[:500]}")
                        except Exception:
                            continue
        except Exception:
            pass
            
    jsonl_content = "\n".join(jsonl_lines)
    
    if not session_diary and not jsonl_content:
        return False
        
    # Generate daily log entry summary
    summary = await summarizer.generate_session_summary(
        client,
        model,
        session_diary,
        jsonl_content
    )
    
    if not summary.strip():
        return False
        
    # Append to today's daily log
    today_str = datetime.now().strftime("%Y-%m-%d")
    append_to_daily_log(user_id, today_str, f"### Session {session_id} Summary\n{summary}")
    return True
