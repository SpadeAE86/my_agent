# core/memory/long_term.py — Long-term memory (MEMORY.md Consolidation)
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from core.memory import summarizer

MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "memory"

def load_long_term_memory(user_id: str) -> str:
    """Loads the main MEMORY.md file for the user."""
    memory_path = MEMORY_ROOT / user_id / "MEMORY.md"
    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_long_term_memory(user_id: str, content: str) -> None:
    """Saves or updates the main MEMORY.md file for the user."""
    user_dir = MEMORY_ROOT / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    memory_path = user_dir / "MEMORY.md"
    with open(memory_path, "w", encoding="utf-8") as f:
        f.write(content)

async def dream_consolidation(
    user_id: str,
    client,
    model: str,
    days_back: int = 1
) -> bool:
    """
    Consolidates daily logs from the past N days into MEMORY.md.
    Returns True if consolidation succeeded, False otherwise.
    """
    logs_dir = MEMORY_ROOT / user_id / "logs"
    if not logs_dir.exists():
        return False
        
    # Find all daily log files
    log_pattern = str(logs_dir / "**" / "*.md")
    log_files = glob.glob(log_pattern, recursive=True)
    if not log_files:
        return False
        
    # Filter log files modified or dated in the past N days
    cutoff_time = datetime.now() - timedelta(days=days_back)
    recent_logs = []
    
    for file_path in log_files:
        path_obj = Path(file_path)
        mtime = datetime.fromtimestamp(path_obj.stat().st_mtime)
        if mtime >= cutoff_time:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    recent_logs.append(f.read())
            except Exception:
                continue
                
    if not recent_logs:
        return False
        
    # Read existing MEMORY.md
    old_memory = load_long_term_memory(user_id)
    
    # Run the consolidation dream merging process
    consolidated_content = await summarizer.dream_and_merge(
        client,
        model,
        old_memory,
        recent_logs
    )
    
    if not consolidated_content.strip():
        return False
        
    # Save the consolidated output back to MEMORY.md
    save_long_term_memory(user_id, consolidated_content)
    return True
