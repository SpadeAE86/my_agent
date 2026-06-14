# core/memory/memory_manager.py — Unified Memory interface
from pathlib import Path
from core.memory import long_term

MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25000

def truncate_memory_content(raw: str) -> str:
    """Truncates memory content to line and byte limits, appending a warning if exceeded."""
    trimmed = raw.strip()
    if not trimmed:
        return ""
        
    lines = trimmed.split("\n")
    line_count = len(lines)
    byte_count = len(trimmed)
    
    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES
    
    if not was_line_truncated and not was_byte_truncated:
        return trimmed
        
    # Truncate by lines first
    truncated_lines = lines[:MAX_ENTRYPOINT_LINES]
    truncated = "\n".join(truncated_lines)
    
    # Truncate by bytes if still too large
    if len(truncated) > MAX_ENTRYPOINT_BYTES:
        cut_at = truncated.rfind("\n", 0, MAX_ENTRYPOINT_BYTES)
        truncated = truncated[:cut_at if cut_at > 0 else MAX_ENTRYPOINT_BYTES]
        
    reason = ""
    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit: {MAX_ENTRYPOINT_BYTES})"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"
        
    warning = (
        f"\n\n> WARNING: MEMORY.md is truncated due to exceeding {reason}. "
        "Only part of it was loaded. Keep index entries to one line under ~200 chars; "
        "move detail into topic files."
    )
    return truncated + warning

async def get_memory_prompt(user_id: str, skip_memory: bool = False, role_id: str = "default") -> str | None:
    """
    Reads MEMORY.md and constructs the memory section for the system prompt.

    - If role_id is provided and is not "default", reads from data/roles/{role_id}/MEMORY.md.
    - Otherwise reads from the legacy data/memory/{user_id}/MEMORY.md (CC mode).
    - Returns None if skip_memory is True or memory is empty.
    """
    if skip_memory:
        return None

    if role_id and role_id != "default":
        raw_memory = long_term.load_role_memory(role_id)
    else:
        raw_memory = long_term.load_long_term_memory(user_id)

    if not raw_memory.strip():
        return None

    formatted = truncate_memory_content(raw_memory)
    if not formatted.strip():
        return None

    return f"# Memory\n\n{formatted}"

