# core/memory/short_term.py — Short-term memory (JSONL history and session diary)
import os
import json
from pathlib import Path
from datetime import datetime
import tiktoken
from core.memory import summarizer

MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "memory"

def get_session_dir(user_id: str, session_id: str) -> Path:
    """Gets the path to the session directory."""
    path = MEMORY_ROOT / user_id / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_session_diary(user_id: str, session_id: str, summary: str) -> None:
    """Saves or appends the session-level short-term diary (session.md)."""
    session_dir = get_session_dir(user_id, session_id)
    file_path = session_dir / "session.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # If file exists, append; otherwise write new
    mode = "a" if file_path.exists() else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        f.write(f"\n## Session Update [{now_str}]\n{summary}\n")

def load_session_diary(user_id: str, session_id: str) -> str:
    """Loads the session-level short-term diary (session.md)."""
    session_dir = get_session_dir(user_id, session_id)
    file_path = session_dir / "session.md"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def count_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """Accurately estimates token usage using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # message metadata overhead
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("type") == "text":
                        num_tokens += len(encoding.encode(item.get("text", "")))
            if key == "name":
                num_tokens += -1
    num_tokens += 2  # priming reply token count
    return num_tokens

async def compact_session_history(agent, force: bool = False) -> bool:
    """
    Checks token usage of agent.messages and runs compaction if threshold exceeded.
    Returns True if compaction was performed, False otherwise.
    """
    messages = agent.messages
    if len(messages) < 4:
        return False
        
    model_name = agent.llm.get("model", "gpt-4")
    token_count = count_tokens(messages, model=model_name)
    
    # Compaction threshold: 16384 tokens (50% of 32768 context size)
    threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", 16384))
    
    if not force and token_count <= threshold:
        return False
        
    # Drop oldest 40% of messages
    drop_count = int(len(messages) * 0.40)
    if drop_count < 2:
        return False
        
    dropped_messages = messages[:drop_count]
    kept_messages = messages[drop_count:]
    
    # --- repairToolUseResultPairing logic ---
    # Find all valid tool call IDs in the kept messages section
    valid_tool_call_ids = set()
    for msg in kept_messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict) and "id" in tc:
                    valid_tool_call_ids.add(tc["id"])
                    
    # Drop orphaned tool results from kept messages
    filtered_kept = []
    dropped_orphans_count = 0
    for msg in kept_messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id not in valid_tool_call_ids:
                dropped_orphans_count += 1
                continue
        filtered_kept.append(msg)
        
    # Generate LLM summary of dropped turns
    summary = await summarizer.generate_compaction_summary(
        agent.llm["client"],
        model_name,
        dropped_messages
    )
    
    # Boundary and summary messages
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    boundary_marker = {
        "role": "system",
        "content": f"[Compaction Boundary: {now_str}]"
    }
    summary_message = {
        "role": "user",
        "content": f"[Conversation Summary of previous turns:\n{summary}]",
        "is_summary": True
    }
    
    # Replace in agent messages
    agent.messages = [boundary_marker, summary_message] + filtered_kept
    agent.save_history()
    
    # Set compaction flags for loops to yield event
    agent.just_compacted = True
    agent.compaction_summary = summary
    
    # Update short-term session diary
    save_session_diary(agent.user_id, agent.session_id, summary)
    return True
