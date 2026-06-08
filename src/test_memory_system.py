# test_memory_system.py — Unit tests for the memory and scheduler architecture
import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Ensure src in sys.path
import sys
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.agent.agent import Agent
from core.memory import short_term, mid_term, long_term, memory_manager, summarizer
from infra.scheduler.scheduler import scheduler_manager
from infra.scheduler.jobs.memory_summary import run_daily_memory_consolidation_and_dream

# Temporary data directory for tests
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data_memory"

def setup_module():
    # Override data directories for isolation
    short_term.MEMORY_ROOT = TEST_DATA_DIR / "memory"
    mid_term.MEMORY_ROOT = TEST_DATA_DIR / "memory"
    mid_term.SESSIONS_DIR = TEST_DATA_DIR / "sessions"
    long_term.MEMORY_ROOT = TEST_DATA_DIR / "memory"
    
    # Ensure clean directory
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def teardown_module():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)

async def test_token_counting():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How many tokens is this?"}
    ]
    tokens = short_term.count_tokens(messages, model="gpt-4")
    print(f"Token count: {tokens}")
    assert tokens > 0

async def test_session_diary():
    user_id = "test_user_1"
    session_id = "sess_100"
    
    short_term.save_session_diary(user_id, session_id, "Session summary content.")
    content = short_term.load_session_diary(user_id, session_id)
    print(f"Loaded session diary:\n{content}")
    assert "Session summary content." in content

async def test_compaction():
    # Setup mock agent
    class MockClient:
        pass
    mock_agent = Agent(
        user_id="test_user_1",
        llm={"client": MockClient(), "model": "gpt-4"},
        tools=[],
        skills={},
        session_id="sess_100"
    )
    # Patch sessions save location
    mock_agent.save_history = lambda: None
    
    # Set messages
    mock_agent.messages = [
        {"role": "user", "content": "Initial user query 1"},
        {"role": "assistant", "content": "Assistant response 1", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_123", "content": "tool result"},
        {"role": "user", "content": "Query 2"},
        {"role": "assistant", "content": "Assistant response 2"}
    ]
    
    # Mock summarizer function
    with patch("core.memory.summarizer.generate_compaction_summary", AsyncMock(return_value="Summarized compaction output")):
        # Force compaction
        compacted = await short_term.compact_session_history(mock_agent, force=True)
        assert compacted is True
        
        # Verify compaction result
        assert len(mock_agent.messages) > 0
        assert mock_agent.messages[0]["role"] == "system"
        assert "Compaction Boundary" in mock_agent.messages[0]["content"]
        assert mock_agent.messages[1]["role"] == "user"
        assert "Summarized compaction output" in mock_agent.messages[1]["content"]

async def test_daily_log_and_consolidation():
    user_id = "test_user_1"
    session_id = "sess_100"
    
    # Prepare dummy session diary and transcript
    short_term.save_session_diary(user_id, session_id, "Summary of active task.")
    
    # Write dummy JSONL
    sess_dir = TEST_DATA_DIR / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = sess_dir / f"{session_id}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": "Let's build a feature."}) + "\n")
        f.write(json.dumps({"role": "assistant", "content": "Understood. I will write code."}) + "\n")
        
    with patch("core.memory.summarizer.generate_session_summary", AsyncMock(return_value="Consolidated Session Entry Content")):
        success = await mid_term.consolidate_session_to_daily_log(
            user_id=user_id,
            session_id=session_id,
            client=None,
            model="gpt-4"
        )
        assert success is True
        
        # Verify daily log file
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_log_path = mid_term.get_daily_log_path(user_id, today_str)
        assert daily_log_path.exists()
        with open(daily_log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
            print(f"Daily log content:\n{log_content}")
            assert "Consolidated Session Entry Content" in log_content

async def test_long_term_memory_dream():
    user_id = "test_user_1"
    
    # Mock dream consolidation
    with patch("core.memory.summarizer.dream_and_merge", AsyncMock(return_value="# Memory\n\n- [Index](topic.md) -- pointer\n- Fact: User likes python.")):
        success = await long_term.dream_consolidation(
            user_id=user_id,
            client=None,
            model="gpt-4",
            days_back=1
        )
        assert success is True
        
        # Verify MEMORY.md
        memory_content = long_term.load_long_term_memory(user_id)
        print(f"Memory content:\n{memory_content}")
        assert "Fact: User likes python." in memory_content

async def test_memory_prompt_injection():
    user_id = "test_user_1"
    
    # Get memory prompt
    prompt = await memory_manager.get_memory_prompt(user_id, skip_memory=False)
    assert prompt is not None
    assert "# Memory" in prompt
    assert "Fact: User likes python." in prompt
    
    # Skip memory
    skipped_prompt = await memory_manager.get_memory_prompt(user_id, skip_memory=True)
    assert skipped_prompt is None

async def test_scheduler_manager():
    scheduler_manager.start()
    
    dummy_called = asyncio.Event()
    async def dummy_job():
        dummy_called.set()
        
    scheduler_manager.add_cron_job(dummy_job, "test_job_1", hour=0, minute=0)
    assert scheduler_manager._scheduler.get_job("test_job_1") is not None
    
    scheduler_manager.shutdown()

async def run_all_tests():
    print("Initializing test setup...")
    setup_module()
    try:
        print("\n--- Running test_token_counting ---")
        await test_token_counting()
        print("\n--- Running test_session_diary ---")
        await test_session_diary()
        print("\n--- Running test_compaction ---")
        await test_compaction()
        print("\n--- Running test_daily_log_and_consolidation ---")
        await test_daily_log_and_consolidation()
        print("\n--- Running test_long_term_memory_dream ---")
        await test_long_term_memory_dream()
        print("\n--- Running test_memory_prompt_injection ---")
        await test_memory_prompt_injection()
        print("\n--- Running test_scheduler_manager ---")
        await test_scheduler_manager()
        print("\nSUCCESS: All unit tests passed!")
    finally:
        print("Cleaning up test data...")
        teardown_module()

if __name__ == "__main__":
    import json
    from datetime import datetime
    asyncio.run(run_all_tests())
