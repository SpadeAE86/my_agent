import httpx
import json

import httpx
import json

def test_endpoints():
    import os
    from pathlib import Path
    
    print("Testing chat history endpoints...")
    
    project_root = Path(__file__).resolve().parent.parent
    memory_dir = project_root / "data" / "sessions"
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    temp_sid = "test_endpoint_session_temp"
    temp_file = memory_dir / f"{temp_sid}.jsonl"
    
    # Write mock data with abstract metadata
    mock_history = [
        {"abstract": "测试临时会话"},
        {"role": "user", "content": "你好，这是一个测试消息"},
        {"role": "assistant", "content": "你好！我是AI助手。"}
    ]
    
    with open(temp_file, "w", encoding="utf-8") as f:
        for item in mock_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Created temporary session file: {temp_file}")
    
    try:
        with httpx.Client(trust_env=False) as client:
            # 2. Test session list
            resp = client.get("http://127.0.0.1:8001/chat/sessions")
            print("Sessions list status:", resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                print("Sessions count:", len(data.get("sessions", [])))
                # Check if our temp session is in the list and has correct title/abstract
                found_temp = None
                for s in data.get("sessions", []):
                    if s["session_id"] == temp_sid:
                        found_temp = s
                        break
                if found_temp:
                    print("Found temporary session in list!")
                    print("Title (Abstract):", found_temp["title"])
                    assert found_temp["title"] == "测试临时会话", "Abstract title mismatch!"
                else:
                    print("Warning: temporary session not found in list (page size limit?)")
            
            # 3. Test individual session details
            resp_detail = client.get(f"http://127.0.0.1:8001/chat/sessions/{temp_sid}")
            print(f"Session detail status for {temp_sid}:", resp_detail.status_code)
            if resp_detail.status_code == 200:
                events = resp_detail.json()
                print("Events loaded:", len(events))
                print("Events list:", [e["type"] for e in events])
                
            # 4. Test export endpoint
            resp_export = client.get(f"http://127.0.0.1:8001/chat/sessions/{temp_sid}/export")
            print("Export endpoint status:", resp_export.status_code)
            if resp_export.status_code == 200:
                print("Export file size:", len(resp_export.content))
                # Validate export content is valid jsonl matching our temporary file
                lines = resp_export.content.decode("utf-8").strip().split("\n")
                print(f"Exported JSONL lines count: {len(lines)}")
                assert len(lines) == 3, "Exported line count mismatch"
                
            # 5. Test delete endpoint
            resp_delete = client.delete(f"http://127.0.0.1:8001/chat/sessions/{temp_sid}")
            print("Delete endpoint status:", resp_delete.status_code)
            if resp_delete.status_code == 200:
                delete_res = resp_delete.json()
                print("Delete result:", delete_res)
                assert delete_res.get("ok") is True, "Delete failed"
                
            # 6. Verify file is deleted on backend
            if not temp_file.exists():
                print("Verified: Session file was successfully deleted from disk.")
            else:
                print("Error: Session file still exists on disk after DELETE request!")
                
    except Exception as e:
        print("An error occurred during verification:", e)
        # Cleanup file if it still exists
        if temp_file.exists():
            temp_file.unlink()
            print("Cleaned up temp file after exception.")
        raise e

if __name__ == "__main__":
    test_endpoints()
