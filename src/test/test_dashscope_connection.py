import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

def test_voice_design():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"Using API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
    
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": "qwen3-tts-vd-2026-01-26",
            "preferred_name": "hutao_design",
            "voice_prompt": "一个声音有些沙哑、语调活泼灵动的年轻女孩，语气中带着俏皮，语速轻快。",
            "preview_text": "你好，我是本堂主，今天有什么好玩的冒险吗？"
        },
        "parameters": {
            "sample_rate": 24000,
            "response_format": "wav"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("\n--- Sending Voice Design Creation Request ---")
    resp = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {resp.status_code}")
    print(f"Response Headers: {resp.headers}")
    try:
        res_json = resp.json()
        print(f"Response JSON: {json.dumps(res_json, indent=2, ensure_ascii=False)[:1000]}")
        return res_json
    except Exception as e:
        print(f"Raw Response: {resp.text}")
        return None

if __name__ == "__main__":
    test_voice_design()
