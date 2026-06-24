import os
import sys
import json
import requests
import asyncio
from dotenv import load_dotenv

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", ".env")
load_dotenv(env_path)

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"Using API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")

def create_voice():
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": "qwen3-tts-vd-realtime",
            "preferred_name": "hutaovd",
            "voice_prompt": "一个声音有些沙哑、语调活泼灵动的年轻女孩，语气中带着俏皮，语速轻快。",
            "preview_text": "你好，我是由你用文字描述设计出来的全新音色。"
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
    
    resp = requests.post(url, json=payload, headers=headers)
    print(f"Customization Status Code: {resp.status_code}")
    res_json = resp.json()
    
    # Print the output without base64 audio data to keep it clean
    clean_json = json.loads(json.dumps(res_json))
    if "output" in clean_json and "preview_audio" in clean_json["output"]:
        clean_json["output"]["preview_audio"]["data"] = "<base64 audio data truncated>"
    print("Customization Response:")
    print(json.dumps(clean_json, indent=2, ensure_ascii=False))
    return res_json

async def test_synthesis(voice_id, model_name):
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer
    
    dashscope.api_key = api_key
    
    text = "你不想要的是那种收到指令、执行指令、汇报完成的秘书型助手。你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。"
    print(f"\n--- Starting WebSocket Synthesis with model={model_name}, voice={voice_id} ---")
    
    synthesizer = SpeechSynthesizer(model=model_name, voice=voice_id)
    
    t0 = time.time() if 'time' in globals() else asyncio.get_event_loop().time()
    
    # SpeechSynthesizer call uses WebSocket streaming under the hood. 
    # Let's see if we can get metrics or check if it streams.
    audio_bytes = await asyncio.to_thread(synthesizer.call, text)
    
    t1 = asyncio.get_event_loop().time()
    duration = t1 - t0
    
    try:
        delay = synthesizer.get_first_package_delay()
        req_id = synthesizer.get_last_request_id()
        print(f"WebSocket Synthesis Success!")
        print(f"RequestID: {req_id}")
        print(f"First Packet Delay (TTFT): {delay}ms")
        print(f"Total Time: {duration:.2f}s")
        print(f"Audio length: {len(audio_bytes)} bytes")
        
        # Save output audio
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_design_test_synthesis.mp3")
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio saved to: {out_path}")
    except Exception as e:
        print(f"Failed to get metrics or save audio: {e}")

if __name__ == "__main__":
    import time
    res = create_voice()
    voice_id = res.get("output", {}).get("voice") or res.get("output", {}).get("voice_id")
    if voice_id:
        asyncio.run(test_synthesis(voice_id, "qwen3-tts-vd-realtime"))
    else:
        print("Failed to design voice - no voice ID returned.")
