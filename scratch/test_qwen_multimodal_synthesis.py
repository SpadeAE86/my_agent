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

def create_voice(target_model, name):
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": name,
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
    print(f"Customization for {target_model} Status Code: {resp.status_code}")
    res_json = resp.json()
    return res_json

async def try_multimodal_call(model_name, voice_id):
    import dashscope
    dashscope.api_key = api_key
    
    text = "你不想要的是那种收到指令、执行指令、汇报完成的秘书型助手。你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    print(f"\n--- Trying MultiModalConversation.call with model={model_name}, voice={voice_id} ---")
    
    try:
        t0 = asyncio.get_event_loop().time()
        response = await asyncio.to_thread(
            dashscope.MultiModalConversation.call,
            model=model_name,
            api_key=api_key,
            text=text,
            voice=voice_id,
            stream=False
        )
        t1 = asyncio.get_event_loop().time()
        print(f"Status Code: {response.status_code if hasattr(response, 'status_code') else 'Unknown'}")
        
        if not hasattr(response, 'status_code') or response.status_code != 200:
            print(f"Error Message: {getattr(response, 'message', 'N/A')}")
            return False
            
        audio_url = response.get("output", {}).get("audio", {}).get("url")
        if audio_url:
            print(f"Success! Audio URL: {audio_url}")
            # Download the wav file
            print("Downloading audio file...")
            audio_resp = requests.get(audio_url)
            if audio_resp.status_code == 200:
                out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"qwen_multimodal_{model_name.replace('-', '_')}.wav")
                with open(out_path, "wb") as f:
                    f.write(audio_resp.content)
                print(f"Audio downloaded and saved to: {out_path}")
                return True
            else:
                print(f"Failed to download audio from OSS: {audio_resp.status_code}")
        else:
            print("No audio URL found in response.")
            
    except Exception as e:
        print(f"Call failed: {e}")
        return False

async def try_websocket_synthesis(model_name, voice_id):
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer
    dashscope.api_key = api_key
    
    text = "你不想要的是那种收到指令、执行指令、汇报完成的秘书型助手。你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。"
    print(f"\n--- Trying WebSocket SpeechSynthesizer with model={model_name}, voice={voice_id} ---")
    
    try:
        t0 = asyncio.get_event_loop().time()
        synthesizer = SpeechSynthesizer(model=model_name, voice=voice_id)
        
        # This will execute WebSocket connection and streaming
        audio_bytes = await asyncio.to_thread(synthesizer.call, text)
        t1 = asyncio.get_event_loop().time()
        
        delay = synthesizer.get_first_package_delay()
        req_id = synthesizer.get_last_request_id()
        
        print(f"WebSocket SpeechSynthesizer Success!")
        print(f"RequestID: {req_id}")
        print(f"First Packet Delay (TTFT): {delay}ms")
        print(f"Total Time taken: {t1 - t0:.2f}s")
        print(f"Audio length: {len(audio_bytes)} bytes")
        
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"qwen_websocket_{model_name.replace('-', '_')}.mp3")
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print(f"WebSocket Audio saved to: {out_path}")
        return True
    except Exception as e:
        print(f"WebSocket Synthesis failed for {model_name}: {e}")
        return False

async def main():
    import random
    
    # Part 1: Download from non-real-time model qwen3-tts-vd-2026-01-26
    # Let's use the voice_id from our previous successful run
    prev_voice_id = "qwen-tts-vd-htmulti793-voice-20260621044630419-d9a5"
    print("=== PART 1: Testing Non-Real-time Model Synthesis ===")
    await try_multimodal_call("qwen3-tts-vd-2026-01-26", prev_voice_id)
    
    # Part 2: Test Real-time Model customization + WebSocket streaming
    print("\n=== PART 2: Testing Real-time Model WebSocket Streaming ===")
    realtime_models = [
        "qwen3-tts-vd-realtime-2026-01-15",
        "qwen3-tts-vd-realtime-2025-12-16",
        "qwen3-tts-vd-realtime"
    ]
    
    for rt_model in realtime_models:
        print(f"\n--- Testing Realtime Model: {rt_model} ---")
        rt_name = f"htreal{random.randint(100, 999)}"
        res = create_voice(rt_model, rt_name)
        voice_id = res.get("output", {}).get("voice") or res.get("output", {}).get("voice_id")
        
        if voice_id:
            print(f"Voice designed successfully: {voice_id}")
            success = await try_websocket_synthesis(rt_model, voice_id)
            if success:
                print(f"WebSocket synthesis succeeded for model {rt_model}!")
                break
        else:
            print(f"Failed to create voice for model {rt_model}. Error: {res.get('message', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
