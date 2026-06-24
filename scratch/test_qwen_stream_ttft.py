import os
import sys
import json
import requests
import asyncio
import time
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
            "target_model": "qwen3-tts-vd-2026-01-26",
            "preferred_name": "htstreamttft",
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
    res_json = resp.json()
    voice_id = res_json.get("output", {}).get("voice") or res_json.get("output", {}).get("voice_id")
    return voice_id

async def test_streaming_ttft(voice_id):
    import dashscope
    dashscope.api_key = api_key
    
    text = "你不想要的是那种“收到指令—执行指令—汇报完成”的秘书型助手。你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    model_name = "qwen3-tts-vd-2026-01-26"
    
    print(f"\n--- Synthesizing Text: '{text}' ---")
    print(f"Using model: {model_name}, Voice ID: {voice_id}")
    
    t_start = time.perf_counter()
    
    responses = await asyncio.to_thread(
        dashscope.MultiModalConversation.call,
        model=model_name,
        api_key=api_key,
        text=text,
        voice=voice_id,
        stream=True
    )
    
    ttft = None
    chunk_count = 0
    audio_bytes_accumulated = bytearray()
    
    def iterate_stream():
        nonlocal ttft, chunk_count, audio_bytes_accumulated
        for response in responses:
            chunk_count += 1
            status = response.status_code if hasattr(response, 'status_code') else 'Unknown'
            
            output = response.get("output", {})
            audio_info = output.get("audio", {})
            audio_data = audio_info.get("data", "")
            
            if audio_data:
                import base64
                raw_chunk = base64.b64decode(audio_data)
                audio_bytes_accumulated.extend(raw_chunk)
                
                if ttft is None:
                    # Capture the time when the first chunk of audio data arrives
                    ttft = time.perf_counter() - t_start
                    print(f"\n[TTFT] First audio packet received!")
                    print(f"Time from request start to first packet: {ttft*1000:.2f}ms")
            
            # Print periodic progress
            if chunk_count % 5 == 0 or not audio_data:
                print(f"Received chunk #{chunk_count}... Total bytes so far: {len(audio_bytes_accumulated)}")
                
    await asyncio.to_thread(iterate_stream)
    total_time = time.perf_counter() - t_start
    print(f"\n--- Synthesis Completed ---")
    print(f"Total Chunks: {chunk_count}")
    print(f"Total Audio Size: {len(audio_bytes_accumulated)} bytes")
    print(f"First Packet Latency (TTFT): {ttft*1000:.2f}ms" if ttft else "Failed to measure TTFT.")
    print(f"Total Request Duration: {total_time:.2f}s")
    
    # Save the output audio file
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_stream_ttft_output.wav")
    with open(out_path, "wb") as f:
        f.write(audio_bytes_accumulated)
    print(f"Audio saved to: {out_path}")
    
    # Copy to artifacts directory
    artifact_path = r"C:\Users\admin\.gemini\antigravity\brain\de3144c0-d9c7-4f69-8ea3-0d3b171d0d76\qwen_stream_ttft_output.wav"
    try:
        import shutil
        shutil.copy(out_path, artifact_path)
        print(f"Copied audio to artifacts: {artifact_path}")
    except Exception as e:
        print(f"Failed to copy to artifacts: {e}")

if __name__ == "__main__":
    print("Designing voice...")
    voice_id = create_voice()
    if voice_id:
        print(f"Voice designed successfully: {voice_id}")
        asyncio.run(test_streaming_ttft(voice_id))
    else:
        print("Failed to design voice.")
