import os
import sys
import json
import base64
import pathlib
import requests
import asyncio
import time
from dotenv import load_dotenv

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", ".env")
load_dotenv(env_path)

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"Using API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")

DEFAULT_TARGET_MODEL = "qwen3-tts-vc-2026-01-22"
VOICE_FILE_PATH = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"

def create_voice(file_path: str, target_model: str = DEFAULT_TARGET_MODEL) -> str:
    """
    Creates custom voice using base64 encoded audio data and returns the voice ID
    """
    file_path_obj = pathlib.Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    print("Encoding reference audio to Base64...")
    base64_str = base64.b64encode(file_path_obj.read_bytes()).decode()
    data_uri = f"data:audio/wav;base64,{base64_str}"

    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": "htclone",
            "audio": {"data": data_uri}
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("Registering voice on DashScope...")
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"创建 voice 失败: {resp.status_code}, {resp.text}")

    res_json = resp.json()
    print("Enrollment response JSON:")
    print(json.dumps(res_json, indent=2, ensure_ascii=False))
    return res_json["output"]["voice"]

async def synthesize_clone_with_instruct(voice_id):
    import dashscope
    dashscope.api_key = api_key
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    instructions = "极其害羞、语速缓慢地颤抖说话"
    
    print(f"\n--- Synthesizing Text: '{text}' ---")
    print(f"Using Model: {DEFAULT_TARGET_MODEL}")
    print(f"Voice ID: {voice_id}")
    print(f"Instructions: {instructions}")
    
    t0 = time.perf_counter()
    response = await asyncio.to_thread(
        dashscope.MultiModalConversation.call,
        model=DEFAULT_TARGET_MODEL,
        api_key=api_key,
        text=text,
        voice=voice_id,
        instructions=instructions,
        optimize_instructions=True,
        stream=False
    )
    t1 = time.perf_counter()
    
    print(f"Synthesis status code: {response.status_code if hasattr(response, 'status_code') else 'Unknown'}")
    if not hasattr(response, 'status_code') or response.status_code != 200:
        print(f"Error Code: {getattr(response, 'code', 'N/A')}")
        print(f"Error Message: {getattr(response, 'message', 'N/A')}")
        return
        
    audio_url = response.get("output", {}).get("audio", {}).get("url")
    if audio_url:
        print(f"Success! Audio URL: {audio_url}")
        print("Downloading wav audio file...")
        dl_resp = requests.get(audio_url)
        if dl_resp.status_code == 200:
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replicated_clone_instruct.wav")
            with open(out_path, "wb") as f:
                f.write(dl_resp.content)
            print(f"Audio saved locally to: {out_path}")
            
            # Copy to artifacts directory
            artifact_path = r"C:\Users\admin\.gemini\antigravity\brain\de3144c0-d9c7-4f69-8ea3-0d3b171d0d76\replicated_clone_instruct.wav"
            import shutil
            shutil.copy(out_path, artifact_path)
            print(f"Copied audio to artifacts: {artifact_path}")
            print(f"Total time taken: {t1 - t0:.2f}s")
        else:
            print(f"Failed to download audio from OSS: {dl_resp.status_code}")
    else:
        print("No audio URL found in response.")

if __name__ == "__main__":
    try:
        voice_id = create_voice(VOICE_FILE_PATH)
        asyncio.run(synthesize_clone_with_instruct(voice_id))
    except Exception as e:
        print(f"Failed: {e}")
