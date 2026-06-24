import os
import sys
import json
import requests
import asyncio
import time
from dotenv import load_dotenv

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", ".env")
load_dotenv(env_path)

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"Using API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")

async def upload_ref_audio():
    from utils.obs_utils import upload_audio
    ref_audio_local = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    print(f"Uploading local reference audio to OBS: {ref_audio_local}")
    
    project_id = f"ref_clone_{int(time.time())}"
    url = await upload_audio(ref_audio_local, project_id=project_id)
    print(f"Reference Audio OBS URL: {url}")
    return url

def enroll_voice(audio_url, target_model, prefix):
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
      "model": "qwen-voice-enrollment",
      "input": {
        "action": "create",
        "target_model": target_model,
        "preferred_name": prefix,
        "audio": {
          "data": audio_url
        }
      }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"\n--- Calling voice-enrollment for target_model: {target_model} ---")
    resp = requests.post(url, json=payload, headers=headers)
    print(f"Enrollment Status Code: {resp.status_code}")
    res_json = resp.json()
    print("Enrollment Response:")
    print(json.dumps(res_json, indent=2, ensure_ascii=False))
    return res_json.get("output", {}).get("voice") or res_json.get("output", {}).get("voice_id")

async def synthesize_qwen(voice_id, model_name):
    import dashscope
    dashscope.api_key = api_key
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    instructions = "极其害羞、语速缓慢地颤抖说话"
    
    print(f"\n--- Calling MultiModalConversation with cloned voice for {model_name} ---")
    t0 = time.perf_counter()
    
    try:
        response = await asyncio.to_thread(
            dashscope.MultiModalConversation.call,
            model=model_name,
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
            return None
            
        audio_url = response.get("output", {}).get("audio", {}).get("url")
        if audio_url:
            print(f"Download Audio URL: {audio_url}")
            dl_resp = requests.get(audio_url)
            if dl_resp.status_code == 200:
                out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"qwen_cloned_styled_{model_name.replace('-', '_')}.wav")
                with open(out_path, "wb") as f:
                    f.write(dl_resp.content)
                print(f"Saved audio to: {out_path}")
                print(f"Time taken: {t1 - t0:.2f} seconds.")
                return out_path
        else:
            print(f"No audio URL returned in output: {response}")
    except Exception as e:
        print(f"Synthesis failed: {e}")
    return None

async def main():
    # 1. Upload reference audio
    audio_url = await upload_ref_audio()
    
    # 2. Try Qwen3 Voice Cloning enrollment
    # Model target names for Qwen3: we'll try "qwen3-tts-vc-2026-01-22" first or just "qwen3-tts-vc"
    voice_id = enroll_voice(audio_url, "qwen3-tts-vc-2026-01-22", "htcl")
    if not voice_id:
        print("Retrying voice enrollment with 'qwen3-tts-vc'...")
        voice_id = enroll_voice(audio_url, "qwen3-tts-vc", "htcl")
        
    if voice_id:
        print(f"Successfully obtained cloned voice ID: {voice_id}")
        
        # 3. Synthesize with Qwen3 Voice Cloning target model
        # We can also try qwen3-tts-instruct-flash or qwen3-tts-vc-2026-01-22
        await synthesize_qwen(voice_id, "qwen3-tts-vc-2026-01-22")
    else:
        print("Failed to enroll voice on Qwen.")

if __name__ == "__main__":
    asyncio.run(main())
