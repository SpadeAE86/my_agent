import os
import sys
import time
import asyncio
from dotenv import load_dotenv

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(threadName)s][%(name)s] %(levelname)s:\t%(message)s")

async def test_dashscope_tts():
    from services.tts_services.dashscope_tts_service import DashScopeServiceEngine
    
    text = "大家好，欢迎收听由人工智能为您进行的第一期语音播报测试。"
    # Natural language voice design prompt
    instruct = "年轻活泼的女性声音，语速偏快，带有明显的上扬语调，适合介绍时尚产品。"
    
    print("=== DashScope TTS Engine Test via TTSManager ===")
    print(f"Using DASHSCOPE_API_KEY: {os.getenv('DASHSCOPE_API_KEY')}")
    print(f"Text: '{text}'")
    print(f"Instruct: '{instruct}'\n")
    
    print("--- 1. Testing CosyVoice (cosyvoice-v3.5-plus) ---")
    cosy_engine = DashScopeServiceEngine(model_name="cosyvoice-v3.5-plus")
    try:
        t0 = time.time()
        url = await cosy_engine.generate_voice(
            text=text,
            voice_character="cosyvoice_test_char",
            instruct=instruct
        )
        duration = time.time() - t0
        print(f"[CosyVoice Success] Generated Audio URL: {url}")
        print(f"Time taken: {duration:.2f} seconds.\n")
    except Exception as e:
        print("[CosyVoice Error] DashScope CosyVoice synthesis failed:")
        import traceback
        traceback.print_exc()
        print()

    print("--- 2. Testing Qwen3-TTS (qwen3-tts-vd-2026-01-26) ---")
    qwen_engine = DashScopeServiceEngine(model_name="qwen3-tts-vd-2026-01-26")
    try:
        t0 = time.time()
        url = await qwen_engine.generate_voice(
            text=text,
            voice_character="qwen_test_char",
            instruct=instruct
        )
        duration = time.time() - t0
        print(f"[Qwen3-TTS Success] Generated Audio URL: {url}")
        print(f"Time taken: {duration:.2f} seconds.\n")
    except Exception as e:
        print("[Qwen3-TTS Error] DashScope Qwen3-TTS synthesis failed:")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    asyncio.run(test_dashscope_tts())
