import os
import sys
import time
import asyncio
import urllib.request
import re

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved audio locally: {save_path}")
    except Exception as e:
        print(f"Failed to download audio from {url}: {e}")

async def main():
    from services.tts_services import tts_manager
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    
    # Let's test the CustomVoice model with different styling instructions!
    # Predefined speaker 'vivian' or 'ryan'
    speaker = "vivian"
    
    test_cases = [
        {
            "name": "shy",
            "description": "害羞颤抖",
            "instruct": "害羞低沉地说话，语速很慢，语气有些颤抖和娇羞"
        },
        {
            "name": "confident",
            "description": "自信满满",
            "instruct": "自信满满地大声说话，情绪高昂，语气坚定有力，充满热情"
        },
        {
            "name": "friendly",
            "description": "亲切自然",
            "instruct": "用十分温柔、亲切自然的语气说话，声音甜美，像是在耳边和好朋友聊天"
        }
    ]
    
    print("=== CustomVoice Style Control Evaluation ===")
    print(f"Speaker: {speaker}")
    print(f"Text: '{text}'")
    print("============================================\n")
    
    # Warm up first
    print("Warming up CustomVoice model...")
    await tts_manager.generate_voice(
        text="预热。",
        voice_character=speaker,
        engine_name="qwen3"
    )
    print("Warmup complete.\n")
    
    test_id = f"style_eval_{int(time.time())}"
    
    for case in test_cases:
        name = case["name"]
        instruct = case["instruct"]
        desc = case["description"]
        
        print(f"--- Running Case: {name} ({desc}) ---")
        print(f"Instruct Prompt: '{instruct}'")
        
        t0 = time.time()
        # CustomVoice matches because voice_character = 'vivian' which is in preset list,
        # and we pass 'instruct' in kwargs.
        url = await tts_manager.generate_voice(
            text=text,
            voice_character=speaker,
            engine_name="qwen3",
            instruct=instruct
        )
        duration = time.time() - t0
        print(f"Completed in {duration:.2f} seconds. URL: {url}")
        
        # Save output locally
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug_tts",
            "style_control",
            test_id,
            f"{name}.mp3"
        )
        download_file(url, save_path)
        print()

if __name__ == "__main__":
    asyncio.run(main())
