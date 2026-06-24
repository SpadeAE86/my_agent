import os
import sys
import time
import asyncio

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass
import json
import urllib.request
import re

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded audio to local path: {save_path}")
    except Exception as e:
        print(f"Failed to download audio from {url}: {e}")

def slugify_instruction(instruct):
    if not instruct:
        return "none"
    # Keep only Chinese characters, English letters, and numbers
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", instruct)
    return cleaned[:20]  # truncate to keep it reasonable

async def run_evaluation():
    from services.tts_services import tts_manager
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    
    ref_dir = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日"
    all_files = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".wav")])
    first_5_files = all_files[:5]
    
    if len(first_5_files) < 5:
        print(f"Error: Found only {len(first_5_files)} wav files in {ref_dir}. Need at least 5.")
        sys.exit(1)
        
    print(f"Using top 5 reference audio files for clone test:")
    for idx, path in enumerate(first_5_files, 1):
        print(f"  {idx}: {os.path.basename(path)}")
        
    test_id = f"eval_{int(time.time())}"
    
    results = []

    # =========================================================================
    # 1. Voice Clone Mode (Base Model)
    # =========================================================================
    print("\n" + "="*50)
    print("Evaluating: Qwen3 Voice Clone (Base Model)")
    print("="*50)
    
    # Warm up base model first (to load it into VRAM and avoid counting load time)
    print("Warming up Qwen3 Base model...")
    await tts_manager.generate_voice(
        text="预热输入。",
        voice_character="warmup",
        engine_name="qwen3",
        ref_audio=first_5_files[0],
        x_vector_only_mode=True
    )
    print("Warmup complete. Commencing evaluation...")
    
    clone_times = []
    clone_urls = []
    
    # Run 5 consecutive runs
    for i in range(5):
        ref_path = first_5_files[i]
        print(f"\n--- Run {i+1}/5 ---")
        print(f"Ref Audio: {os.path.basename(ref_path)}")
        
        t0 = time.time()
        url = await tts_manager.generate_voice(
            text=text,
            voice_character=f"clone_{i+1}",
            engine_name="qwen3",
            ref_audio=ref_path,
            x_vector_only_mode=True
        )
        duration = time.time() - t0
        clone_times.append(duration)
        clone_urls.append(url)
        print(f"Success in {duration:.2f} seconds.")
        
        # Save local copy
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_tts", "clone", test_id)
        local_path = os.path.join(save_dir, f"{i+1:02d}.mp3")
        download_file(url, local_path)
        
    avg_clone_time = sum(clone_times) / len(clone_times)
    print(f"\nVoice Clone Evaluation Complete!")
    print(f"Average pure inference time: {avg_clone_time:.2f} seconds")

    results.append({
        "category": "clone",
        "test_id": test_id,
        "average_time_sec": round(avg_clone_time, 2),
        "run_times": [round(t, 2) for t in clone_times],
        "urls": clone_urls
    })

    # =========================================================================
    # 2. Voice Design Mode (VoiceDesign Model)
    # =========================================================================
    instruct = "一个声音有些沙哑、语调活泼灵动的年轻女孩，语气中带着俏皮"
    slug = slugify_instruction(instruct)
    design_test_id = f"{test_id}_instruct_{slug}"
    
    print("\n" + "="*50)
    print("Evaluating: Qwen3 Voice Design (VoiceDesign Model)")
    print(f"Instruction: {instruct}")
    print("="*50)
    
    # Warm up design model first
    print("Warming up Qwen3 VoiceDesign model...")
    await tts_manager.generate_voice(
        text="预热输入。",
        voice_character="warmup",
        engine_name="qwen3",
        instruct=instruct
    )
    print("Warmup complete. Commencing evaluation...")
    
    design_times = []
    design_urls = []
    
    # Run 5 consecutive runs to get average design inference speed
    for i in range(5):
        print(f"\n--- Run {i+1}/5 ---")
        t0 = time.time()
        url = await tts_manager.generate_voice(
            text=text,
            voice_character=f"design_{i+1}",
            engine_name="qwen3",
            instruct=instruct
        )
        duration = time.time() - t0
        design_times.append(duration)
        design_urls.append(url)
        print(f"Success in {duration:.2f} seconds.")
        
        # Save local copy
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_tts", "design", design_test_id)
        local_path = os.path.join(save_dir, f"{i+1:02d}.mp3")
        download_file(url, local_path)
        
    avg_design_time = sum(design_times) / len(design_times)
    print(f"\nVoice Design Evaluation Complete!")
    print(f"Average pure inference time: {avg_design_time:.2f} seconds")

    results.append({
        "category": "design",
        "test_id": design_test_id,
        "instruction": instruct,
        "average_time_sec": round(avg_design_time, 2),
        "run_times": [round(t, 2) for t in design_times],
        "urls": design_urls
    })
    
    # Save overall summary results
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_tts", f"evaluation_summary_{test_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("EVALUATION RUN COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
