import os
import sys
import time
import asyncio
import urllib.request
import re
import json

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved audio locally: {save_path}")
    except Exception as e:
        print(f"Failed to download audio from {url}: {e}")

def slugify_instruction(instruct):
    if not instruct:
        return "none"
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", instruct)
    return cleaned[:20]

def split_text_into_clauses(text):
    parts = re.split(r'([。，、！？\n,.!?])', text)
    clauses = []
    current = ""
    for i in range(0, len(parts)-1, 2):
        part = parts[i].strip()
        punct = parts[i+1].strip()
        if part:
            clauses.append(part + punct)
    if len(parts) % 2 == 1 and parts[-1].strip():
        clauses.append(parts[-1].strip())
    return [c for c in clauses if c.strip()]

async def main():
    from services.tts_services import tts_manager
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    ref_audio = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    
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
    
    print("=== Combined Clone + Style Control Evaluation ===")
    print(f"Reference Audio: {os.path.basename(ref_audio)}")
    print(f"Text: '{text}'")
    print("=================================================\n")
    
    # Warm up first
    print("Warming up Qwen3 Base model...")
    await tts_manager.generate_voice(
        text="预热。",
        voice_character="warmup",
        engine_name="qwen3",
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    print("Warmup complete.\n")
    
    test_id = f"eval_clone_instruct_{int(time.time())}"
    
    clauses = split_text_into_clauses(text)
    
    results = []
    
    for case in test_cases:
        name = case["name"]
        instruct = case["instruct"]
        desc = case["description"]
        slug = slugify_instruction(instruct)
        
        case_test_id = f"{test_id}_instruct_{slug}"
        print(f"--- Running Case: {name} ({desc}) ---")
        print(f"Instruct Prompt: '{instruct}'")
        
        segment_times = []
        urls = []
        
        total_start = time.time()
        
        for idx, clause in enumerate(clauses, 1):
            t0 = time.time()
            url = await tts_manager.generate_voice(
                text=clause,
                voice_character=f"clone_instruct_{name}_{idx}",
                engine_name="qwen3",
                ref_audio=ref_audio,
                x_vector_only_mode=True,
                instruct=instruct
            )
            duration = time.time() - t0
            segment_times.append(duration)
            urls.append(url)
            
            if idx == 1:
                print(f"  >>> [FIRST PACKET (首包 / TTFT)] Clause 1 finished in {duration:.2f} seconds.")
            else:
                print(f"  >>> [Clause {idx}] Finished in {duration:.2f} seconds.")
                
            # Download audio segment locally
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug_tts",
                "clone_instruct",
                case_test_id,
                f"{idx:02d}.mp3"
            )
            download_file(url, save_path)
            
        # Concat downloaded segments using FFmpeg
        import subprocess
        case_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug_tts",
            "clone_instruct",
            case_test_id
        )
        concat_txt_path = os.path.join(case_dir, "concat.txt")
        combined_mp3_path = os.path.join(case_dir, "combined.mp3")
        try:
            with open(concat_txt_path, "w", encoding="utf-8") as concat_f:
                for idx in range(1, len(clauses) + 1):
                    concat_f.write(f"file '{idx:02d}.mp3'\n")
            
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "combined.mp3"]
            subprocess.run(cmd, cwd=case_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"  >>> Combined coherent audio saved to: {combined_mp3_path}")
            if os.path.exists(concat_txt_path):
                os.remove(concat_txt_path)
        except Exception as merge_err:
            print(f"  >>> Warning: Failed to combine audio segments using FFmpeg: {merge_err}")
            
        total_duration = time.time() - total_start
        avg_segment_time = sum(segment_times) / len(segment_times)
        
        case_results = {
            "name": name,
            "description": desc,
            "instruct": instruct,
            "ttft_sec": round(segment_times[0], 2),
            "avg_clause_sec": round(avg_segment_time, 2),
            "total_duration_sec": round(total_duration, 2),
            "segment_times": [round(t, 2) for t in segment_times],
            "urls": urls
        }
        results.append(case_results)
        
        print(f"\nCompleted Case: {name}")
        print(f"  First Packet Time (TTFT): {segment_times[0]:.2f} seconds")
        print(f"  Average Clause Time: {avg_segment_time:.2f} seconds")
        print(f"  Total Duration: {total_duration:.2f} seconds\n")
        
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
    asyncio.run(main())
