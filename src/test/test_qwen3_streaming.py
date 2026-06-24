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
import re

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_text_into_clauses(text):
    # Split by common punctuation marks (both Chinese and English)
    parts = re.split(r'([。，、！？\n,.!?])', text)
    clauses = []
    current = ""
    for i in range(0, len(parts)-1, 2):
        part = parts[i].strip()
        punct = parts[i+1].strip()
        if part:
            clauses.append(part + punct)
    # Add any trailing text
    if len(parts) % 2 == 1 and parts[-1].strip():
        clauses.append(parts[-1].strip())
    return [c for c in clauses if c.strip()]

async def run_streaming_test():
    from services.tts_services import tts_manager
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    ref_audio = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    
    clauses = split_text_into_clauses(text)
    print("=== Text Splitting for Streaming Evaluation ===")
    for idx, clause in enumerate(clauses, 1):
        print(f"Segment {idx}: '{clause}'")
    print("===============================================")
    
    # Warm up base model first
    print("Warming up Qwen3 Base model...")
    await tts_manager.generate_voice(
        text="预热。",
        voice_character="warmup",
        engine_name="qwen3",
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    print("Warmup complete.")
    
    print("\nStarting streaming simulation (consecutive clause generation)...")
    
    segment_times = []
    total_start = time.time()
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        url = await tts_manager.generate_voice(
            text=clause,
            voice_character=f"stream_seg_{idx}",
            engine_name="qwen3",
            ref_audio=ref_audio,
            x_vector_only_mode=True
        )
        duration = time.time() - t0
        segment_times.append(duration)
        
        if idx == 1:
            print(f">>> [FIRST PACKET (首包)] Segment 1 finished in {duration:.2f} seconds. URL: {url}")
        else:
            print(f">>> [Segment {idx}] Finished in {duration:.2f} seconds. URL: {url}")
            
    total_duration = time.time() - total_start
    avg_segment_time = sum(segment_times) / len(segment_times)
    
    print("\n===============================================")
    print("STREAMING METRICS SUMMARY:")
    print("===============================================")
    print(f"First Packet Time (首包延迟 / TTFT): {segment_times[0]:.2f} seconds")
    print(f"Average Segment Generation Time: {avg_segment_time:.2f} seconds")
    print(f"Total Combined Pipeline Duration: {total_duration:.2f} seconds")
    print(f"Individual Segment Durations: {[round(t, 2) for t in segment_times]}")
    print("===============================================")

if __name__ == "__main__":
    asyncio.run(run_streaming_test())
