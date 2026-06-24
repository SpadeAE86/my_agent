import os
import sys
import time
import asyncio
import urllib.request
import re
import json
import soundfile as sf
import numpy as np

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_text_into_sentences(text):
    # Split only by sentence-level punctuation: 。！？!? or newline
    parts = re.split(r'([。！？!?\n])', text)
    sentences = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        punct = parts[i+1].strip() if i + 1 < len(parts) else ""
        i += 2
        # Clean quotes or brackets if they end up empty
        full_sentence = part + punct
        if full_sentence.strip():
            # If sentence ends with a quote, let's keep it clean
            sentences.append(full_sentence)
    return [s for s in sentences if s.strip()]

async def main():
    from services.tts_services import tts_manager
    
    text = (
        "你不想要的是那种“收到指令—执行指令—汇报完成”的秘书型助手。\n"
        "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    )
    ref_audio = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    ref_text = "哟，找本堂主有何贵干啊？"
    instruct = "用十分温柔、亲切自然的语气说话，声音甜美，像是在耳边和好朋友聊天"
    
    sentences = split_text_into_sentences(text)
    print("=== Option B: Sentence Bundling splitting ===")
    for idx, s in enumerate(sentences, 1):
        print(f"Sentence {idx} ({len(s)} chars): '{s}'")
    print("=============================================\n")
    
    # Warm up base model first
    print("Warming up Qwen3 Base model...")
    await tts_manager.generate_voice(
        text="预热。",
        voice_character="warmup",
        engine_name="qwen3",
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    print("Warmup complete.\n")
    
    test_id = f"option_b_{int(time.time())}"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_tts", "option_b", test_id)
    os.makedirs(output_dir, exist_ok=True)
    
    segment_times = []
    urls = []
    
    total_start = time.time()
    
    for idx, sentence in enumerate(sentences, 1):
        t0 = time.time()
        # Use Static ICL (x_vector_only_mode=False) with temperature 0.9 (unrestricted)
        url = await tts_manager.generate_voice(
            text=sentence,
            voice_character=f"option_b_{idx}",
            engine_name="qwen3",
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
            instruct=instruct,
            temperature=0.9,
            do_sample=True
        )
        duration = time.time() - t0
        segment_times.append(duration)
        urls.append(url)
        
        if idx == 1:
            print(f"  >>> [FIRST PACKET (首包 / TTFT)] Sentence 1 finished in {duration:.2f} seconds.")
        else:
            print(f"  >>> [Sentence {idx}] Finished in {duration:.2f} seconds.")
            
        # Download segment audio locally
        save_path = os.path.join(output_dir, f"{idx:02d}.mp3")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"  Saved locally: {save_path}")
        except Exception as e:
            print(f"  Failed to save segment locally: {e}")
            
    # Concat downloaded segments using FFmpeg
    import subprocess
    concat_txt_path = os.path.join(output_dir, "concat.txt")
    combined_mp3_path = os.path.join(output_dir, "combined.mp3")
    try:
        with open(concat_txt_path, "w", encoding="utf-8") as concat_f:
            for idx in range(1, len(sentences) + 1):
                concat_f.write(f"file '{idx:02d}.mp3'\n")
        
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "combined.mp3"]
        subprocess.run(cmd, cwd=output_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"\n  >>> Combined coherent audio saved to: {combined_mp3_path}")
        if os.path.exists(concat_txt_path):
            os.remove(concat_txt_path)
    except Exception as merge_err:
        print(f"\n  >>> Warning: Failed to combine audio segments using FFmpeg: {merge_err}")
        
    total_duration = time.time() - total_start
    avg_segment_time = sum(segment_times) / len(segment_times)
    
    print("\n" + "="*50)
    print("OPTION B EVALUATION RUN COMPLETE")
    print(f"First Packet Time (首包 / TTFT): {segment_times[0]:.2f} seconds")
    print(f"Average Sentence Time: {avg_segment_time:.2f} seconds")
    print(f"Total Combined Duration: {total_duration:.2f} seconds")
    print(f"Urls generated: {urls}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
