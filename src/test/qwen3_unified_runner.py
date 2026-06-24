import os
import sys
import time
import asyncio
import urllib.request
import re
import shutil
import soundfile as sf
import numpy as np

# Ensure sys.modules mapping for flash_attn_3 is set up
# This mimics the environment setup of the other test scripts
try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

# Add src directory to Python path so it can be run easily from PyCharm
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

# ----------------- 可配置变量 (Configuration Variables) -----------------
# 1. 朗读文本 (Text to generate)
TEXT = (
    "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
)

# 2. 参考音频与文本 (Reference audio and transcript)
REF_AUDIO = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
REF_TEXT = "哟，找本堂主有何贵干啊？"

# 3. 情感/语气风格提示词 (Style / Emotion instruction)
INSTRUCT = "极其害羞、语速缓慢地颤抖说话"

# 4. 推理策略切换 (Strategy selection)
# True: 按句切分生成并合并 (Option B 策略，用于超低首包延迟)
# False: 整段文本一次性输入生成 (用于最佳语调连贯性)
STREAM = False

# 5. 特征提取模式 (Reference Feature extraction mode)
# False: 全参考模式 (Full Reference Mode / Static ICL) —— 会同时提取音频与 REF_TEXT 文本 of the acoustic features, voice reconstruction is highest
# True: 极速极简模式 (X-Vector Only Mode) —— 忽略参考文本，只从音频快速提取声线特征
X_VECTOR_ONLY_MODE = True

# 6. 采样参数 (Sampling configurations)
TEMPERATURE = 0.9
DO_SAMPLE = True
# ------------------------------------------------------------------------

def split_text_into_sentences(text):
    """根据句号、感叹号、问号及换行符将文本切分为短句"""
    parts = re.split(r'([。！？!?\n])', text)
    sentences = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        punct = parts[i+1].strip() if i + 1 < len(parts) else ""
        i += 2
        full_sentence = part + punct
        if full_sentence.strip():
            sentences.append(full_sentence)
    return [s for s in sentences if s.strip()]

async def main():
    # Import the registered tts_manager
    from services.tts_services import tts_manager
    
    print("=" * 60)
    print("Qwen3-TTS Unified Runner")
    print(f"Strategy: {'Sentence Bundling (STREAM=True)' if STREAM else 'Full-Text Direct (STREAM=False)'}")
    print(f"Instruct: '{INSTRUCT}'")
    print(f"X-Vector Only: {X_VECTOR_ONLY_MODE}")
    print("=" * 60)
    
    # Create output directories
    output_base_dir = os.path.join(SRC_DIR, "test", "debug_tts", "unified_run")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Warm up Qwen3 model first (ensures model loading time is excluded from TTFT)
    print("Warming up Qwen3 model...")
    t_warmup = time.time()
    await tts_manager.generate_voice(
        text="预热。",
        voice_character="warmup",
        engine_name="qwen3",
        ref_audio=REF_AUDIO,
        x_vector_only_mode=True
    )
    print(f"Warmup complete in {time.time() - t_warmup:.2f}s.\n")
    
    t_inference_start = time.time()
    
    if STREAM:
        # --- Option B: Sentence Bundling Generation ---
        sentences = split_text_into_sentences(TEXT)
        print(f"Split text into {len(sentences)} segments:")
        for idx, s in enumerate(sentences, 1):
            print(f"  [{idx}] '{s}'")
        print("-" * 40)
        
        segment_times = []
        local_files = []
        
        for idx, sentence in enumerate(sentences, 1):
            t0 = time.time()
            url = await tts_manager.generate_voice(
                text=sentence,
                voice_character=f"unified_seg_{idx}",
                engine_name="qwen3",
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                x_vector_only_mode=X_VECTOR_ONLY_MODE,
                instruct=INSTRUCT,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE
            )
            duration = time.time() - t0
            segment_times.append(duration)
            
            # Print TTFT / Step timings
            if idx == 1:
                print(f"  >>> [FIRST PACKET (首包 / TTFT)] Sentence 1 generated in {duration:.2f} seconds.")
            else:
                print(f"  >>> [Sentence {idx}] Generated in {duration:.2f} seconds.")
            
            # Download audio segments locally
            save_path = os.path.join(output_base_dir, f"seg_{idx:02d}.mp3")
            try:
                urllib.request.urlretrieve(url, save_path)
                local_files.append(save_path)
            except Exception as e:
                print(f"  Failed to save segment locally: {e}")
        
        # Concat segments using FFmpeg
        combined_path = os.path.join(output_base_dir, "combined_streamed.mp3")
        concat_txt_path = os.path.join(output_base_dir, "concat.txt")
        try:
            with open(concat_txt_path, "w", encoding="utf-8") as concat_f:
                for f_path in local_files:
                    # Use relative basename to prevent path issues in ffmpeg txt
                    concat_f.write(f"file '{os.path.basename(f_path)}'\n")
            
            import subprocess
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "combined_streamed.mp3"]
            subprocess.run(cmd, cwd=output_base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"\n  >>> Combined streamed audio saved successfully to: {combined_path}")
            
            # Cleanup temporary list file
            if os.path.exists(concat_txt_path):
                os.remove(concat_txt_path)
        except Exception as e:
            print(f"\n  >>> Failed to concatenate audio segments: {e}")
            
        total_time = time.time() - t_inference_start
        print("\n" + "="*50)
        print("STREAMING MODE COMPLETED")
        print(f"First Packet Latency (首包 / TTFT): {segment_times[0]:.2f}s")
        print(f"Average Segment Duration: {sum(segment_times)/len(segment_times):.2f}s")
        print(f"Total Combined Generation Time: {total_time:.2f}s")
        print("="*50)
        
    else:
        # --- Single-sentence Full Text Direct Generation ---
        print("Generating full text in one single request...")
        t0 = time.time()
        url = await tts_manager.generate_voice(
            text=TEXT,
            voice_character="unified_full",
            engine_name="qwen3",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            x_vector_only_mode=X_VECTOR_ONLY_MODE,
            instruct=INSTRUCT,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE
        )
        total_time = time.time() - t0
        
        # Download output
        save_path = os.path.join(output_base_dir, "combined_full.mp3")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"\n  >>> Full direct audio saved successfully to: {save_path}")
        except Exception as e:
            print(f"  Failed to save audio locally: {e}")
            
        print("\n" + "="*50)
        print("FULL DIRECT MODE COMPLETED")
        print(f"First Packet Latency (首包 / TTFT): {total_time:.2f}s (same as total)")
        print(f"Total Generation Time: {total_time:.2f}s")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
