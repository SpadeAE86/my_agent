import os
import sys
import time
import asyncio
import soundfile as sf
import re
import json
import torch
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

async def run_consistency_test():
    from qwen_tts import Qwen3TTSModel
    
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    ref_text = "哟，找本堂主有何贵干啊？"
    instruct = "用十分温柔、亲切自然的语气说话，声音甜美，像是在耳边和好朋友聊天"
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    clauses = split_text_into_clauses(text)
    
    print("Loading Base model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded successfully.")
    
    # Tokenize the instruction text
    instruct_text = model._build_instruct_text(instruct)
    instruct_ids = model._tokenize_texts([instruct_text])
    
    output_base_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\consistency_eval"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # =========================================================================
    # Case 1: Baseline (x_vector_only_mode=True, temp=0.9)
    # =========================================================================
    print("\n--- Running Case 1: Baseline (x_vector_only, temp=0.9) ---")
    case1_dir = os.path.join(output_base_dir, "case1_baseline")
    os.makedirs(case1_dir, exist_ok=True)
    
    case1_wavs = []
    case1_sr = 24000  # Default Qwen3-TTS sample rate
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=clause,
            language="chinese",
            ref_audio=ref_audio,
            x_vector_only_mode=True,
            instruct_ids=instruct_ids,
            temperature=0.9,
            do_sample=True
        )
        case1_sr = sr
        print(f"Clause {idx} generated in {time.time() - t0:.2f}s")
        sf.write(os.path.join(case1_dir, f"{idx:02d}.wav"), wavs[0], sr)
        case1_wavs.append(wavs[0])
        
    combined_wav1 = np.concatenate(case1_wavs)
    sf.write(os.path.join(case1_dir, "combined.wav"), combined_wav1, case1_sr)
    print(f"Combined audio saved to: {case1_dir}\\combined.wav")
    
    # =========================================================================
    # Case 2: Low Temperature (x_vector_only_mode=True, temp=0.3)
    # =========================================================================
    print("\n--- Running Case 2: Low Temp (x_vector_only, temp=0.3) ---")
    case2_dir = os.path.join(output_base_dir, "case2_low_temp")
    os.makedirs(case2_dir, exist_ok=True)
    
    case2_wavs = []
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=clause,
            language="chinese",
            ref_audio=ref_audio,
            x_vector_only_mode=True,
            instruct_ids=instruct_ids,
            temperature=0.3,
            do_sample=True
        )
        print(f"Clause {idx} generated in {time.time() - t0:.2f}s")
        sf.write(os.path.join(case2_dir, f"{idx:02d}.wav"), wavs[0], sr)
        case2_wavs.append(wavs[0])
        
    combined_wav2 = np.concatenate(case2_wavs)
    sf.write(os.path.join(case2_dir, "combined.wav"), combined_wav2, case1_sr)
    print(f"Combined audio saved to: {case2_dir}\\combined.wav")
    
    # =========================================================================
    # Case 3: Static ICL Mode (x_vector_only_mode=False, ref_text=ref_text, temp=0.9)
    # =========================================================================
    print("\n--- Running Case 3: Static ICL (x_vector_only=False, temp=0.9) ---")
    case3_dir = os.path.join(output_base_dir, "case3_static_icl")
    os.makedirs(case3_dir, exist_ok=True)
    
    case3_wavs = []
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=clause,
            language="chinese",
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
            instruct_ids=instruct_ids,
            temperature=0.9,
            do_sample=True
        )
        print(f"Clause {idx} generated in {time.time() - t0:.2f}s")
        sf.write(os.path.join(case3_dir, f"{idx:02d}.wav"), wavs[0], sr)
        case3_wavs.append(wavs[0])
        
    combined_wav3 = np.concatenate(case3_wavs)
    sf.write(os.path.join(case3_dir, "combined.wav"), combined_wav3, case1_sr)
    print(f"Combined audio saved to: {case3_dir}\\combined.wav")

    # =========================================================================
    # Case 4: Static ICL Mode + Low Temperature (x_vector_only_mode=False, temp=0.3)
    # =========================================================================
    print("\n--- Running Case 4: Static ICL + Low Temp (temp=0.3) ---")
    case4_dir = os.path.join(output_base_dir, "case4_static_icl_low_temp")
    os.makedirs(case4_dir, exist_ok=True)
    
    case4_wavs = []
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=clause,
            language="chinese",
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
            instruct_ids=instruct_ids,
            temperature=0.3,
            do_sample=True
        )
        print(f"Clause {idx} generated in {time.time() - t0:.2f}s")
        sf.write(os.path.join(case4_dir, f"{idx:02d}.wav"), wavs[0], sr)
        case4_wavs.append(wavs[0])
        
    combined_wav4 = np.concatenate(case4_wavs)
    sf.write(os.path.join(case4_dir, "combined.wav"), combined_wav4, case1_sr)
    print(f"Combined audio saved to: {case4_dir}\\combined.wav")

    # =========================================================================
    # Case 5: Stateful ICL Chain (Stateful prompting, temp=0.3)
    # =========================================================================
    print("\n--- Running Case 5: Stateful ICL Chain (temp=0.3) ---")
    case5_dir = os.path.join(output_base_dir, "case5_stateful_icl_chain")
    os.makedirs(case5_dir, exist_ok=True)
    
    case5_wavs = []
    
    prev_wav = None
    prev_sr = None
    prev_text = ref_text
    
    for idx, clause in enumerate(clauses, 1):
        t0 = time.time()
        
        if idx == 1:
            cur_ref_audio = ref_audio
            cur_ref_text = ref_text
        else:
            cur_ref_audio = (prev_wav, prev_sr)
            cur_ref_text = prev_text
            
        wavs, sr = model.generate_voice_clone(
            text=clause,
            language="chinese",
            ref_audio=cur_ref_audio,
            ref_text=cur_ref_text,
            x_vector_only_mode=False,
            instruct_ids=instruct_ids,
            temperature=0.3,
            do_sample=True
        )
        print(f"Clause {idx} generated in {time.time() - t0:.2f}s")
        sf.write(os.path.join(case5_dir, f"{idx:02d}.wav"), wavs[0], sr)
        case5_wavs.append(wavs[0])
        
        # Save for next chunk reference
        prev_wav = wavs[0]
        prev_sr = sr
        prev_text = clause
        
    combined_wav5 = np.concatenate(case5_wavs)
    sf.write(os.path.join(case5_dir, "combined.wav"), combined_wav5, case1_sr)
    print(f"Combined audio saved to: {case5_dir}\\combined.wav")
    print("\nAll cases complete!")

if __name__ == "__main__":
    asyncio.run(run_consistency_test())
