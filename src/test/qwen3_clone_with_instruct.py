import os
import sys
import time
import asyncio
import torch
import soundfile as sf

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

async def clone_with_instruct():
    from qwen_tts import Qwen3TTSModel
    
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Base model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.")
    
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    
    # 1. Base Voice Clone without instruction (for comparison)
    print("\n--- Running Case 1: Pure Clone ---")
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="chinese",
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    print(f"Completed in {time.time() - t0:.2f} seconds.")
    sf.write(os.path.join(output_dir, "clone_pure.wav"), wavs[0], sr)
    
    # 2. Base Voice Clone with tokenized instruct_ids passed to kwargs!
    # Let's see if we can tokenize the instruction
    instruct = "极其害羞、语速缓慢地颤抖说话"
    print(f"\n--- Running Case 2: Clone with Instruct '{instruct}' ---")
    
    try:
        # Tokenize the instruction text just like VoiceDesign does
        instruct_text = model._build_instruct_text(instruct)
        instruct_ids = model._tokenize_texts([instruct_text])
        
        t0 = time.time()
        # Pass instruct_ids in kwargs to see if the talker accepts it
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="chinese",
            ref_audio=ref_audio,
            x_vector_only_mode=True,
            instruct_ids=instruct_ids
        )
        print(f"Completed in {time.time() - t0:.2f} seconds.")
        sf.write(os.path.join(output_dir, "clone_with_instruct.wav"), wavs[0], sr)
        print("Success! The model generated audio with both clone and instruct parameters.")
    except Exception as e:
        print("Failed to run clone with instruct:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(clone_with_instruct())
