import os
import sys
import torch
import soundfile as sf

# Ensure sys.modules mapping for flash_attn_3 is set up
try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts import Qwen3TTSModel

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    voiceprint_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\synthetic_voiceprint.pt"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_experiments"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check voiceprint exists
    if not os.path.exists(voiceprint_path):
        print(f"Error: Saved voiceprint file not found at: {voiceprint_path}")
        return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.")

    # Load the saved voiceprint
    print("Loading the voiceprint file...")
    loaded_prompt = torch.load(voiceprint_path, weights_only=False)

    # ============================================================
    # 实验一：同一句话，3 种不同说话方式 (Same Text, 3 Different Styles)
    # ============================================================
    print("\n--- Running Experiment 1: Same Text, 3 Different Styles ---")
    exp1_text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。"
    
    styles = {
        "shy": "极其害羞、语速缓慢地颤抖说话",
        "confident": "自信满满地大声说话，情绪高昂，充满热情",
        "friendly": "用十分温柔、亲切自然的语气说话，声音甜美，像是在耳边和好朋友聊天"
    }

    for name, instruct in styles.items():
        print(f"Generating Experiment 1 - Style: {name} ('{instruct}')")
        instruct_text = model._build_instruct_text(instruct)
        instruct_ids = model._tokenize_texts([instruct_text])
        
        wavs, sr = model.generate_voice_clone(
            text=exp1_text,
            language="chinese",
            voice_clone_prompt=loaded_prompt,
            instruct_ids=instruct_ids
        )
        
        save_path = os.path.join(output_dir, f"exp1_{name}.wav")
        sf.write(save_path, wavs[0], sr)
        print(f"   Saved to: {save_path}")

    # ============================================================
    # 实验二：同一说话方式，3 种不同文本 (Same Style, 3 Different Texts)
    # ============================================================
    print("\n--- Running Experiment 2: Same Style, 3 Different Texts ---")
    exp2_style = "极其害羞、语速缓慢地颤抖说话"
    
    texts = {
        "text1": "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。",
        "text2": "今天的天气真好呀，我们一起出去散散步吧？",
        "text3": "哟，找本堂主有何贵干啊？"
    }

    instruct_text = model._build_instruct_text(exp2_style)
    instruct_ids = model._tokenize_texts([instruct_text])

    for name, text in texts.items():
        print(f"Generating Experiment 2 - Text: {name} ('{text}')")
        
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="chinese",
            voice_clone_prompt=loaded_prompt,
            instruct_ids=instruct_ids
        )
        
        save_path = os.path.join(output_dir, f"exp2_{name}_shy.wav")
        sf.write(save_path, wavs[0], sr)
        print(f"   Saved to: {save_path}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Outputs are stored in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
