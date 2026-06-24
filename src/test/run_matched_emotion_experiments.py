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
    # 实验二重构：情绪匹配测试 (Matched Emotion & Text Experiment)
    # ============================================================
    print("\n--- Running Matched Emotion & Text Experiment ---")
    
    cases = [
        {
            "name": "matched_shy",
            "text": "那个……请问，我、我可以坐在你旁边吗？我有点害怕……",
            "instruct": "极其害羞、语速缓慢地颤抖说话"
        },
        {
            "name": "matched_confident",
            "text": "没问题！交给我吧！今天我们绝对会取得胜利的，出发！",
            "instruct": "自信满满地大声说话，情绪高昂，充满热情"
        },
        {
            "name": "matched_angry",
            "text": "你给我站住！为什么要骗我？！我再也不想见到你了！滚！",
            "instruct": "极其愤怒地咆哮说话，语气粗暴，情绪非常激动和失控"
        }
    ]

    for case in cases:
        name = case["name"]
        text = case["text"]
        instruct = case["instruct"]
        
        print(f"Generating case: {name}")
        print(f"   Text: '{text}'")
        print(f"   Style: '{instruct}'")
        
        instruct_text = model._build_instruct_text(instruct)
        instruct_ids = model._tokenize_texts([instruct_text])
        
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="chinese",
            voice_clone_prompt=loaded_prompt,
            instruct_ids=instruct_ids
        )
        
        save_path = os.path.join(output_dir, f"exp2_{name}.wav")
        sf.write(save_path, wavs[0], sr)
        print(f"   Saved to: {save_path}\n")

    print("="*60)
    print("MATCHED EMOTION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Outputs are stored in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
