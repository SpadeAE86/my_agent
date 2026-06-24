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
    hu_tao_ref_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_experiments"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check reference audio exists
    if not os.path.exists(hu_tao_ref_path):
        print(f"Error: Original reference audio not found at: {hu_tao_ref_path}")
        return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.")

    # 1. Extract voiceprint from the ORIGINAL Hu Tao audio
    print(f"Extracting voiceprint from original audio: {os.path.basename(hu_tao_ref_path)}")
    original_prompt = model.create_voice_clone_prompt(
        ref_audio=hu_tao_ref_path,
        x_vector_only_mode=True
    )

    # 2. Run the matched scenarios
    print("\n--- Running Matched Emotion & Text Experiment (Using ORIGINAL voiceprint) ---")
    
    cases = [
        {
            "name": "original_matched_shy",
            "text": "那个……请问，我、我可以坐在你旁边吗？我有点害怕……",
            "instruct": "极其害羞、语速缓慢地颤抖说话"
        },
        {
            "name": "original_matched_confident",
            "text": "没问题！交给我吧！今天我们绝对会取得胜利的，出发！",
            "instruct": "自信满满地大声说话，情绪高昂，充满热情"
        },
        {
            "name": "original_matched_angry",
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
            voice_clone_prompt=original_prompt,
            instruct_ids=instruct_ids
        )
        
        save_path = os.path.join(output_dir, f"exp3_{name}.wav")
        sf.write(save_path, wavs[0], sr)
        print(f"   Saved to: {save_path}\n")

    print("="*60)
    print("ORIGINAL VOICEPRINT EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Outputs are stored in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
