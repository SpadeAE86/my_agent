import os
import sys
import torch
import json

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
from utils.voiceprint_validator import VoiceprintValidator

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    hu_tao_ref_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    
    # 待检测的三个音频
    test_files = {
        "1. 真人原始干音 (Hu Tao Original)": hu_tao_ref_path,
        "2. 第一代合成音 (Synthetic Ref - clone_with_instruct.wav)": r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\clone_with_instruct.wav",
        "3. 第二代复制音 (Reproduced - reproduced_from_voiceprint.wav)": r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\reproduced_from_voiceprint.wav"
    }

    # 检查基本文件是否存在
    if not os.path.exists(hu_tao_ref_path):
        print(f"Error: Hu Tao reference audio not found at: {hu_tao_ref_path}")
        return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.\n")

    # 提取真人原始音色指纹，作为比对基准
    print("Extracting Hu Tao original voiceprint as baseline...")
    original_prompt = model.create_voice_clone_prompt(
        ref_audio=hu_tao_ref_path,
        x_vector_only_mode=True
    )
    
    print("\n" + "="*80)
    print("VOICEPRINT QUALITY & EVOLUTION SUITABILITY REPORT")
    print("="*80)

    # 对三个测试文件逐一执行质量评估
    for label, path in test_files.items():
        print(f"\nEvaluating: {label}...")
        if not os.path.exists(path):
            print(f"   [SKIPPED] File not found: {path}")
            continue
            
        report = VoiceprintValidator.evaluate_audio_for_evolution(
            audio_path=path,
            model=model,
            ref_voiceprint=original_prompt
        )
        
        # 格式化输出报告
        print(f"   - File Name: {report['audio_file']}")
        print(f"   - High-Freq Energy (>8kHz): {report['hfe_ratio']:.6f} (Limit < 0.005)")
        print(f"   - Spectral Flatness (SFM):   {report['sfm']:.6f} (Limit < 0.060)")
        if report['similarity'] is not None:
            print(f"   - Cosine Similarity to Ref:  {report['similarity']:.4f} (Limit >= 0.965)")
        
        status_str = "[PASS] APPROVED" if report['suitable_for_evolution'] else "[FAIL] REJECTED"
        print(f"   - Suitability Status:        {status_str}")
        
        if report['warnings']:
            print("   - Warnings (警告细节):")
            for w in report['warnings']:
                print(f"     * {w}")
        print(f"   - Advice (改进建议): {report['advice']}")
        print("-" * 50)

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
