import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal

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

def lowpass_filter(wav, cutoff_hz=7800.0, sr=24000, order=8):
    """
    使用高阶巴特沃斯低通滤波器切除高频白噪
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.lfilter(b, a, wav)
    return filtered

def normalize_volume(wav, target_peak=0.9):
    """
    音量峰值归一化
    """
    max_val = np.max(np.abs(wav))
    if max_val > 1e-4:
        return wav / max_val * target_peak
    return wav

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    hu_tao_ref_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    synthetic_ref_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\clone_with_instruct.wav"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_reconstruction"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 检查路径
    if not os.path.exists(hu_tao_ref_path):
        print(f"Error: Hu Tao reference audio not found at: {hu_tao_ref_path}")
        return
    if not os.path.exists(synthetic_ref_path):
        print(f"Error: Synthetic audio not found at: {synthetic_ref_path}")
        return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.\n")

    # Load original voiceprint as baseline
    print("Extracting Hu Tao original voiceprint as baseline...")
    original_prompt = model.create_voice_clone_prompt(
        ref_audio=hu_tao_ref_path,
        x_vector_only_mode=True
    )
    
    # Load synthetic audio
    wav, sr = librosa.load(synthetic_ref_path, sr=24000)
    
    # Test different cutoffs
    cutoffs = [7800.0, 9000.0, 10000.0]
    
    print("\n" + "="*80)
    print("TESTING DIFFERENT LOW-PASS CUTOFF FREQUENCIES")
    print("="*80)
    
    for cutoff in cutoffs:
        print(f"\n--- Testing Cutoff: {cutoff/1000:.1f} kHz ---")
        processed_wav = lowpass_filter(wav, cutoff_hz=cutoff, sr=sr, order=8)
        processed_wav = normalize_volume(processed_wav, target_peak=0.9)
        
        tmp_path = os.path.join(output_dir, f"temp_cutoff_{int(cutoff)}.wav")
        sf.write(tmp_path, processed_wav, sr)
        
        # Validate
        report = VoiceprintValidator.evaluate_audio_for_evolution(
            audio_path=tmp_path,
            model=model,
            ref_voiceprint=original_prompt
        )
        print(f"   - HFE Ratio (>8kHz): {report['hfe_ratio']:.6f} (Limit < 0.005)")
        print(f"   - SFM Flatness:      {report['sfm']:.6f} (Limit < 0.060)")
        print(f"   - Similarity to Ref: {report['similarity']:.4f} (Limit >= 0.965)")
        print(f"   - Suitability:       {report['suitable_for_evolution']}")
        
        # Try generating shy audio
        preprocessed_prompt = model.create_voice_clone_prompt(
            ref_audio=tmp_path,
            x_vector_only_mode=True
        )
        
        text = "那个……请问，我、我可以坐在你旁边吗？我有点害怕……"
        instruct = "极其害羞、语速缓慢地颤抖说话"
        instruct_text = model._build_instruct_text(instruct)
        instruct_ids = model._tokenize_texts([instruct_text])
        
        wavs_proc, sr_proc = model.generate_voice_clone(
            text=text,
            language="chinese",
            voice_clone_prompt=preprocessed_prompt,
            instruct_ids=instruct_ids
        )
        
        gen_path = os.path.join(output_dir, f"reproduced_shy_cutoff_{int(cutoff)}.wav")
        sf.write(gen_path, wavs_proc[0], sr_proc)
        
        # Evaluate generated audio similarity
        gen_report = VoiceprintValidator.evaluate_audio_for_evolution(
            audio_path=gen_path,
            model=model,
            ref_voiceprint=original_prompt
        )
        print(f"   - Generated Audio Similarity to Ref: {gen_report['similarity']:.4f}")
        
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
