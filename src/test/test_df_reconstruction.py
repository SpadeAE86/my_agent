import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio

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
from analyze_pitch_variance import analyze_pitch

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 初始化 DeepFilterNet 并运行增强
    print("\n--- Step 1: Loading DeepFilterNet model ---")
    try:
        # init_df will automatically download v3 model weights if not present
        df_model, df_state, _ = init_df()
        print("DeepFilterNet model initialized.")
    except Exception as e:
        print(f"Error initializing DeepFilterNet: {e}")
        return

    print("\n--- Step 2: Enhancing audio with DeepFilterNet ---")
    try:
        df_output_path = os.path.join(output_dir, "clone_with_instruct_df.wav")
        # Load audio (automatically resampled to 48kHz for DFv3)
        audio, _ = load_audio(synthetic_ref_path, sr=df_state.sr())
        
        # Run enhancement
        # Move tensor to GPU if available for faster execution, though DF is very fast on CPU
        # DeepFilterNet enhance API automatically handles tensor device or CPU numpy
        enhanced_audio = enhance(df_model, df_state, audio)
        
        # Save output
        save_audio(df_output_path, enhanced_audio, df_state.sr())
        print(f"DeepFilterNet Enhanced audio saved to: {df_output_path}")
    except Exception as e:
        print(f"Error during DeepFilterNet enhancement: {e}")
        return

    # 3. 加载 Qwen3-TTS 模型并验证 DF 增强后的音频
    print("\n--- Step 3: Loading Qwen3 model ---")
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Qwen3 Model loaded.")
    
    # 提取 Hu Tao 原始音色指纹作为比对基准
    original_prompt = model.create_voice_clone_prompt(
        ref_audio=hu_tao_ref_path,
        x_vector_only_mode=True
    )
    
    # 验证 DF 增强后的音频
    df_report = VoiceprintValidator.evaluate_audio_for_evolution(
        audio_path=df_output_path,
        model=model,
        ref_voiceprint=original_prompt
    )
    
    print("\n" + "="*80)
    print("DEEPFILTERNET ENHANCED AUDIO VALIDATION REPORT")
    print("="*80)
    print(f"   - HFE Ratio (>8kHz): {df_report['hfe_ratio']:.6f} (Limit < 0.005)")
    print(f"   - SFM Flatness:      {df_report['sfm']:.6f} (Limit < 0.060)")
    print(f"   - Similarity to Ref: {df_report['similarity']:.4f} (Limit >= 0.965)")
    print(f"   - Suitability Status: {df_report['suitable_for_evolution']}")
    print("="*80)

    # 4. 提取特征并生成害羞克隆音
    print("\n--- Step 4: Extracting voiceprint from DF audio and generating shy clone ---")
    df_prompt = model.create_voice_clone_prompt(
        ref_audio=df_output_path,
        x_vector_only_mode=True
    )
    
    text = "那个……请问，我、我可以坐在你旁边吗？我有点害怕……"
    instruct = "极其害羞、语速缓慢地颤抖说话"
    instruct_text = model._build_instruct_text(instruct)
    instruct_ids = model._tokenize_texts([instruct_text])
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="chinese",
        voice_clone_prompt=df_prompt,
        instruct_ids=instruct_ids
    )
    
    reproduced_gen_path = os.path.join(output_dir, "reproduced_shy_from_df.wav")
    sf.write(reproduced_gen_path, wavs[0], sr)
    print(f"Saved df-reproduced audio to: {reproduced_gen_path}")
    
    # Evaluate generated audio similarity
    gen_report = VoiceprintValidator.evaluate_audio_for_evolution(
        audio_path=reproduced_gen_path,
        model=model,
        ref_voiceprint=original_prompt
    )

    # 5. 音高与情感可塑性分析
    print("\n--- Step 5: Analyzing F0 Pitch Statistics ---")
    
    # Paths of other baselines for print comparison
    baseline_path = os.path.join(output_dir, "..", "style_experiments", "exp3_original_matched_shy.wav")
    unprocessed_path = os.path.join(output_dir, "reproduced_shy_from_unprocessed.wav")
    lp_path = os.path.join(output_dir, "reproduced_shy_cutoff_7800.wav")
    dac_path = os.path.join(output_dir, "reproduced_shy_from_dac.wav")
    
    # Run pitch analysis
    baseline_stats = analyze_pitch(baseline_path)
    unprocessed_stats = analyze_pitch(unprocessed_path)
    lp_stats = analyze_pitch(lp_path)
    dac_stats = analyze_pitch(dac_path)
    df_stats = analyze_pitch(reproduced_gen_path)
    
    print("\n" + "="*80)
    print("COMPARATIVE EMOTIONAL PLASTICITY SUMMARY")
    print("="*80)
    
    if baseline_stats:
        print(f"1. Golden Baseline (Original Hu Tao):")
        print(f"   - Pitch Std Dev: {baseline_stats['std_f0']:.2f} Hz | Pitch Range: {baseline_stats['range_f0']:.2f} Hz")
        
    if unprocessed_stats:
        print(f"2. Unprocessed Synth voiceprint (Flat):")
        print(f"   - Pitch Std Dev: {unprocessed_stats['std_f0']:.2f} Hz | Pitch Range: {unprocessed_stats['range_f0']:.2f} Hz")
        
    if lp_stats:
        print(f"3. 7.8kHz Low-pass Preprocessed (DSP):")
        print(f"   - Pitch Std Dev: {lp_stats['std_f0']:.2f} Hz | Pitch Range: {lp_stats['range_f0']:.2f} Hz")
        print(f"   - Generated Audio Similarity: 0.9553")
        
    if dac_stats:
        print(f"4. DAC Reconstructed (Neural Resynthesis):")
        print(f"   - Pitch Std Dev: {dac_stats['std_f0']:.2f} Hz | Pitch Range: {dac_stats['range_f0']:.2f} Hz")
        print(f"   - Generated Audio Similarity: 0.9496")

    if df_stats:
        print(f"5. DeepFilterNet Enhanced (Spectral Filtering):")
        print(f"   - Pitch Std Dev: {df_stats['std_f0']:.2f} Hz | Pitch Range: {df_stats['range_f0']:.2f} Hz")
        print(f"   - Generated Audio Similarity: {gen_report['similarity']:.4f}")
        
    print("="*80)

if __name__ == "__main__":
    main()
