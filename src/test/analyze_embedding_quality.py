import os
import sys
import torch
import librosa
import numpy as np

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

def load_audio_24k(path):
    wav, sr = librosa.load(path, sr=24000)
    return wav

def high_freq_energy_ratio(wav, sr=24000, cutoff=8000):
    """计算高于 cutoff 频率的能量占总能量的比例"""
    # FFT
    fft_vals = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1/sr)
    
    total_energy = np.sum(fft_vals**2)
    high_freq_energy = np.sum(fft_vals[freqs >= cutoff]**2)
    
    return float(high_freq_energy / (total_energy + 1e-10))

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    
    hu_tao_ref_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    synthetic_ref_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\clone_with_instruct.wav"
    
    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    
    # Load audios
    print("\nLoading audio waveforms...")
    ref_wav = load_audio_24k(hu_tao_ref_path)
    synth_wav = load_audio_24k(synthetic_ref_path)
    
    # Extract speaker embeddings
    print("Extracting speaker embeddings...")
    with torch.no_grad():
        ref_emb = model.model.extract_speaker_embedding(ref_wav, 24000).cpu().float().numpy()
        synth_emb = model.model.extract_speaker_embedding(synth_wav, 24000).cpu().float().numpy()
        
    # 1. Analyze Audio High Frequency Energy (波形高频能量比)
    ref_hf_ratio = high_freq_energy_ratio(ref_wav)
    synth_hf_ratio = high_freq_energy_ratio(synth_wav)
    
    # 2. Analyze Embedding Stats (指纹向量统计量)
    ref_norm = np.linalg.norm(ref_emb)
    synth_norm = np.linalg.norm(synth_emb)
    
    ref_std = np.std(ref_emb)
    synth_std = np.std(synth_emb)
    
    ref_max = np.max(np.abs(ref_emb))
    synth_max = np.max(np.abs(synth_emb))
    
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS REPORT")
    print("="*60)
    print(f"1. Audio Source Quality (波形声学特征分析):")
    print(f"   - Human Original High-Freq Energy Ratio (>8kHz): {ref_hf_ratio:.6f}")
    print(f"   - Synthetic Audio High-Freq Energy Ratio (>8kHz): {synth_hf_ratio:.6f}")
    print(f"   (注: 高频能量比例降低说明发生了声码器重构损失，声音细节被抹平)")
    
    print(f"\n2. Speaker Embedding Statistics (指纹向量特征量对比):")
    print(f"   - Human Original Embedding L2 Norm (模长): {ref_norm:.4f}")
    print(f"   - Synthetic Embedding L2 Norm (模长):       {synth_norm:.4f}")
    print(f"   - Human Original Embedding Std (方差):      {ref_std:.4f}")
    print(f"   - Synthetic Embedding Std (方差):           {synth_std:.4f}")
    print(f"   - Human Original Embedding Max-Abs (峰值):  {ref_max:.4f}")
    print(f"   - Synthetic Embedding Max-Abs (峰值):       {synth_max:.4f}")
    print("="*60)
    
if __name__ == "__main__":
    main()
