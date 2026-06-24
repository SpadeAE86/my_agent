import os
import sys
import librosa
import numpy as np

def analyze_pitch(path):
    if not os.path.exists(path):
        return None
    
    # Load audio
    wav, sr = librosa.load(path, sr=24000)
    
    # Use YIN algorithm to estimate F0 (pitch)
    # Human speech pitch typically ranges from 50Hz to 400Hz (or higher for high-pitched female voices)
    fmin = 60
    fmax = 500
    f0 = librosa.yin(wav, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
    
    # Calculate RMS energy to filter out silence/unvoiced frames
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    
    # Align F0 and RMS frames (YIN hop_length defaults to 512, frame_length to 2048)
    min_len = min(len(f0), len(rms))
    f0 = f0[:min_len]
    rms = rms[:min_len]
    
    # Keep only voiced frames (where RMS energy > 0.015)
    voiced_f0 = f0[rms > 0.015]
    
    # Filter out outliers (e.g. octave jumps)
    if len(voiced_f0) > 0:
        q25, q75 = np.percentile(voiced_f0, [25, 75])
        iqr = q75 - q25
        # keep values within IQR range
        voiced_f0 = voiced_f0[(voiced_f0 >= q25 - 1.5 * iqr) & (voiced_f0 <= q75 + 1.5 * iqr)]
        
    if len(voiced_f0) == 0:
        return {
            "mean_f0": 0.0,
            "std_f0": 0.0,
            "range_f0": 0.0
        }
        
    return {
        "mean_f0": float(np.mean(voiced_f0)),
        "std_f0": float(np.std(voiced_f0)),
        "range_f0": float(np.max(voiced_f0) - np.min(voiced_f0))
    }

def main():
    base_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts"
    
    files = {
        "1. Golden Baseline (Original Hu Tao - Emotional)": os.path.join(base_dir, "style_experiments", "exp3_original_matched_shy.wav"),
        "2. Unprocessed Clone (Flat/No emotion)": os.path.join(base_dir, "style_reconstruction", "reproduced_shy_from_unprocessed.wav"),
        "3. Preprocessed Clone (Cutoff 7.8kHz)": os.path.join(base_dir, "style_reconstruction", "reproduced_shy_cutoff_7800.wav"),
        "4. Preprocessed Clone (Cutoff 9.0kHz)": os.path.join(base_dir, "style_reconstruction", "reproduced_shy_cutoff_9000.wav"),
        "5. Preprocessed Clone (Cutoff 10.0kHz)": os.path.join(base_dir, "style_reconstruction", "reproduced_shy_cutoff_10000.wav")
    }
    
    print("=" * 80)
    print("PITCH (F0) VARIANCE & EMOTIONAL PLASTICITY ANALYSIS")
    print("=" * 80)
    
    for label, path in files.items():
        print(f"\nAnalyzing: {label}")
        if not os.path.exists(path):
            print(f"   [ERROR] File not found at: {path}")
            continue
            
        stats = analyze_pitch(path)
        if stats:
            print(f"   - File Path: {os.path.basename(path)}")
            print(f"   - Mean Pitch (平均音高): {stats['mean_f0']:.2f} Hz")
            print(f"   - Pitch Std Dev (音高标准差): {stats['std_f0']:.2f} Hz  <-- (指标越低表示越死板/棒读)")
            print(f"   - Pitch Range (音高跨度):  {stats['range_f0']:.2f} Hz")
        else:
            print("   - Failed to analyze pitch.")
            
    print("=" * 80)

if __name__ == "__main__":
    main()
