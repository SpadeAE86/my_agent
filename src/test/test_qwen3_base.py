import os
import sys
import torch
import soundfile as sf
import time

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts import Qwen3TTSModel

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\tts_1780262424778.wav"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts"
    output_wav = os.path.join(output_dir, "qwen3_base_voice_clone_test.wav")
    
    print(f"Loading Qwen3-TTS-12Hz-1.7B-Base from: {model_dir}")
    t0 = time.time()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print(f"Model loaded successfully in {time.time() - t0:.2f} seconds.")
    
    target_text = "你好，我是使用昆仑三系列语音合成大模型克隆出来的虚拟音色，很高兴与你通话。"
    print(f"Synthesizing: {target_text}")
    
    t1 = time.time()
    # Using x_vector_only_mode=True to clone based purely on speaker embedding without requiring ref_text
    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="chinese",
        ref_audio=ref_audio,
        x_vector_only_mode=True
    )
    print(f"Inference completed in {time.time() - t1:.2f} seconds.")
    
    # Save output
    sf.write(output_wav, wavs[0], sr)
    print(f"Saved audio output to: {output_wav}")

if __name__ == "__main__":
    main()
