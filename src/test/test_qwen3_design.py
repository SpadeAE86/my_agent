import os
import sys
import torch
import soundfile as sf
import time

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts import Qwen3TTSModel

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts"
    output_wav = os.path.join(output_dir, "qwen3_design_voice_test.wav")
    
    print(f"Loading Qwen3-TTS-12Hz-1.7B-VoiceDesign from: {model_dir}")
    t0 = time.time()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print(f"Model loaded successfully in {time.time() - t0:.2f} seconds.")
    
    instruct = "一个声音有些沙哑、语调活泼的年轻女孩"
    target_text = "你好，我是由你用文字描述设计出来的全新音色。现在我正在用有些沙哑、活泼的声音说话。"
    print(f"Instruction: {instruct}")
    print(f"Synthesizing: {target_text}")
    
    t1 = time.time()
    wavs, sr = model.generate_voice_design(
        text=target_text,
        instruct=instruct,
        language="chinese"
    )
    print(f"Inference completed in {time.time() - t1:.2f} seconds.")
    
    # Save output
    sf.write(output_wav, wavs[0], sr)
    print(f"Saved audio output to: {output_wav}")

if __name__ == "__main__":
    main()
