import os
import sys
import torch
import soundfile as sf
import time

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts import Qwen3TTSModel

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-CustomVoice"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts"
    output_wav = os.path.join(output_dir, "qwen3_custom_voice_test.wav")
    
    print(f"Loading Qwen3-TTS-12Hz-1.7B-CustomVoice from: {model_dir}")
    t0 = time.time()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print(f"Model loaded successfully in {time.time() - t0:.2f} seconds.")
    
    speakers = model.get_supported_speakers()
    print(f"Supported speakers: {speakers}")
    
    # Pick a speaker. Standard ones in Qwen TTS are typically speaker names or IDs.
    # Let's print out what is returned, and pick the first one, or fall back to a default.
    speaker = "default"
    if speakers:
        speaker = speakers[0]
        print(f"Selecting speaker: {speaker}")
    else:
        print("No preset speakers returned. Using 'default'")
        
    target_text = "你好，我是使用昆仑三系列预设角色合成出来的声音。这里支持不同的角色和情感控制。"
    print(f"Synthesizing: {target_text}")
    
    t1 = time.time()
    wavs, sr = model.generate_custom_voice(
        text=target_text,
        speaker=speaker,
        language="chinese",
        instruct="Cheerful and warm tone" # we can pass an optional instruction/style
    )
    print(f"Inference completed in {time.time() - t1:.2f} seconds.")
    
    # Save output
    sf.write(output_wav, wavs[0], sr)
    print(f"Saved audio output to: {output_wav}")

if __name__ == "__main__":
    main()
