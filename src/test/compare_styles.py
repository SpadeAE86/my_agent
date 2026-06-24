import os
import sys
import torch
import librosa
import numpy as np

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts import Qwen3TTSModel

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    style_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control\style_eval_1781979143"
    
    files = {
        "confident": os.path.join(style_dir, "confident.mp3"),
        "friendly": os.path.join(style_dir, "friendly.mp3"),
        "shy": os.path.join(style_dir, "shy.mp3")
    }
    
    # Check files exist
    for k, p in files.items():
        if not os.path.exists(p):
            print(f"Error: file not found for {k}: {p}")
            return

    print("Loading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(model_dir, device_map=device, dtype=torch.bfloat16)
    print("Model loaded.")

    # Load waveforms and extract embeddings
    embs = {}
    with torch.no_grad():
        for k, p in files.items():
            wav, sr = librosa.load(p, sr=24000)
            emb = model.model.extract_speaker_embedding(wav, 24000).cpu().float().numpy()
            embs[k] = emb

    # Compare similarities
    sim_conf_friend = cosine_similarity(embs["confident"], embs["friendly"])
    sim_conf_shy = cosine_similarity(embs["confident"], embs["shy"])
    sim_friend_shy = cosine_similarity(embs["friendly"], embs["shy"])

    print("\n" + "="*60)
    print("SIMILARITY BETWEEN DIFFERENT STYLES OF THE SAME SPEAKER")
    print("="*60)
    print(f"1. Confident style vs Friendly style:")
    print(f"   Similarity: {sim_conf_friend:.4f}")
    print(f"\n2. Confident style vs Shy style:")
    print(f"   Similarity: {sim_conf_shy:.4f}")
    print(f"\n3. Friendly style vs Shy style:")
    print(f"   Similarity: {sim_friend_shy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
