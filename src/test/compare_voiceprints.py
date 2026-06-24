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

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return float(dot_product / (norm_v1 * norm_v2))

def load_audio_24k(path):
    wav, sr = librosa.load(path, sr=24000)
    return wav

def main():
    model_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base"
    
    # Paths to files
    ref_audio_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    clone_with_instruct_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control\clone_with_instruct.wav"
    combined_full_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\unified_run\combined_full.mp3"
    
    # Check paths exist
    for p in [ref_audio_path, clone_with_instruct_path, combined_full_path]:
        if not os.path.exists(p):
            print(f"Error: Path does not exist: {p}")
            return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.")

    # Load audios
    print("\nLoading audio waveforms...")
    ref_wav = load_audio_24k(ref_audio_path)
    clone_wav = load_audio_24k(clone_with_instruct_path)
    combined_wav = load_audio_24k(combined_full_path)

    # Extract speaker embeddings
    print("Extracting speaker embeddings (voiceprints)...")
    with torch.no_grad():
        ref_emb = model.model.extract_speaker_embedding(ref_wav, 24000).cpu().float().numpy()
        clone_emb = model.model.extract_speaker_embedding(clone_wav, 24000).cpu().float().numpy()
        combined_emb = model.model.extract_speaker_embedding(combined_wav, 24000).cpu().float().numpy()

    # Calculate similarities
    sim_ref_to_clone = cosine_similarity(ref_emb, clone_emb)
    sim_ref_to_combined = cosine_similarity(ref_emb, combined_emb)
    sim_clone_to_combined = cosine_similarity(clone_emb, combined_emb)

    print("\n" + "="*60)
    print("VOICEPRINT SIMILARITY COMPARISON")
    print("="*60)
    print(f"1. Reference Audio vs clone_with_instruct.wav (Good Quality):")
    print(f"   Cosine Similarity: {sim_ref_to_clone:.4f}")
    print(f"\n2. Reference Audio vs combined_full.mp3 (Low Quality):")
    print(f"   Cosine Similarity: {sim_ref_to_combined:.4f}")
    print(f"\n3. clone_with_instruct.wav vs combined_full.mp3:")
    print(f"   Cosine Similarity: {sim_clone_to_combined:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
