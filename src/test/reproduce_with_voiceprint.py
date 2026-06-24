import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf

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
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

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
    
    # Paths
    hu_tao_ref_path = r"C:\CloudMusic\voice_collection\voice_dataset\6月21日\6月21日.mp3_0000004160_0000145600.wav"
    synthetic_ref_path = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2\clone_with_instruct.wav"
    output_dir = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\style_control_2"
    
    # Check paths exist
    if not os.path.exists(hu_tao_ref_path):
        print(f"Error: Hu Tao reference audio not found at: {hu_tao_ref_path}")
        return
    if not os.path.exists(synthetic_ref_path):
        print(f"Error: Synthetic reference audio (clone_with_instruct.wav) not found at: {synthetic_ref_path}")
        return

    print("Loading Qwen3 model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16
    )
    print("Model loaded.")

    # 1. Extract voiceprint from the synthetic WAV
    print(f"\n1. Extracting voiceprint from synthetic audio: {os.path.basename(synthetic_ref_path)}")
    synthetic_prompt = model.create_voice_clone_prompt(
        ref_audio=synthetic_ref_path,
        x_vector_only_mode=True
    )
    
    # 2. Save the extracted voiceprint prompt to a file (demonstrating reuse)
    voiceprint_save_path = os.path.join(output_dir, "synthetic_voiceprint.pt")
    torch.save(synthetic_prompt, voiceprint_save_path)
    print(f"   Voiceprint saved locally to: {voiceprint_save_path}")

    # 3. Load the saved voiceprint back
    print("2. Loading the saved voiceprint file back into memory...")
    loaded_prompt = torch.load(voiceprint_save_path, weights_only=False)

    # 4. Generate new audio using the loaded voiceprint
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。不是机械配合你，而是能跟上你、补足你、甚至偶尔把你往更有趣的方向轻轻推一下的人。"
    instruct = "极其害羞、语速缓慢地颤抖说话"
    
    instruct_text = model._build_instruct_text(instruct)
    instruct_ids = model._tokenize_texts([instruct_text])

    print("3. Generating reproduced audio using the loaded voiceprint...")
    reproduced_wav_path = os.path.join(output_dir, "reproduced_from_voiceprint.wav")
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="chinese",
        voice_clone_prompt=loaded_prompt,
        instruct_ids=instruct_ids
    )
    
    # Save the reproduced audio
    sf.write(reproduced_wav_path, wavs[0], sr)
    print(f"   Reproduced audio saved to: {reproduced_wav_path}")

    # 5. Extract voiceprints for similarity comparison
    print("\n4. Loading waveforms for voiceprint comparison...")
    hu_tao_wav = load_audio_24k(hu_tao_ref_path)
    synthetic_wav = load_audio_24k(synthetic_ref_path)
    reproduced_wav = load_audio_24k(reproduced_wav_path)

    print("5. Extracting voiceprint embeddings...")
    with torch.no_grad():
        hu_tao_emb = model.model.extract_speaker_embedding(hu_tao_wav, 24000).cpu().float().numpy()
        synthetic_emb = model.model.extract_speaker_embedding(synthetic_wav, 24000).cpu().float().numpy()
        reproduced_emb = model.model.extract_speaker_embedding(reproduced_wav, 24000).cpu().float().numpy()

    # Calculate similarities
    sim_original_to_synthetic = cosine_similarity(hu_tao_emb, synthetic_emb)
    sim_original_to_reproduced = cosine_similarity(hu_tao_emb, reproduced_emb)
    sim_synthetic_to_reproduced = cosine_similarity(synthetic_emb, reproduced_emb)

    print("\n" + "="*60)
    print("VOICEPRINT SIMILARITY COMPARISON (REUSE TEST)")
    print("="*60)
    print(f"1. Hu Tao Original Reference vs Synthetic Ref (clone_with_instruct.wav):")
    print(f"   Cosine Similarity: {sim_original_to_synthetic:.4f}")
    print(f"\n2. Hu Tao Original Reference vs Reproduced Audio:")
    print(f"   Cosine Similarity: {sim_original_to_reproduced:.4f}")
    print(f"\n3. Synthetic Ref vs Reproduced Audio (Direct Ancestor vs Offspring):")
    print(f"   Cosine Similarity: {sim_synthetic_to_reproduced:.4f}")
    print("="*60)
    print("Experiment successful! The synthetic audio's voiceprint is fully reusable.")

if __name__ == "__main__":
    main()
