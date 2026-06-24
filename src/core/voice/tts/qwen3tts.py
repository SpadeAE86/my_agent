import os
import sys
import time
import torch
import logging

try:
    import flash_attn_3
    import flash_attn_interface
    sys.modules['flash_attn'] = flash_attn_3
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
except ImportError:
    pass
import asyncio
import random
import soundfile as sf
import subprocess
from typing import Dict, Optional, Any
from core.voice.tts.tts_manager import BaseTTSEngine
from utils.obs_utils import upload_audio
from qwen_tts import Qwen3TTSModel

log = logging.getLogger("qwen3tts")

class Qwen3TTSEngine(BaseTTSEngine):
    def __init__(self, model_parent_dir: Optional[str] = None):
        if not model_parent_dir:
            # check config
            try:
                from config.config import MY_CONFIG
                model_parent_dir = MY_CONFIG.get("audio", {}).get("Qwen3", {}).get(
                    "model_parent_dir", r"C:\Users\admin\.cache\modelscope\hub\models\Qwen"
                )
            except Exception:
                model_parent_dir = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen"
        
        self.model_parent_dir = model_parent_dir
        self._models = {}
        self._lock = asyncio.Lock()
        
    def _get_model_path(self, model_name: str) -> str:
        return os.path.join(self.model_parent_dir, model_name)

    def load_model(self, model_type: str) -> Qwen3TTSModel:
        """
        model_type can be 'base', 'custom', 'design'
        """
        if model_type in self._models:
            return self._models[model_type]
            
        model_names = {
            "base": "Qwen3-TTS-12Hz-1.7B-Base",
            "custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        }
        
        model_name = model_names.get(model_type)
        if not model_name:
            raise ValueError(f"Unknown Qwen3 model type: {model_type}")
            
        model_path = self._get_model_path(model_name)
        log.info(f"Loading local Qwen3 model '{model_type}' from {model_path} ...")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        t0 = time.time()
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16
        )
        log.info(f"Successfully loaded {model_name} in {time.time() - t0:.2f} seconds.")
        
        self._models[model_type] = model
        return model

    def offload_models(self):
        """
        Unload all cached models from VRAM and free torch cache.
        """
        log.info("Offloading all Qwen3 models from VRAM...")
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcode_audio(self, source_audio: str, output_audio: str, file_format: str) -> str:
        codec = "libmp3lame" if file_format == "mp3" else "pcm_s16le"
        cmd = [
            "ffmpeg", "-y", "-i", source_audio, "-c:a", codec, output_audio
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            log.error(f"FFmpeg transcode failed in Qwen3TTSEngine: {e}")
            raise RuntimeError(f"音频转码失败: {e}")
        return output_audio

    async def generate_voice(
        self,
        text: str,
        voice_character: str,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        disable_segmentation: bool = True,
        **kwargs
    ) -> str:
        """
        Unified implementation for Qwen3-TTS generation.
        """
        # Determine the model type to use
        # 1. Voice Clone / Base Mode
        ref_audio = kwargs.get("ref_audio")
        instruct = kwargs.get("instruct") or kwargs.get("voice_design_instruct")
        
        # If ref_audio is a remote URL, download it locally first
        local_ref_audio_path = None
        if ref_audio and (ref_audio.startswith("http://") or ref_audio.startswith("https://")):
            try:
                import httpx
                log.info(f"Downloading reference audio from URL: {ref_audio}")
                ref_audio_dir = os.path.join("./final", f"ref_download_{int(time.time() * 1000)}")
                os.makedirs(ref_audio_dir, exist_ok=True)
                ref_audio_filename = f"ref_{int(time.time() * 1000)}.wav"
                local_ref_audio_path = os.path.join(ref_audio_dir, ref_audio_filename)
                
                with open(local_ref_audio_path, "wb") as f:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.get(ref_audio)
                        resp.raise_for_status()
                        f.write(resp.content)
                log.info(f"Successfully downloaded reference audio to: {local_ref_audio_path}")
                ref_audio = local_ref_audio_path
            except Exception as e:
                log.error(f"Failed to download reference audio from URL '{ref_audio}': {e}", exc_info=True)
                raise RuntimeError(f"下载参考音频失败: {e}")

        # We run the actual heavy GPU loading & inference under a thread/lock to avoid race conditions on GPU.
        async with self._lock:
            if ref_audio:
                # Base model (Voice cloning)
                model = self.load_model("base")
                x_vector_only_mode = kwargs.get("x_vector_only_mode", True)
                ref_text = kwargs.get("ref_text")
                
                clone_kwargs = {}
                if instruct:
                    # Tokenize the instruction text
                    instruct_text = model._build_instruct_text(instruct)
                    instruct_ids = model._tokenize_texts([instruct_text])
                    clone_kwargs["instruct_ids"] = instruct_ids
                
                log.info(f"Running Qwen3 Voice Clone. Target text: '{text[:20]}...' (with instruct: {instruct if instruct else 'none'})")
                # Run sync generate in thread pool
                wavs, sr = await asyncio.to_thread(
                    model.generate_voice_clone,
                    text=text,
                    language="chinese",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **clone_kwargs
                )
            elif instruct:
                # Voice Design model
                model = self.load_model("design")
                log.info(f"Running Qwen3 Voice Design with instruct: '{instruct}'. Target text: '{text[:20]}...'")
                wavs, sr = await asyncio.to_thread(
                    model.generate_voice_design,
                    text=text,
                    instruct=instruct,
                    language="chinese"
                )
            else:
                # CustomVoice model (Preset characters)
                model = self.load_model("custom")
                supported_speakers = model.get_supported_speakers() or []
                
                # Check if voice_character is supported, case insensitively
                speaker = "vivian"  # default fallback
                for s in supported_speakers:
                    if s.lower() == voice_character.lower():
                        speaker = s
                        break
                
                log.info(f"Running Qwen3 Custom Voice. Speaker: {speaker}. Target text: '{text[:20]}...'")
                wavs, sr = await asyncio.to_thread(
                    model.generate_custom_voice,
                    text=text,
                    speaker=speaker,
                    language="chinese"
                )
                
        # Save output and transcode/upload
        project_id = f"qwen_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        temp_dir = os.path.join("./final", project_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            raw_wav_path = os.path.join(temp_dir, f"raw_{project_id}.wav")
            # Save raw wav output array
            await asyncio.to_thread(sf.write, raw_wav_path, wavs[0], sr)
            
            # Transcode to mp3 for smaller size and compatibility
            final_mp3_path = os.path.join(temp_dir, f"voice_{project_id}.mp3")
            await asyncio.to_thread(self.transcode_audio, raw_wav_path, final_mp3_path, "mp3")
            
            # Upload to OBS
            audio_url = await upload_audio(final_mp3_path, project_id=project_id)
            log.info(f"Generated Qwen3 voice URL: {audio_url}")
            return audio_url
            
        finally:
            if local_ref_audio_path and os.path.exists(local_ref_audio_path):
                try:
                    ref_audio_dir = os.path.dirname(local_ref_audio_path)
                    import shutil
                    shutil.rmtree(ref_audio_dir)
                    log.info(f"Cleaned up downloaded reference audio directory: {ref_audio_dir}")
                except Exception as ex:
                    log.warning(f"Failed to clean up downloaded ref audio path {local_ref_audio_path}: {ex}")

            # Clean up temp files after a short delay
            async def delayed_delete(path: str, delay: int = 300):
                await asyncio.sleep(delay)
                try:
                    import shutil
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    log.warning(f"Failed to clean up Qwen3 temp path {path}: {e}")
            
            asyncio.create_task(delayed_delete(temp_dir, delay=300))
