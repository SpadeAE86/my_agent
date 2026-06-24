import os
import time
import hashlib
import logging
import asyncio
import random
import requests
import soundfile as sf
from typing import Dict, Optional, Any
from core.voice.tts.tts_manager import BaseTTSEngine
from utils.obs_utils import upload_audio
from diskcache import Cache

log = logging.getLogger("dashscope_tts")

class DashScopeTTSEngine(BaseTTSEngine):
    def __init__(self, model_name: str = "cosyvoice-v3.5-plus"):
        self.model_name = model_name
        self._lock = asyncio.Lock()
        
        # Initialize a diskcache to persist voice_id mappings
        # This prevents generating a new voice_id for the same instruct prompt every time
        from config.config import MY_CONFIG, ENV
        try:
            cache_dir = os.path.join(MY_CONFIG['cache_config'][ENV]['cache_dir'], 'dashscope_voices')
        except Exception:
            cache_dir = "./final/cache/dashscope_voices"
        os.makedirs(cache_dir, exist_ok=True)
        self.voice_cache = Cache(cache_dir)

    def _get_or_create_voice_design(self, api_key: str, instruct: str) -> str:
        """
        Creates a custom designed voice using ModelStudio/DashScope customization API.
        Caches and returns the voice_id.
        """
        instruct_hash = hashlib.sha256(instruct.encode("utf-8")).hexdigest()
        cache_key = f"{self.model_name}:{instruct_hash}"
        cached_voice_id = self.voice_cache.get(cache_key)
        if cached_voice_id:
            log.info(f"DashScope Voice Design: Cache hit for key {cache_key} -> {cached_voice_id}")
            return cached_voice_id

        log.info(f"DashScope Voice Design: Cache miss. Creating custom voice for instruct: '{instruct}'")
        
        # Determine the target model and customized service type
        # CosyVoice: voice-enrollment customization
        # Qwen-TTS: qwen-voice-design customization
        is_qwen = "qwen" in self.model_name.lower()
        
        url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
        
        if is_qwen:
            payload = {
                "model": "qwen-voice-design",
                "input": {
                    "action": "create",
                    "target_model": self.model_name,
                    "preferred_name": f"qwenvd{''.join(c if c.isalpha() else chr(ord(c) - 48 + 103) for c in instruct_hash[:10])}",
                    "voice_prompt": instruct,
                    "preview_text": "大家好，欢迎来到我们的直播间！今天给大家推荐的这款产品真的超级好用。"
                },
                "parameters": {
                    "sample_rate": 24000,
                    "response_format": "wav"
                }
            }
        else:
            payload = {
                "model": "voice-enrollment",
                "input": {
                    "action": "create_voice",
                    "target_model": self.model_name,
                    "voice_prompt": instruct,
                    "preview_text": "各位听众朋友，大家好，欢迎收听由人工智能为您朗读的资讯。",
                    "prefix": "custom"
                },
                "parameters": {
                    "sample_rate": 24000,
                    "response_format": "wav"
                }
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"ModelStudio Voice Customization failed: {resp.status_code}, {resp.text}")
        
        res_data = resp.json()
        try:
            output_data = res_data.get("output", {})
            voice_id = output_data.get("voice_id") or output_data.get("voice")
            if not voice_id:
                raise KeyError("Neither 'voice_id' nor 'voice' found in output")
            # Cache the generated voice_id indefinitely
            self.voice_cache.set(cache_key, voice_id)
            log.info(f"Successfully created custom voice on DashScope: {voice_id}")
            return voice_id
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Failed to parse custom voice_id from response: {res_data}. Error: {e}")

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
        Synthesizes speech using DashScope online model and returns the OBS URL of the generated audio.
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable 'DASHSCOPE_API_KEY' is missing in .env file.")

        import dashscope
        dashscope.api_key = api_key
        
        instruct = kwargs.get("instruct") or kwargs.get("voice_design_instruct")
        
        # Check if the voice is a Qwen custom cloned or designed voice
        from models.sqlmodel.voice_cloned import VoiceCloned
        from models.sqlmodel.voice_designed import VoiceDesigned
        from infra.storage.mysql_connector import mysql_connector
        from sqlmodel import select
        
        is_qwen_cloned = False
        cloned_voice_id = None
        try:
            async with mysql_connector.session_scope() as session:
                stmt_clone = select(VoiceCloned).where(
                    (VoiceCloned.custom_speaker_id == voice_character) |
                    (VoiceCloned.voice_character == voice_character)
                )
                res_clone = await session.execute(stmt_clone)
                clone_rec = res_clone.scalar_one_or_none()
                if clone_rec and clone_rec.provider == "qwen":
                    is_qwen_cloned = True
                    cloned_voice_id = clone_rec.custom_speaker_id
                else:
                    stmt_design = select(VoiceDesigned).where(
                        (VoiceDesigned.custom_speaker_id == voice_character) |
                        (VoiceDesigned.voice_character == voice_character)
                    )
                    res_design = await session.execute(stmt_design)
                    design_rec = res_design.scalar_one_or_none()
                    if design_rec and design_rec.provider == "qwen":
                        is_qwen_cloned = True
                        cloned_voice_id = design_rec.custom_speaker_id
        except Exception as e:
            log.warning(f"Failed to check Qwen VoiceCloned/VoiceDesigned in DashScopeTTSEngine: {e}")

        model_to_use = "qwen3-tts-vc-2026-01-22" if is_qwen_cloned else self.model_name
        is_qwen = "qwen" in model_to_use.lower()
        
        async with self._lock:
            # 1. Obtain voice_id
            voice_id = None
            if is_qwen:
                if is_qwen_cloned:
                    voice_id = cloned_voice_id
                else:
                    # Map Chinese names to English parameter values for Qwen Online Voices
                    from models.qwen_online_voice_enums import qwen_online_voice_options
                    if voice_character in qwen_online_voice_options:
                        voice_id = qwen_online_voice_options[voice_character]
                    elif voice_character in qwen_online_voice_options.values():
                        voice_id = voice_character
                    else:
                        voice_id = voice_character if voice_character else "Cherry"
            else:
                if instruct:
                    # Custom Voice Design based on instruction prompt (for CosyVoice)
                    voice_id = await asyncio.to_thread(self._get_or_create_voice_design, api_key, instruct)
                else:
                    # Fallback or pre-configured voice_character voice_id
                    # If voice_character is passed and matches custom voice ID format, use it directly
                    # For standard voices, map default values
                    if voice_character.startswith("voice_") or voice_character.startswith("qwen_") or len(voice_character) > 20:
                        voice_id = voice_character
                    else:
                        # Default voice character mapping for cosyvoice-v3.5-plus
                        # E.g. cosyvoice preset speakers
                        voice_id = voice_character if voice_character else "vivi"

            # 2. Call Synthesis
            log.info(f"Calling DashScope SpeechSynthesizer. Model: {model_to_use}, Voice: {voice_id}, Text: '{text[:20]}...'")
            
            t0 = time.time()
            if is_qwen:
                # Qwen-TTS uses MultiModalConversation endpoint
                call_kwargs = {
                    "model": model_to_use,
                    "api_key": api_key,
                    "text": text,
                    "voice": voice_id,
                    "stream": False
                }
                if instruct:
                    call_kwargs["instructions"] = instruct
                    call_kwargs["optimize_instructions"] = True
                    
                response = await asyncio.to_thread(
                    dashscope.MultiModalConversation.call,
                    **call_kwargs
                )
                # Check status code immediately
                if not hasattr(response, 'status_code') or response.status_code != 200:
                    err_msg = getattr(response, 'message', 'Unknown DashScope Error')
                    err_code = getattr(response, 'code', 'UnknownCode')
                    req_id = getattr(response, 'request_id', 'N/A')
                    raise RuntimeError(f"DashScope Qwen synthesis failed: Status {response.status_code if hasattr(response, 'status_code') else 'Unknown'}, Code: {err_code}, Message: {err_msg}, RequestID: {req_id}")
                
                # Parse output audio
                # Note: Qwen3-TTS API output details vary, but generally return audio bytes in output
                try:
                    log.info(f"Qwen MultiModalResponse: {response}")
                    audio_info = response.get("output", {}).get("audio", {})
                    audio_bytes = None
                    if audio_info:
                        if audio_info.get("data"):
                            import base64
                            audio_bytes = base64.b64decode(audio_info["data"])
                        elif audio_info.get("url"):
                            audio_url = audio_info["url"]
                            log.info(f"Downloading Qwen3 synthesized audio from {audio_url}")
                            dl_resp = await asyncio.to_thread(requests.get, audio_url)
                            if dl_resp.status_code == 200:
                                audio_bytes = dl_resp.content
                            else:
                                raise ValueError(f"Failed to download audio from OSS: {dl_resp.status_code}")
                    
                    # Fallback to choices-based structure if needed
                    if not audio_bytes:
                        content = response.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", [])
                        for item in content:
                            if isinstance(item, dict) and "audio" in item:
                                import base64
                                audio_bytes = base64.b64decode(item["audio"])
                                break
                    
                    if not audio_bytes:
                        raise ValueError(f"No audio found in Qwen3 response: {response}")
                except Exception as e:
                    raise RuntimeError(f"Failed to parse Qwen3-TTS MultiModalConversation response: {e}")
            else:
                # CosyVoice uses SpeechSynthesizer WebSocket client (supports streaming/delay metrics)
                from dashscope.audio.tts_v2 import SpeechSynthesizer
                # Setup synthesizer
                synthesizer = SpeechSynthesizer(model=model_to_use, voice=voice_id)
                audio_bytes = await asyncio.to_thread(synthesizer.call, text)
                
                try:
                    delay = synthesizer.get_first_package_delay()
                    req_id = synthesizer.get_last_request_id()
                    log.info(f"[Metric] DashScope Synthesis. RequestID: {req_id}, TTFT (首包延迟): {delay}ms")
                except Exception as metric_err:
                    log.warning(f"Failed to extract DashScope metrics: {metric_err}")

        # Save output audio to local temp MP3 and upload to OBS
        project_id = f"dashscope_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        temp_dir = os.path.join("./final", project_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            final_mp3_path = os.path.join(temp_dir, f"voice_{project_id}.mp3")
            
            # Check if input is WAV (starts with RIFF...WAVE)
            is_wav = len(audio_bytes) > 12 and audio_bytes[0:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
            
            if is_wav:
                temp_wav_path = os.path.join(temp_dir, "temp_raw.wav")
                with open(temp_wav_path, "wb") as f:
                    f.write(audio_bytes)
                
                # Transcode WAV to MP3 using FFmpeg
                cmd = ["ffmpeg", "-y", "-i", temp_wav_path, "-acodec", "libmp3lame", "-ab", "128k", final_mp3_path]
                log.info(f"Transcoding Dashscope WAV to MP3: {' '.join(cmd)}")
                import subprocess
                res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
                if res.returncode != 0:
                    err_msg = res.stderr.decode('utf-8', errors='ignore')
                    log.error(f"FFmpeg transcode to MP3 failed: {err_msg}. Falling back to direct raw write.")
                    with open(final_mp3_path, "wb") as f:
                        f.write(audio_bytes)
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except:
                        pass
            else:
                # Write binary audio data directly (already MP3 or other formats)
                with open(final_mp3_path, "wb") as f:
                    f.write(audio_bytes)
            
            # Upload to OBS
            audio_url = await upload_audio(final_mp3_path, project_id=project_id)
            log.info(f"Generated DashScope voice URL: {audio_url}")
            return audio_url
            
        finally:
            # Clean up temp files after short delay
            async def delayed_delete(path: str, delay: int = 300):
                await asyncio.sleep(delay)
                try:
                    import shutil
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    log.warning(f"Failed to clean up DashScope temp path {path}: {e}")
            
            asyncio.create_task(delayed_delete(temp_dir, delay=300))
