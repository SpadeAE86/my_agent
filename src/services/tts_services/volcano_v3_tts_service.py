import os
import time
import random
import asyncio
import subprocess
import logging
from core.voice.tts.tts_manager import BaseTTSEngine
from core.voice.tts.volcano_v3_tts import VolcanoV3TTSEngine
from core.voice.tts.volcano_v3_bidirectional import VolcanoV3BidirectionalEngine
from services.volcovoice_service import VolcoVoiceService
from utils.obs_utils import upload_audio
from sdk.volcano_protocols import MsgType, EventType

log = logging.getLogger("volcano_v3_tts_service")


class VolcanoV3TTSServiceEngine(BaseTTSEngine):
    def __init__(self):
        self.engine = VolcanoV3TTSEngine()

    async def warm_up(self) -> None:
        if hasattr(self.engine, "warm_up"):
            await self.engine.warm_up()

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
        # Clean text
        cleaned_text = VolcoVoiceService.clean_markdown(text)
        if not cleaned_text:
            cleaned_text = "没有可朗读的文本。"

        return await self.engine.generate_voice(
            text=cleaned_text,
            voice_character=voice_character,
            speed=speed,
            volume=volume,
            emotion=emotion,
            disable_segmentation=disable_segmentation,
            **kwargs
        )

    async def generate_voice_stream(
        self,
        text: str,
        voice_character: str,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        **kwargs
    ):
        # Clean text
        cleaned_text = VolcoVoiceService.clean_markdown(text)
        if not cleaned_text:
            cleaned_text = "没有可朗读的文本。"

        async for chunk in self.engine.generate_voice_stream(
            text=cleaned_text,
            voice_character=voice_character,
            speed=speed,
            volume=volume,
            emotion=emotion,
            **kwargs
        ):
            yield chunk


class VolcanoV3BidirectionalServiceEngine(BaseTTSEngine):
    def __init__(self):
        self.engine = VolcanoV3BidirectionalEngine()

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
        # Clean text
        cleaned_text = VolcoVoiceService.clean_markdown(text)
        if not cleaned_text:
            cleaned_text = "没有可朗读的文本。"

        # Connect WebSocket
        await self.engine.connect(voice_character, **kwargs)
        
        session_id = f"sess_bi_{int(time.time() * 1000)}"
        instruct = kwargs.get("instruct") or kwargs.get("voice_design_instruct")
        
        # Start Dialogue Session
        bi_kwargs = kwargs.copy()
        bi_kwargs.pop("instruct", None)
        bi_kwargs.pop("voice_design_instruct", None)
        await self.engine.start_session(
            session_id=session_id,
            voice_character=voice_character,
            speed=speed,
            volume=volume,
            instruct=instruct,
            **bi_kwargs
        )
        
        # Send synthesis request
        t_start = time.time()
        first_packet_time = None
        await self.engine.send_text(cleaned_text, session_id)
        
        # Accumulate audio data stream
        audio_data = bytearray()
        
        try:
            async for msg in self.engine.receive_messages():
                if msg.type == MsgType.AudioOnlyServer:
                    if first_packet_time is None:
                        first_packet_time = time.time()
                        ttft = int((first_packet_time - t_start) * 1000)
                        log.info(f"[Metric] Volcano V3 Bidirectional. SessionID: {session_id}, TTFT (首包延迟): {ttft}ms")
                    audio_data.extend(msg.payload)
                    # Negative sequence indicates end of audio stream
                    if msg.sequence < 0:
                        break
                elif msg.type == MsgType.Error:
                    raise RuntimeError(f"Volcano V3 bidirectional WebSocket error: {msg.error_code}")
                
                # Double-safety check: EventType.TTSEnded or SessionFinished indicates end of current synthesis
                if msg.type == MsgType.FullServerResponse and msg.event in (EventType.TTSEnded, EventType.SessionFinished):
                    break
        finally:
            # Cleanup session and connection
            try:
                await self.engine.finish_session(session_id)
            except Exception as e:
                log.warning(f"Failed to finish session cleanly: {e}")
            
            try:
                await self.engine.close()
            except Exception as e:
                log.warning(f"Failed to close bidirectional WebSocket cleanly: {e}")

        if not audio_data:
            raise RuntimeError("No audio data received via Volcano V3 bidirectional WebSocket.")

        # Save, transcode, and upload
        project_id = f"volco_v3_bi_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        temp_dir = os.path.join("./final", project_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            raw_path = os.path.join(temp_dir, f"raw_{project_id}.mp3")
            with open(raw_path, "wb") as f:
                f.write(audio_data)
                
            final_mp3_path = os.path.join(temp_dir, f"voice_{project_id}.mp3")
            
            # Transcode
            cmd = ["ffmpeg", "-y", "-i", raw_path, "-c:a", "libmp3lame", final_mp3_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            audio_url = await upload_audio(final_mp3_path, project_id=project_id)
            return audio_url
        finally:
            # Clean up temp folder
            async def delayed_delete(path: str, delay: int = 300):
                await asyncio.sleep(delay)
                try:
                    import shutil
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.isfile(path):
                        os.remove(path)
                except Exception as ex:
                    log.warning(f"Failed to clean up V3 bidirectional temp path {path}: {ex}")

            asyncio.create_task(delayed_delete(temp_dir, delay=300))
