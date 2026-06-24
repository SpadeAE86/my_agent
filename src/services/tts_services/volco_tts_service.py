from core.voice.tts.tts_manager import BaseTTSEngine
from services.volcovoice_service import VolcoVoiceService

class VolcoTTSEngine(BaseTTSEngine):
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
        return await VolcoVoiceService.generate_voice(
            text=text,
            voice_character=voice_character,
            speed=speed,
            volume=volume,
            emotion=emotion,
            disable_segmentation=disable_segmentation
        )
