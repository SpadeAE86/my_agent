from core.voice.tts.tts_manager import BaseTTSEngine
from core.voice.tts.qwen3tts import Qwen3TTSEngine
from services.volcovoice_service import VolcoVoiceService

class QwenTTSEngine(BaseTTSEngine):
    def __init__(self):
        self.engine = Qwen3TTSEngine()

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
        # Pre-process text to remove markdown
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
        
    def offload_models(self):
        self.engine.offload_models()
