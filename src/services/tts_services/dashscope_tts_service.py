from core.voice.tts.tts_manager import BaseTTSEngine
from core.voice.tts.dashscope_tts import DashScopeTTSEngine
from services.volcovoice_service import VolcoVoiceService

class DashScopeServiceEngine(BaseTTSEngine):
    def __init__(self, model_name: str = "cosyvoice-v3.5-plus"):
        self.engine = DashScopeTTSEngine(model_name=model_name)

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


class QwenOnlineServiceEngine(BaseTTSEngine):
    def __init__(self, model_name: str = "qwen3-tts-instruct-flash"):
        self.engine = DashScopeTTSEngine(model_name=model_name)

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
