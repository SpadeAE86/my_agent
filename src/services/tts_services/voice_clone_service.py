import logging
from typing import Optional

from models.sqlmodel.voice_cloned import VoiceCloned
from models.sqlmodel.voice_designed import VoiceDesigned

# Import sub-modules
from services.tts_services.clone_utils import upload_original_audio
from services.tts_services.volcano_clone_helper import VolcanoCloneHelper
from services.tts_services.qwen_clone_helper import QwenCloneHelper

log = logging.getLogger("voice_clone_service")

class VoiceCloneService:
    async def upload_original_audio(self, audio_bytes: bytes, audio_format: str, prefix: str = "ref") -> str:
        return await upload_original_audio(audio_bytes, audio_format, prefix)

    async def clone_voice(
        self,
        voice_character: str,
        audio_bytes: bytes,
        audio_format: str,
        prompt_text: Optional[str] = None,
        language: int = 0,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None,
        demo_text: Optional[str] = None
    ) -> VoiceCloned:
        return await VolcanoCloneHelper.clone_voice(
            voice_character=voice_character,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            prompt_text=prompt_text,
            language=language,
            tag=tag,
            note=note,
            age_type=age_type,
            sex=sex,
            demo_text=demo_text
        )

    async def poll_and_finalize_clone(self, voice_id: int, custom_speaker_id: str):
        return await VolcanoCloneHelper.poll_and_finalize_clone(voice_id, custom_speaker_id)

    async def delete_cloned_voice(self, voice_id: int):
        return await VolcanoCloneHelper.delete_cloned_voice(voice_id)

    async def query_and_update_status(self, voice_id: int) -> VoiceCloned:
        return await VolcanoCloneHelper.query_and_update_status(voice_id)

    async def sync_volcano_voices(self, force_refresh: bool = False):
        return await VolcanoCloneHelper.sync_volcano_voices(force_refresh)

    async def upgrade_cloned_voice(self, voice_id: int) -> VoiceCloned:
        return await VolcanoCloneHelper.upgrade_cloned_voice(voice_id)

    async def upgrade_designed_voice(self, voice_id: int) -> VoiceDesigned:
        return await VolcanoCloneHelper.upgrade_designed_voice(voice_id)

    async def query_and_update_designed_status(self, voice_id: int) -> VoiceDesigned:
        return await VolcanoCloneHelper.query_and_update_designed_status(voice_id)

    async def clone_voice_qwen(
        self,
        voice_character: str,
        audio_bytes: bytes,
        audio_format: str,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None,
        prompt_text: Optional[str] = None,
        demo_text: Optional[str] = None
    ) -> VoiceCloned:
        return await QwenCloneHelper.clone_voice_qwen(
            voice_character=voice_character,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            tag=tag,
            note=note,
            age_type=age_type,
            sex=sex,
            prompt_text=prompt_text,
            demo_text=demo_text
        )

    async def _generate_qwen_demo_audio(self, voice_id: int, custom_speaker_id: str, demo_text: Optional[str] = None):
        return await QwenCloneHelper._generate_qwen_demo_audio(voice_id, custom_speaker_id, demo_text)

    async def clone_voice_offline(
        self,
        voice_character: str,
        audio_bytes: bytes,
        audio_format: str,
        prompt_text: Optional[str] = None,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None
    ) -> VoiceCloned:
        return await QwenCloneHelper.clone_voice_offline(
            voice_character=voice_character,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            prompt_text=prompt_text,
            tag=tag,
            note=note,
            age_type=age_type,
            sex=sex
        )

    async def sync_qwen_voices(self, force_refresh: bool = False):
        return await QwenCloneHelper.sync_qwen_voices(force_refresh)

    async def _generate_qwen_design_demo_audio(self, voice_id: int, custom_speaker_id: str):
        return await QwenCloneHelper._generate_qwen_design_demo_audio(voice_id, custom_speaker_id)


voice_clone_service = VoiceCloneService()
