from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Any, AsyncGenerator
import os

log = logging.getLogger("tts_manager")

class BaseTTSEngine(ABC):
    @abstractmethod
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
        Synthesize text to speech and upload the generated audio to OSS/OBS.
        Returns the public URL of the audio.
        """
        pass


class TTSManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TTSManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._engines = {}
            cls._instance._default_engine_name = "volcano"
            cls._instance._engine_cache = {}
        return cls._instance

    def register_engine(self, name: str, engine: BaseTTSEngine):
        self._engines[name] = engine
        log.info(f"Registered TTS Engine: {name}")

    def get_engine(self, name: str) -> Optional[BaseTTSEngine]:
        return self._engines.get(name)

    @property
    def default_engine_name(self) -> str:
        return self._default_engine_name

    @default_engine_name.setter
    def default_engine_name(self, name: str):
        if name in self._engines:
            self._default_engine_name = name
            log.info(f"Default TTS Engine set to: {name}")
        else:
            raise ValueError(f"Engine '{name}' is not registered.")

    async def _resolve_engine_and_kwargs(self, voice_character: str, kwargs: dict) -> str:
        if hasattr(self, "_engine_cache") and voice_character in self._engine_cache:
            engine_name, extra_kwargs = self._engine_cache[voice_character]
            for k, v in extra_kwargs.items():
                if k not in kwargs:
                    kwargs[k] = v
            return engine_name

        engine_name = None
        extra_kwargs = {}
        try:
            from models.sqlmodel.voice_timbre import VoiceTimbre
            from infra.storage.mysql_connector import mysql_connector
            from sqlmodel import select
            
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceTimbre).where(VoiceTimbre.voice_character == voice_character, VoiceTimbre.is_enabled == True)
                result = await session.execute(stmt)
                timbre = result.scalar_one_or_none()
                if timbre:
                    if timbre.provider == "qwen":
                        engine_name = "qwen3_online"
                    elif timbre.provider == "volcano":
                        code_lower = timbre.voice_code.lower()
                        if "uranus" in code_lower or "saturn" in code_lower or "jupiter" in code_lower:
                            engine_name = "volcano_v3"
                        else:
                            engine_name = "volcano"
                else:
                    from models.sqlmodel.voice_cloned import VoiceCloned
                    stmt_clone = select(VoiceCloned).where(
                        (VoiceCloned.custom_speaker_id == voice_character) |
                        (VoiceCloned.voice_character == voice_character)
                    )
                    res_clone = await session.execute(stmt_clone)
                    clone_rec = res_clone.scalar_one_or_none()
                    if clone_rec:
                        if not getattr(clone_rec, "is_online", True):
                            engine_name = "qwen3_local"
                            if "ref_audio" not in kwargs:
                                extra_kwargs["ref_audio"] = clone_rec.demo_audio_url
                            if "ref_text" not in kwargs:
                                extra_kwargs["ref_text"] = clone_rec.prompt_text
                        else:
                            if clone_rec.provider == "qwen":
                                engine_name = "qwen3_online"
                            else:
                                engine_name = "volcano_v3"
                    else:
                        from models.sqlmodel.voice_designed import VoiceDesigned
                        stmt_design = select(VoiceDesigned).where(
                            (VoiceDesigned.custom_speaker_id == voice_character) |
                            (VoiceDesigned.voice_character == voice_character)
                        )
                        res_design = await session.execute(stmt_design)
                        design_rec = res_design.scalar_one_or_none()
                        if design_rec:
                            if not getattr(design_rec, "is_online", True):
                                engine_name = "qwen3_local"
                                if "instruct" not in kwargs and "voice_design_instruct" not in kwargs:
                                    extra_kwargs["instruct"] = design_rec.text_prompt
                            else:
                                engine_name = "volcano_v3"
        except Exception as e:
            log.warning(f"Failed to auto-detect TTS engine for '{voice_character}' from database: {e}")

        resolved_name = engine_name or self._default_engine_name
        if engine_name and hasattr(self, "_engine_cache"):
            self._engine_cache[voice_character] = (resolved_name, extra_kwargs)
            
        for k, v in extra_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        return resolved_name

    def clear_cache(self):
        if hasattr(self, "_engine_cache"):
            self._engine_cache.clear()
            log.info("Cleared TTS engine resolution cache.")

    async def warm_up_engine(self, engine_name: str):
        engine = self.get_engine(engine_name)
        if engine and hasattr(engine, "warm_up"):
            try:
                await engine.warm_up()
            except Exception as e:
                log.warning(f"Failed to warm up engine '{engine_name}': {e}")

    async def generate_voice(
        self,
        text: str,
        voice_character: str,
        engine_name: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        disable_segmentation: bool = True,
        **kwargs
    ) -> str:
        if not engine_name:
            engine_name = await self._resolve_engine_and_kwargs(voice_character, kwargs)

        name = engine_name or self._default_engine_name
        engine = self.get_engine(name)
        if not engine:
            if "volcano" in self._engines:
                log.warning(f"TTS Engine '{name}' is not registered. Falling back to 'volcano'")
                engine = self._engines["volcano"]
            else:
                raise ValueError(f"TTS Engine '{name}' is not registered, and fallback 'volcano' is not available.")
        
        return await engine.generate_voice(
            text=text,
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
        engine_name: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if not engine_name:
            engine_name = await self._resolve_engine_and_kwargs(voice_character, kwargs)

        name = engine_name or self._default_engine_name
        engine = self.get_engine(name)
        if not engine:
            if "volcano" in self._engines:
                engine = self._engines["volcano"]
            else:
                raise ValueError(f"TTS Engine '{name}' is not registered, and fallback 'volcano' is not available.")

        # Check if the engine supports streaming
        if hasattr(engine, "generate_voice_stream"):
            async for chunk in engine.generate_voice_stream(
                text=text,
                voice_character=voice_character,
                speed=speed,
                volume=volume,
                emotion=emotion,
                **kwargs
            ):
                yield chunk
        else:
            # Fallback: call generate_voice and read local file or download URL
            local_kwargs = kwargs.copy()
            local_kwargs.pop("disable_segmentation", None)
            local_path_or_url = await engine.generate_voice(
                text=text,
                voice_character=voice_character,
                speed=speed,
                volume=volume,
                emotion=emotion,
                disable_segmentation=True,
                **local_kwargs
            )
            if local_path_or_url.startswith("http://") or local_path_or_url.startswith("https://"):
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(local_path_or_url)
                    if resp.status_code == 200:
                        yield resp.content
            elif os.path.exists(local_path_or_url):
                with open(local_path_or_url, "rb") as f:
                    yield f.read()
