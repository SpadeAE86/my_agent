import os
import uuid
import base64
import httpx
import asyncio
import logging
import tempfile
import shutil
from typing import Optional, List
from sqlmodel import select

from config.config import MY_CONFIG
from exceptions.infra import ServiceException
from models.sqlmodel.voice_designed import VoiceDesigned
from infra.storage.mysql_connector import mysql_connector
from utils.obs_utils import upload_audio

log = logging.getLogger("voice_design_service")


class VoiceDesignService:
    def _get_api_key(self) -> str:
        volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
        api_key = os.getenv("VOLCANO_API_KEY")
        if not api_key:
            api_key = volcano_config.get("api_key")
            if api_key and api_key.startswith("YOUR_"):
                api_key = None
        if not api_key:
            raise ServiceException(
                code=450,
                message="Volcano Voice Design requires VOLCANO_API_KEY to be configured in .env or config.yml"
            )
        return api_key

    async def design_voice(
        self,
        voice_character: str,
        custom_speaker_id: str,
        text: str,
        text_prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        language: int = 0,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None,
        provider: str = "volcano",
        is_online: bool = True
    ) -> VoiceDesigned:
        if provider == "qwen":
            dashscope_key = os.getenv("DASHSCOPE_API_KEY")
            if not dashscope_key:
                raise ServiceException(
                    code=450,
                    message="Qwen Voice Design requires DASHSCOPE_API_KEY to be configured in .env"
                )
            workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
            if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
                workspace_id = MY_CONFIG.get("audio", {}).get("Ali", {}).get("workspace_id")
            if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
                raise ServiceException(
                    code=450,
                    message="Qwen Voice Design requires DASHSCOPE_WORKSPACE_ID to be configured in .env or config.yml"
                )

            # Check if voice_character name already exists in database
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceDesigned).where(VoiceDesigned.voice_character == voice_character)
                res = await session.execute(stmt)
                if res.scalar_one_or_none():
                    raise ServiceException(
                        code=400,
                        message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                    )

            url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/customization"
            headers = {
                "Authorization": f"Bearer {dashscope_key}",
                "Content-Type": "application/json"
            }
            safe_preferred_name = f"v_{uuid.uuid4().hex[:10]}"
            payload = {
                "model": "qwen-voice-design",
                "input": {
                    "action": "create",
                    "target_model": "qwen3-tts-vd-realtime-2026-01-15",
                    "preferred_name": safe_preferred_name,
                    "voice_prompt": text_prompt or text,
                    "preview_text": text,
                    "language": "zh"
                },
                "parameters": {
                    "sample_rate": 24000,
                    "response_format": "wav"
                }
            }

            log.info(f"Submitting Voice Design request to Aliyun: preferred_name={voice_character}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    body = resp.text[:2000]
                    log.error(f"Aliyun Voice Design request failed: status={resp.status_code}, body={body}")
                    raise ServiceException(
                        code=resp.status_code,
                        message=f"阿里云音色设计请求失败: {body}"
                    )
                resp_data = resp.json()

            # Check error in output
            if "output" not in resp_data or "voice" not in resp_data["output"]:
                err_msg = resp_data.get("message", "Unknown Error")
                log.error(f"Aliyun Voice Design returned error: {resp_data}")
                raise ServiceException(
                    code=400,
                    message=f"阿里云音色设计服务返回错误: {err_msg}"
                )

            ali_voice_id = resp_data["output"]["voice"]
            preview_audio_b64 = resp_data["output"].get("preview_audio", {}).get("data")

            # Save designed record in database
            async with mysql_connector.session_scope() as session:
                voice = VoiceDesigned(
                    voice_character=voice_character,
                    custom_speaker_id=ali_voice_id,
                    base_speaker_id=ali_voice_id,
                    text_prompt=text_prompt,
                    image_url=None,
                    status=2,  # Success immediately
                    language=0,
                    provider="qwen",
                    is_online=True,
                    tag=tag,
                    note=note,
                    age_type=age_type,
                    sex=sex,
                    demo_text=text
                )
                session.add(voice)
                await session.commit()
                await session.refresh(voice)

            if preview_audio_b64:
                # Save base64 audio to OBS in background
                asyncio.create_task(self._save_qwen_design_demo(voice.id, preview_audio_b64))

            return voice

        api_key = self._get_api_key()

        # Check if voice_character name already exists in database
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceDesigned).where(VoiceDesigned.voice_character == voice_character)
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise ServiceException(
                    code=400,
                    message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                )

        is_prepaid = False
        if custom_speaker_id.startswith("vdesign_"):
            from services.tts_services.volcano_clone_helper import VolcanoCloneHelper
            prepaid_slot = await VolcanoCloneHelper.find_available_prepaid_slot()
            if prepaid_slot:
                is_prepaid = True
                custom_speaker_id = prepaid_slot
                log.info(f"Selected available prepaid slot for Voice Design: {custom_speaker_id}")
                
                # Delete placeholder record in VoiceCloned to avoid duplicates
                from models.sqlmodel.voice_cloned import VoiceCloned
                async with mysql_connector.session_scope() as session:
                    stmt_del = select(VoiceCloned).where(VoiceCloned.custom_speaker_id == custom_speaker_id)
                    res_del = await session.execute(stmt_del)
                    placeholder_cloned = res_del.scalars().all()
                    for p in placeholder_cloned:
                        await session.delete(p)
                    await session.commit()
            else:
                raise ServiceException(
                    code=400,
                    message="火山引擎预付费 12 个音色槽位已满，无法设计新音色。为了避免产生意外的后付费槽位账单（每个音色槽位 135 元），系统已自动拦截该操作。如需继续设计，请先在列表中删除不需要保持的自定义音色以释放槽位。"
                )

        # Build prompt payload
        prompt_payload = {}
        if text_prompt:
            prompt_payload["text_prompt"] = text_prompt
        if image_url:
            prompt_payload["image_prompt"] = {"image_url": image_url}

        if not prompt_payload:
            raise ServiceException(
                code=400,
                message="音色设计请求必须提供 text_prompt 或 image_url 提示词之一"
            )

        url = "https://openspeech.bytedance.com/api/v3/tts/voice_design"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        
        payload = {
            "speaker_id": custom_speaker_id,
            "text": text,
            "prompt": prompt_payload,
            "language": language
        }

        log.info(f"Submitting Voice Design request to Volcano: speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                body = resp.text[:2000]
                log.error(f"Voice Design request failed: status={resp.status_code}, body={body}")
                raise ServiceException(
                    code=resp.status_code,
                    message=f"火山音色设计请求失败: {body}"
                )
            resp_data = resp.json()

        # Check for error code in response
        code = resp_data.get("code")
        if code is not None and code != 0:
            err_msg = resp_data.get("message", "Unknown Error")
            log.error(f"Volcano Voice Design returned error code {code}: {err_msg}")
            raise ServiceException(
                code=400,
                message=f"音色设计服务返回错误: {err_msg} (错误码: {code})"
            )

        # Voice design is synchronous. Success returns status 2 and demo_audio URL
        demo_audio_url_volcano = resp_data.get("demo_audio")
        
        # Save initially as Designing (1) or Success (2)
        status = 2 if demo_audio_url_volcano else 1
        
        # We'll save the VoiceDesigned record in database
        async with mysql_connector.session_scope() as session:
            voice = VoiceDesigned(
                voice_character=voice_character,
                custom_speaker_id=custom_speaker_id,
                base_speaker_id=custom_speaker_id,
                text_prompt=text_prompt,
                image_url=image_url,
                status=status,
                language=language,
                provider=provider,
                is_online=is_online,
                tag=tag,
                note=note,
                age_type=age_type,
                sex=sex,
                demo_text=text
            )
            session.add(voice)
            await session.commit()
            await session.refresh(voice)

        if demo_audio_url_volcano:
            # Download Volcano's demo audio and upload it permanently to our OBS
            asyncio.create_task(self._download_and_save_demo_audio(voice.id, demo_audio_url_volcano))

        return voice

    async def _download_and_save_demo_audio(self, voice_id: int, demo_audio_url_volcano: str):
        try:
            log.info(f"Downloading trial audio from Volcano: {demo_audio_url_volcano}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(demo_audio_url_volcano)
                if resp.status_code != 200:
                    raise RuntimeError(f"Failed to download demo audio: status={resp.status_code}")
                audio_bytes = resp.content

            temp_dir = tempfile.mkdtemp(prefix=f"voice_design_demo_{voice_id}_")
            temp_file_path = os.path.join(temp_dir, "demo.mp3")

            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)

            log.info(f"Uploading designed voice demo audio to OBS: {temp_file_path}")
            project_id = f"voice_design_demo_{voice_id}"
            demo_audio_url = await upload_audio(temp_file_path, project_id=project_id)
            log.info(f"Designed voice demo audio uploaded. URL: {demo_audio_url}")

            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.status = 2
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()

            shutil.rmtree(temp_dir)
        except Exception as e:
            log.error(f"Failed to save and upload designed voice demo audio: {e}", exc_info=True)
            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.status = 3
                    voice.error_message = f"下载/上传预览音频失败: {str(e)}"
                    session.add(voice)
                    await session.commit()

    async def _save_qwen_design_demo(self, voice_id: int, preview_audio_b64: str):
        try:
            import base64
            audio_bytes = base64.b64decode(preview_audio_b64)
            temp_dir = tempfile.mkdtemp(prefix=f"qwen_voice_design_demo_{voice_id}_")
            temp_file_path = os.path.join(temp_dir, "demo.wav")

            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)

            log.info(f"Uploading Qwen designed voice demo audio to OBS: {temp_file_path}")
            project_id = f"voice_design_demo_{voice_id}"
            demo_audio_url = await upload_audio(temp_file_path, project_id=project_id)
            log.info(f"Qwen designed voice demo audio uploaded. URL: {demo_audio_url}")

            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()

            shutil.rmtree(temp_dir)
        except Exception as e:
            log.error(f"Failed to save and upload Qwen designed voice demo audio: {e}", exc_info=True)

    async def delete_designed_voice(self, voice_id: int):
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceDesigned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色设计记录")
            
            await session.delete(voice)
            await session.commit()
        log.info(f"Deleted custom designed voice record: id={voice_id}")

    async def design_voice_offline(
        self,
        voice_character: str,
        text: str,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None
    ) -> VoiceDesigned:
        # Check if voice_character name already exists in database
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceDesigned).where(VoiceDesigned.voice_character == voice_character)
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise ServiceException(
                    code=400,
                    message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                )

        custom_speaker_id = f"vd_local_{uuid.uuid4().hex[:12]}"
        
        # Save initially as Success (2)
        status = 2
        
        # We'll save the VoiceDesigned record in database
        async with mysql_connector.session_scope() as session:
            voice = VoiceDesigned(
                voice_character=voice_character,
                custom_speaker_id=custom_speaker_id,
                base_speaker_id="local",
                text_prompt=text,
                image_url=None,
                status=status,
                language=0,
                provider="qwen3_local",
                is_online=False,
                tag=tag,
                note=note,
                age_type=age_type,
                sex=sex,
                demo_text=text
            )
            session.add(voice)
            await session.commit()
            await session.refresh(voice)

        # Generate trial demo audio in background for the offline designed voice
        asyncio.create_task(self._generate_offline_design_demo(voice.id, custom_speaker_id))

        return voice

    async def _generate_offline_design_demo(self, voice_id: int, custom_speaker_id: str):
        try:
            from services.tts_services import tts_manager
            log.info(f"Generating offline designed voice demo audio for voice_id={voice_id}")
            demo_text = "你好，我是你的专属AI克隆声音，希望未来可以好好相处哦"
            demo_audio_url = await tts_manager.generate_voice(
                text=demo_text,
                voice_character=custom_speaker_id,
                engine_name="qwen3_local"
            )
            log.info(f"Offline designed voice demo audio generated: {demo_audio_url}")
            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()
        except Exception as e:
            log.error(f"Failed to generate offline designed voice demo audio: {e}", exc_info=True)


voice_design_service = VoiceDesignService()
