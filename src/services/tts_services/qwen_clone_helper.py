import os
import uuid
import base64
import httpx
import asyncio
import logging
import tempfile
import shutil
from typing import Optional
from sqlmodel import select

from config.config import MY_CONFIG
from exceptions.infra import ServiceException
from models.sqlmodel.voice_cloned import VoiceCloned
from models.sqlmodel.voice_designed import VoiceDesigned
from infra.storage.mysql_connector import mysql_connector
from utils.obs_utils import upload_audio
from services.tts_services.clone_utils import upload_original_audio

log = logging.getLogger("qwen_clone_helper")

class QwenCloneHelper:
    @classmethod
    async def clone_voice_qwen(
        cls,
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
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ServiceException(
                code=450,
                message="Qwen Voice Clone requires DASHSCOPE_API_KEY to be configured in .env"
            )
        workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
        if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
            workspace_id = MY_CONFIG.get("audio", {}).get("Ali", {}).get("workspace_id")
        if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
            raise ServiceException(
                code=450,
                message="Qwen Voice Clone requires DASHSCOPE_WORKSPACE_ID to be configured in .env or config.yml"
            )

        # Check if voice_character name already exists in database
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceCloned).where(VoiceCloned.voice_character == voice_character)
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise ServiceException(
                    code=400,
                    message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                )

        mime_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "m4a": "audio/mp4",
            "aac": "audio/aac",
            "pcm": "audio/pcm"
        }
        audio_mime_type = mime_map.get(audio_format.lower(), "audio/mpeg")
        base64_str = base64.b64encode(audio_bytes).decode("utf-8")
        data_uri = f"data:{audio_mime_type};base64,{base64_str}"

        # Upload original reference audio to OBS
        ref_audio_url = None
        try:
            ref_audio_url = await upload_original_audio(audio_bytes, audio_format, prefix="qwen_clone")
            log.info(f"Uploaded original Qwen clone reference audio: {ref_audio_url}")
        except Exception as e:
            log.error(f"Failed to upload original Qwen clone reference audio: {e}", exc_info=True)

        # 北京地域的 MaaS endpoint
        url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/customization"
        safe_preferred_name = f"v_{uuid.uuid4().hex[:10]}"
        payload = {
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vc-2026-01-22",
                "preferred_name": safe_preferred_name,
                "audio": {"data": data_uri}
            }
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        log.info(f"Submitting Qwen Voice Clone request to Aliyun: preferred_name={voice_character}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                body = resp.text[:2000]
                log.error(f"Qwen Voice Clone request failed: status={resp.status_code}, body={body}")
                raise ServiceException(
                    code=resp.status_code,
                    message=f"阿里云声音复刻请求失败: {body}"
                )
            resp_data = resp.json()

        try:
            custom_speaker_id = resp_data["output"]["voice"]
        except (KeyError, TypeError) as e:
            log.error(f"Failed to parse Qwen custom voice ID from response: {resp_data}. Error: {e}")
            raise ServiceException(
                code=400,
                message=f"解析阿里云声音复刻响应失败: {resp_data}"
            )

        # Save to database
        async with mysql_connector.session_scope() as session:
            voice = VoiceCloned(
                voice_character=voice_character,
                custom_speaker_id=custom_speaker_id,
                status=2,  # Success
                prompt_text=prompt_text or "",
                language=0,
                provider="qwen",
                is_online=True,
                tag=tag,
                note=note,
                age_type=age_type,
                sex=sex,
                ref_audio_url=ref_audio_url,
                demo_text=demo_text
            )
            session.add(voice)
            await session.commit()
            await session.refresh(voice)

        # Generate demo audio in background
        asyncio.create_task(cls._generate_qwen_demo_audio(voice.id, custom_speaker_id, demo_text=demo_text))

        return voice

    @classmethod
    async def _generate_qwen_demo_audio(cls, voice_id: int, custom_speaker_id: str, demo_text: Optional[str] = None):
        try:
            from services.tts_services import tts_manager
            log.info(f"Generating Qwen cloned voice demo audio for voice_id={voice_id}")
            if not demo_text:
                demo_text = "你好，我是你的专属AI克隆声音，希望未来可以好好相处哦"
            demo_audio_url = await tts_manager.generate_voice(
                text=demo_text,
                voice_character=custom_speaker_id,
                engine_name="qwen3_online"
            )
            log.info(f"Qwen cloned voice demo audio generated: {demo_audio_url}")
            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceCloned, voice_id)
                if voice:
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()
        except Exception as e:
            log.error(f"Failed to generate Qwen cloned voice demo audio: {e}", exc_info=True)
            try:
                from infra.storage.mysql_connector import mysql_connector
                async with mysql_connector.session_scope() as session:
                    voice = await session.get(VoiceCloned, voice_id)
                    if voice:
                        voice.status = 3
                        voice.error_message = str(e)
                        voice.demo_audio_url = "failed"
                        session.add(voice)
                        await session.commit()
            except Exception as db_err:
                log.error(f"Failed to mark Qwen cloned voice as failed in DB: {db_err}")

    @classmethod
    async def clone_voice_offline(
        cls,
        voice_character: str,
        audio_bytes: bytes,
        audio_format: str,
        prompt_text: Optional[str] = None,
        tag: Optional[str] = None,
        note: Optional[str] = None,
        age_type: Optional[str] = None,
        sex: Optional[str] = None
    ) -> VoiceCloned:
        # Check if voice_character name already exists in database
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceCloned).where(VoiceCloned.voice_character == voice_character)
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise ServiceException(
                    code=400,
                    message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                )

        custom_speaker_id = f"vc_local_{uuid.uuid4().hex[:12]}"
        
        # Save audio locally to a temp file, upload to OBS to get demo_audio_url
        temp_dir = tempfile.mkdtemp(prefix=f"voice_clone_offline_")
        temp_file_path = os.path.join(temp_dir, f"ref_audio.{audio_format}")
        
        try:
            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)
                
            log.info(f"Uploading offline reference audio to OBS: {temp_file_path}")
            project_id = f"voice_clone_offline_{custom_speaker_id}"
            demo_audio_url = await upload_audio(temp_file_path, project_id=project_id)
            log.info(f"Offline reference audio uploaded. URL: {demo_audio_url}")
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        # Save to database
        async with mysql_connector.session_scope() as session:
            voice = VoiceCloned(
                voice_character=voice_character,
                custom_speaker_id=custom_speaker_id,
                status=2,  # Success immediately for local
                prompt_text=prompt_text or "",
                language=0,
                provider="qwen3_local",
                is_online=False,
                demo_audio_url=demo_audio_url,
                ref_audio_url=demo_audio_url,
                tag=tag,
                note=note,
                age_type=age_type,
                sex=sex
            )
            session.add(voice)
            await session.commit()
            await session.refresh(voice)

        return voice

    @classmethod
    async def sync_qwen_voices(cls, force_refresh: bool = False):
        """
        从阿里云 ModelStudio (MAAS API) 同步所有的 Qwen 自定义音色状态到本地数据库
        """
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                log.info("No DASHSCOPE_API_KEY set, skipping Qwen voices sync.")
                return

            workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
            if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
                workspace_id = MY_CONFIG.get("audio", {}).get("Ali", {}).get("workspace_id")
            if not workspace_id or workspace_id in ("YOUR_DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID", ""):
                log.info("No DASHSCOPE_WORKSPACE_ID set, skipping Qwen voices sync.")
                return

            log.info("Starting sync of Qwen custom voices (enrollment & design) via MAAS API...")
            
            # Fetch all existing cloned and designed custom speaker IDs for qwen
            async with mysql_connector.session_scope() as session:
                stmt_cloned = select(VoiceCloned).where(VoiceCloned.provider == "qwen")
                res_cloned = await session.execute(stmt_cloned)
                cloned_records = {r.custom_speaker_id: r for r in res_cloned.scalars().all()}

                stmt_designed = select(VoiceDesigned).where(VoiceDesigned.provider == "qwen")
                res_designed = await session.execute(stmt_designed)
                designed_records = {r.custom_speaker_id: r for r in res_designed.scalars().all()}

            url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/customization"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # 1. Sync Cloned (qwen-voice-enrollment)
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, headers=headers, json={
                    "model": "qwen-voice-enrollment",
                    "input": {
                        "action": "list",
                        "page_size": 100,
                        "page_index": 0
                    }
                })
                if resp.status_code == 200:
                    data = resp.json()
                    voice_list = data.get("output", {}).get("voice_list", [])
                    for item in voice_list:
                        speaker_id = item.get("voice")
                        if not speaker_id:
                            continue
                        
                        if speaker_id not in cloned_records:
                            # Auto-create new record in database!
                            voice_character = f"Qwen克隆_{speaker_id[-4:]}"
                            
                            # Verify uniqueness of character name
                            async with mysql_connector.session_scope() as session:
                                stmt_dup = select(VoiceCloned).where(VoiceCloned.voice_character == voice_character)
                                res_dup = await session.execute(stmt_dup)
                                if res_dup.scalar_one_or_none():
                                    voice_character = f"Qwen克隆_{uuid.uuid4().hex[:4]}_{speaker_id[-4:]}"

                            async with mysql_connector.session_scope() as session:
                                new_voice = VoiceCloned(
                                    voice_character=voice_character,
                                    custom_speaker_id=speaker_id,
                                    status=2,  # Success immediately for Aliyun
                                    language=0,
                                    provider="qwen",
                                    is_online=True,
                                    tag=None,
                                    note=f"同步自阿里云 (创建时间: {item.get('gmt_create')})",
                                    age_type="青年",
                                    sex="0"
                                )
                                session.add(new_voice)
                                await session.commit()
                                await session.refresh(new_voice)

                            # Generate demo audio in background
                            asyncio.create_task(cls._generate_qwen_demo_audio(new_voice.id, speaker_id))
                            log.info(f"Auto-created and synced Qwen cloned voice: {voice_character} ({speaker_id})")
                        elif force_refresh:
                            record = cloned_records[speaker_id]
                            asyncio.create_task(cls._generate_qwen_demo_audio(record.id, speaker_id))
                            log.info(f"Force refreshed Qwen cloned voice demo audio: {record.voice_character} ({speaker_id})")
                else:
                    log.warning(f"Failed to query Qwen cloned voices list: HTTP {resp.status_code}")

            # 2. Sync Designed (qwen-voice-design)
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, headers=headers, json={
                    "model": "qwen-voice-design",
                    "input": {
                        "action": "list",
                        "page_size": 100,
                        "page_index": 0
                    }
                })
                if resp.status_code == 200:
                    data = resp.json()
                    voice_list = data.get("output", {}).get("voice_list", [])
                    for item in voice_list:
                        speaker_id = item.get("voice")
                        if not speaker_id:
                            continue
                        
                        if speaker_id not in designed_records:
                            # Auto-create new record in database!
                            voice_character = f"Qwen设计_{speaker_id[-4:]}"
                            
                            # Verify uniqueness of character name
                            async with mysql_connector.session_scope() as session:
                                stmt_dup = select(VoiceDesigned).where(VoiceDesigned.voice_character == voice_character)
                                res_dup = await session.execute(stmt_dup)
                                if res_dup.scalar_one_or_none():
                                    voice_character = f"Qwen设计_{uuid.uuid4().hex[:4]}_{speaker_id[-4:]}"

                            async with mysql_connector.session_scope() as session:
                                new_voice = VoiceDesigned(
                                    voice_character=voice_character,
                                    custom_speaker_id=speaker_id,
                                    base_speaker_id=speaker_id,
                                    text_prompt=item.get("voice_prompt"),
                                    image_url=None,
                                    status=2,  # Success immediately
                                    language=0,
                                    provider="qwen",
                                    is_online=True,
                                    tag=None,
                                    note=f"同步自阿里云 (创建时间: {item.get('gmt_create')}, 描述: {item.get('voice_prompt')})",
                                    age_type="青年",
                                    sex="0"
                                )
                                session.add(new_voice)
                                await session.commit()
                                await session.refresh(new_voice)

                            # Generate demo audio in background
                            asyncio.create_task(cls._generate_qwen_design_demo_audio(new_voice.id, speaker_id))
                            log.info(f"Auto-created and synced Qwen designed voice: {voice_character} ({speaker_id})")
                        elif force_refresh:
                            record = designed_records[speaker_id]
                            asyncio.create_task(cls._generate_qwen_design_demo_audio(record.id, speaker_id))
                            log.info(f"Force refreshed Qwen designed voice demo audio: {record.voice_character} ({speaker_id})")
                else:
                    log.warning(f"Failed to query Qwen designed voices list: HTTP {resp.status_code}")

        except Exception as e:
            log.error(f"Failed to sync Qwen custom voices: {e}", exc_info=True)

    @classmethod
    async def _generate_qwen_design_demo_audio(cls, voice_id: int, custom_speaker_id: str):
        try:
            from services.tts_services import tts_manager
            log.info(f"Generating Qwen designed voice demo audio for voice_id={voice_id}")
            demo_text = "你好，我是你的专属设计声音，希望未来可以好好相处哦"
            demo_audio_url = await tts_manager.generate_voice(
                text=demo_text,
                voice_character=custom_speaker_id,
                engine_name="qwen3_online"
            )
            log.info(f"Qwen designed voice demo audio generated: {demo_audio_url}")
            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()
        except Exception as e:
            log.error(f"Failed to generate Qwen designed voice demo audio: {e}", exc_info=True)
            try:
                from infra.storage.mysql_connector import mysql_connector
                async with mysql_connector.session_scope() as session:
                    voice = await session.get(VoiceDesigned, voice_id)
                    if voice:
                        voice.status = 3
                        voice.error_message = str(e)
                        voice.demo_audio_url = "failed"
                        session.add(voice)
                        await session.commit()
            except Exception as db_err:
                log.error(f"Failed to mark Qwen designed voice as failed in DB: {db_err}")
