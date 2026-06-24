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

log = logging.getLogger("volcano_clone_helper")

class VolcanoCloneHelper:
    @classmethod
    def _get_api_key(cls) -> str:
        volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
        api_key = os.getenv("VOLCANO_API_KEY")
        if not api_key:
            api_key = volcano_config.get("api_key")
            if api_key and api_key.startswith("YOUR_"):
                api_key = None
        if not api_key:
            raise ServiceException(
                code=450,
                message="Volcano Voice Clone requires VOLCANO_API_KEY to be configured in .env or config.yml"
            )
        return api_key

    @classmethod
    async def find_available_prepaid_slot(cls) -> Optional[str]:
        # Query all records from VoiceCloned and VoiceDesigned that start with "S_"
        async with mysql_connector.session_scope() as session:
            stmt_cloned = select(VoiceCloned).where(VoiceCloned.custom_speaker_id.like("S_%"))
            res_cloned = await session.execute(stmt_cloned)
            cloned_records = res_cloned.scalars().all()
            
            stmt_designed = select(VoiceDesigned).where(VoiceDesigned.custom_speaker_id.like("S_%"))
            res_designed = await session.execute(stmt_designed)
            designed_records = res_designed.scalars().all()
            
        used_slots = set()
        all_slots = set()
        
        for r in cloned_records:
            all_slots.add(r.custom_speaker_id)
            if r.voice_character != r.custom_speaker_id and r.status in (2, 4):
                used_slots.add(r.custom_speaker_id)
                
        for r in designed_records:
            all_slots.add(r.custom_speaker_id)
            if r.voice_character != r.custom_speaker_id and r.status in (2, 4):
                used_slots.add(r.custom_speaker_id)
                
        available_slots = sorted(list(all_slots - used_slots))
        if available_slots:
            return available_slots[0]
        return None

    @classmethod
    async def clone_voice(
        cls,
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
        # Check if voice_character name already exists in database
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceCloned).where(VoiceCloned.voice_character == voice_character)
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise ServiceException(
                    code=400,
                    message=f"已存在名为 '{voice_character}' 的自定义音色，请更换名称"
                )

        custom_speaker_id = await cls.find_available_prepaid_slot()
        is_prepaid = False
        if custom_speaker_id:
            is_prepaid = True
            log.info(f"Selected available prepaid slot: {custom_speaker_id}")
        else:
            raise ServiceException(
                code=400,
                message="火山引擎预付费 12 个音色槽位已满，无法创建新音色。为了避免产生意外的后付费槽位账单（每个音色槽位 135 元），系统已自动拦截该操作。如需继续创建，请先在列表中删除不需要的自定义音色以释放槽位。"
            )

        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Fixed trial text or custom demo text
        if not demo_text:
            demo_text = "你好，我是你的专属AI克隆声音，希望未来可以好好相处哦"

        if not prompt_text:
            prompt_text = ""

        # Upload original reference audio to OBS
        ref_audio_url = None
        try:
            ref_audio_url = await upload_original_audio(audio_bytes, audio_format, prefix="clone")
            log.info(f"Uploaded original clone reference audio: {ref_audio_url}")
        except Exception as e:
            log.error(f"Failed to upload original clone reference audio: {e}", exc_info=True)

        url = "https://openspeech.bytedance.com/api/v3/tts/voice_clone"
        if is_prepaid:
            payload = {
                "speaker_id": custom_speaker_id,
                "audio": {
                    "data": base64_audio,
                    "format": audio_format
                },
                "text": prompt_text,
                "language": language,
                "extra_params": {
                    "demo_text": demo_text,
                    "enable_audio_denoise": False,
                    "disable_volume_normalization": False
                }
            }
        else:
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id,
                "audio": {
                    "data": base64_audio,
                    "format": audio_format
                },
                "text": prompt_text,
                "language": language,
                "extra_params": {
                    "demo_text": demo_text,
                    "enable_audio_denoise": False,
                    "disable_volume_normalization": False
                }
            }

        # Authentication Setup (API Key)
        api_key = os.getenv("VOLCANO_CLONE_API_KEY") or cls._get_api_key()
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }

        log.info(f"Submitting Voice Clone request to Volcano: custom_speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                body = resp.text[:2000]
                log.error(f"Voice Clone request failed: status={resp.status_code}, body={body}")
                raise ServiceException(
                    code=resp.status_code,
                    message=f"火山声音复刻请求失败: {body}"
                )
            
            resp_data = resp.json()

        code = resp_data.get("code")
        if code is not None and code != 0:
            err_msg = resp_data.get("message", "Unknown Error")
            log.error(f"Volcano Voice Clone returned error code {code}: {err_msg}")
            raise ServiceException(
                code=400,
                message=f"声音复刻服务返回错误: {err_msg} (错误码: {code})"
            )

        status = resp_data.get("status", 1) # Default to 1 (Training)

        # Save to database
        async with mysql_connector.session_scope() as session:
            if is_prepaid:
                stmt_find = select(VoiceCloned).where(VoiceCloned.custom_speaker_id == custom_speaker_id)
                res_find = await session.execute(stmt_find)
                voice = res_find.scalar_one_or_none()
                if voice:
                    voice.voice_character = voice_character
                    voice.status = status
                    voice.prompt_text = prompt_text
                    voice.language = language
                    voice.provider = "volcano"
                    voice.is_online = True
                    voice.tag = tag
                    voice.note = note
                    voice.age_type = age_type
                    voice.sex = sex
                    voice.ref_audio_url = ref_audio_url
                    voice.demo_text = demo_text
                    session.add(voice)
                    await session.commit()
                    voice_id = voice.id
                else:
                    voice = VoiceCloned(
                        voice_character=voice_character,
                        custom_speaker_id=custom_speaker_id,
                        status=status,
                        prompt_text=prompt_text,
                        language=language,
                        provider="volcano",
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
                    voice_id = voice.id
            else:
                voice = VoiceCloned(
                    voice_character=voice_character,
                    custom_speaker_id=custom_speaker_id,
                    status=status,
                    prompt_text=prompt_text,
                    language=language,
                    provider="volcano",
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
                voice_id = voice.id

        if status == 1:
            log.info(f"Timbre '{voice_character}' training started in background.")
            asyncio.create_task(cls.poll_and_finalize_clone(voice_id, custom_speaker_id))
        elif status in (2, 4):
            # Already success, finalize immediately
            demo_audio_b64 = resp_data.get("demo_audio")
            if demo_audio_b64:
                asyncio.create_task(cls._save_demo_audio(voice_id, demo_audio_b64))

        return voice

    @classmethod
    async def poll_and_finalize_clone(cls, voice_id: int, custom_speaker_id: str):
        api_key = os.getenv("VOLCANO_CLONE_API_KEY") or cls._get_api_key()
        url = "https://openspeech.bytedance.com/api/v3/tts/get_voice"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        if custom_speaker_id.startswith("vc_") or custom_speaker_id.startswith("vclone_"):
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id
            }
        else:
            payload = {
                "speaker_id": custom_speaker_id
            }

        # Poll Volcano Voice status up to 20 times (every 5 seconds)
        for attempt in range(20):
            await asyncio.sleep(5)
            try:
                log.info(f"Polling clone status for custom_speaker_id={custom_speaker_id} (attempt {attempt+1})")
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code != 200:
                        log.warning(f"Failed to poll voice status: status={resp.status_code}")
                        continue
                    resp_data = resp.json()

                if resp_data.get("code") is not None and resp_data.get("code") != 0:
                    continue

                status = resp_data.get("status", 1)
                log.info(f"Voice {custom_speaker_id} status polled: {status}")

                if status in (2, 4): # Success or Active
                    demo_audio_b64 = resp_data.get("demo_audio")
                    if demo_audio_b64:
                        await cls._save_demo_audio(voice_id, demo_audio_b64)
                    else:
                        async with mysql_connector.session_scope() as session:
                            voice = await session.get(VoiceCloned, voice_id)
                            if voice:
                                voice.status = 2
                                session.add(voice)
                                await session.commit()
                    log.info(f"Cloned voice custom_speaker_id={custom_speaker_id} finalized successfully.")
                    return

                elif status == 3: # Failed
                    err_msg = resp_data.get("message", "训练失败")
                    async with mysql_connector.session_scope() as session:
                        voice = await session.get(VoiceCloned, voice_id)
                        if voice:
                            voice.status = 3
                            voice.error_message = err_msg
                            session.add(voice)
                            await session.commit()
                    log.warning(f"Cloned voice custom_speaker_id={custom_speaker_id} failed: {err_msg}")
                    return

            except Exception as e:
                log.warning(f"Error during polling voice status: {e}", exc_info=True)

        # Timeout, mark as failed
        log.warning(f"Cloned voice custom_speaker_id={custom_speaker_id} status polling timed out.")
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if voice and voice.status == 1:
                voice.status = 3
                voice.error_message = "训练超时，请重试"
                session.add(voice)
                await session.commit()

    @classmethod
    async def _save_demo_audio(cls, voice_id: int, demo_audio_b64: str):
        try:
            if demo_audio_b64.startswith("http://") or demo_audio_b64.startswith("https://"):
                log.info(f"Downloading demo audio from URL: {demo_audio_b64}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(demo_audio_b64)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Failed to download demo audio: status={resp.status_code}")
                    audio_bytes = resp.content
            else:
                audio_bytes = base64.b64decode(demo_audio_b64)
            temp_dir = tempfile.mkdtemp(prefix=f"voice_clone_demo_{voice_id}_")
            temp_file_path = os.path.join(temp_dir, "demo.mp3")

            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)

            log.info(f"Uploading cloned voice demo audio to OBS: {temp_file_path}")
            project_id = f"voice_clone_demo_{voice_id}"
            demo_audio_url = await upload_audio(temp_file_path, project_id=project_id)
            log.info(f"Cloned voice demo audio uploaded. URL: {demo_audio_url}")

            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceCloned, voice_id)
                if voice:
                    voice.status = 2
                    voice.demo_audio_url = demo_audio_url
                    session.add(voice)
                    await session.commit()

            shutil.rmtree(temp_dir)
        except Exception as e:
            log.error(f"Failed to save and upload cloned voice demo audio: {e}", exc_info=True)

    @classmethod
    async def delete_cloned_voice(cls, voice_id: int):
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色复刻记录")
            
            # Delete record
            await session.delete(voice)
            await session.commit()
        log.info(f"Deleted custom voice cloned record: id={voice_id}")

    @classmethod
    async def query_and_update_status(cls, voice_id: int) -> VoiceCloned:
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色复刻记录")
            if voice.provider == "qwen":
                return voice
            custom_speaker_id = voice.custom_speaker_id
            
        api_key = cls._get_api_key()
            
        url = "https://openspeech.bytedance.com/api/v3/tts/get_voice"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        
        if custom_speaker_id.startswith("vc_") or custom_speaker_id.startswith("vclone_"):
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id
            }
        else:
            payload = {
                "speaker_id": custom_speaker_id
            }
            
        log.info(f"Querying live status from Volcano for custom_speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise ServiceException(
                    code=resp.status_code,
                    message=f"查询音色状态失败 (HTTP {resp.status_code}): {resp.text}"
                )
            resp_data = resp.json()
            
        if resp_data.get("code") is not None and resp_data.get("code") != 0:
            err_msg = resp_data.get("message", "查询失败")
            raise ServiceException(
                code=400,
                message=f"查询服务返回错误: {err_msg}"
            )
            
        status = resp_data.get("status", 1)
        demo_audio_b64 = resp_data.get("demo_audio")
        
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if voice:
                if status in (2, 4) and demo_audio_b64 and not voice.demo_audio_url:
                    # Finalize audio if demo audio is available but not saved yet
                    await cls._save_demo_audio(voice_id, demo_audio_b64)
                else:
                    voice.status = status
                    if status == 3:
                        voice.error_message = resp_data.get("message", "训练失败")
                    session.add(voice)
                    await session.commit()
                    await session.refresh(voice)
                return voice

    @classmethod
    async def _save_designed_demo_audio(cls, voice_id: int, demo_audio_url: str):
        try:
            if demo_audio_url.startswith("http://") or demo_audio_url.startswith("https://"):
                log.info(f"Downloading designed demo audio from URL: {demo_audio_url}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(demo_audio_url)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Failed to download demo audio: status={resp.status_code}")
                    audio_bytes = resp.content
            else:
                audio_bytes = base64.b64decode(demo_audio_url)
            temp_dir = tempfile.mkdtemp(prefix=f"voice_design_demo_{voice_id}_")
            temp_file_path = os.path.join(temp_dir, "demo.mp3")

            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)

            log.info(f"Uploading designed voice demo audio to OBS: {temp_file_path}")
            project_id = f"voice_design_demo_{voice_id}"
            obs_url = await upload_audio(temp_file_path, project_id=project_id)
            log.info(f"Designed voice demo audio uploaded. URL: {obs_url}")

            async with mysql_connector.session_scope() as session:
                voice = await session.get(VoiceDesigned, voice_id)
                if voice:
                    voice.status = 2
                    voice.demo_audio_url = obs_url
                    session.add(voice)
                    await session.commit()

            shutil.rmtree(temp_dir)
        except Exception as e:
            log.error(f"Failed to save and upload designed voice demo audio: {e}", exc_info=True)

    @classmethod
    async def sync_volcano_voices(cls, force_refresh: bool = False):
        """
        从火山引擎 OpenAPI 同步所有的自定义音色状态到本地数据库
        """
        from utils.volcano_utils import query_volcano_train_statuses
        try:
            log.info("Starting sync of Volcano custom voices via OpenAPI...")
            result = await query_volcano_train_statuses(1, 100)
            statuses = result.get("Statuses", [])
            if not statuses:
                log.info("No custom voices found on Volcano Engine.")
                return

            # Fetch all existing cloned and designed custom speaker IDs
            async with mysql_connector.session_scope() as session:
                stmt_cloned = select(VoiceCloned).where(VoiceCloned.provider == "volcano")
                res_cloned = await session.execute(stmt_cloned)
                cloned_records = {r.custom_speaker_id: r for r in res_cloned.scalars().all()}

                stmt_designed = select(VoiceDesigned).where(VoiceDesigned.provider == "volcano")
                res_designed = await session.execute(stmt_designed)
                designed_records = {r.custom_speaker_id: r for r in res_designed.scalars().all()}

            for item in statuses:
                speaker_id = item.get("SpeakerID")
                if not speaker_id:
                    continue

                # Map state to database status
                volc_state = item.get("State", "Unknown")
                if volc_state == "Training":
                    mapped_status = 1
                elif volc_state == "Success":
                    mapped_status = 2
                elif volc_state == "Active":
                    mapped_status = 4
                elif volc_state in ("Expired", "Reclaimed"):
                    mapped_status = 3
                else:
                    mapped_status = 1  # default to training/unknown

                # Prioritize ModelType 5 (Uranus) -> ModelType 4 (Saturn) -> top-level DemoAudio
                demo_audio = None
                details = item.get("ModelTypeDetails") or []
                for d in details:
                    if d.get("ModelType") == 5 and d.get("DemoAudio"):
                        demo_audio = d.get("DemoAudio")
                        break
                if not demo_audio:
                    for d in details:
                        if d.get("ModelType") == 4 and d.get("DemoAudio"):
                            demo_audio = d.get("DemoAudio")
                            break
                if not demo_audio:
                    demo_audio = item.get("DemoAudio")

                # Check if it exists in designed records
                avail_times = item.get("AvailableTrainingTimes")
                if speaker_id in designed_records:
                    record = designed_records[speaker_id]
                    if record.status != mapped_status or force_refresh or (demo_audio and not record.demo_audio_url) or record.available_training_times != avail_times:
                        async with mysql_connector.session_scope() as session:
                            db_record = await session.get(VoiceDesigned, record.id)
                            if db_record:
                                db_record.status = mapped_status
                                db_record.available_training_times = avail_times
                                session.add(db_record)
                                await session.commit()
                                if demo_audio and (force_refresh or not db_record.demo_audio_url):
                                    # Download and upload demo audio in background
                                    asyncio.create_task(cls._save_designed_demo_audio(db_record.id, demo_audio))
                    continue

                # Map or create in cloned records
                if speaker_id in cloned_records:
                    record = cloned_records[speaker_id]
                    if record.status != mapped_status or force_refresh or (demo_audio and not record.demo_audio_url) or record.available_training_times != avail_times:
                        async with mysql_connector.session_scope() as session:
                            db_record = await session.get(VoiceCloned, record.id)
                            if db_record:
                                db_record.status = mapped_status
                                db_record.available_training_times = avail_times
                                session.add(db_record)
                                await session.commit()
                                if demo_audio and (force_refresh or not db_record.demo_audio_url):
                                    # Download and upload demo audio in background
                                    asyncio.create_task(cls._save_demo_audio(db_record.id, demo_audio))
                else:
                    # Auto-create new record in database!
                    alias = item.get("Alias") or speaker_id
                    
                    # Generate a unique voice_character name
                    voice_character = alias
                    async with mysql_connector.session_scope() as session:
                        # Check if duplicate voice_character name exists
                        stmt_dup = select(VoiceCloned).where(VoiceCloned.voice_character == voice_character)
                        res_dup = await session.execute(stmt_dup)
                        if res_dup.scalar_one_or_none():
                            voice_character = f"{alias}_{speaker_id[-4:]}"

                    # Labels parsing
                    labels = item.get("Labels") or []
                    tag = ",".join(labels) if labels else None
                    sex = "0" if any(g in labels for g in ("女", "female")) else ("1" if any(g in labels for g in ("男", "male")) else "0")
                    
                    age_type = "青年"
                    for a in ("儿童", "少年", "少女", "青年", "中年", "老年"):
                        if a in labels:
                            age_type = a
                            break

                    # Insert new cloned record
                    async with mysql_connector.session_scope() as session:
                        new_voice = VoiceCloned(
                            voice_character=voice_character,
                            custom_speaker_id=speaker_id,
                            status=mapped_status,
                            language=0,
                            provider="volcano",
                            is_online=True,
                            tag=tag,
                            note=item.get("Description") or "同步自火山引擎",
                            age_type=age_type,
                            sex=sex,
                            available_training_times=avail_times
                        )
                        session.add(new_voice)
                        await session.commit()
                        await session.refresh(new_voice)
                        
                        if demo_audio:
                            # Download and upload demo audio in background
                            asyncio.create_task(cls._save_demo_audio(new_voice.id, demo_audio))
                            
                    log.info(f"Auto-created and synced voice from Volcano Open API: {voice_character} ({speaker_id})")

        except Exception as e:
            log.error(f"Failed to sync Volcano Engine voices: {e}", exc_info=True)

    @classmethod
    async def upgrade_cloned_voice(cls, voice_id: int) -> VoiceCloned:
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色复刻记录")
            if voice.provider == "qwen":
                raise ServiceException(code=400, message="阿里云音色复刻不需要且不支持升级操作")
            custom_speaker_id = voice.custom_speaker_id
            
        api_key = cls._get_api_key()
            
        url = "https://openspeech.bytedance.com/api/v3/tts/upgrade_voice"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        
        if custom_speaker_id.startswith("vc_") or custom_speaker_id.startswith("vclone_"):
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id
            }
        else:
            payload = {
                "speaker_id": custom_speaker_id
            }
            
        log.info(f"Upgrading custom voice to V3 for custom_speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise ServiceException(
                    code=resp.status_code,
                    message=f"音色升级失败 (HTTP {resp.status_code}): {resp.text}"
                )
            resp_data = resp.json()
            
        if resp_data.get("code") is not None and resp_data.get("code") != 0:
            err_msg = resp_data.get("message", "升级失败")
            raise ServiceException(
                code=400,
                message=f"音色升级服务返回错误: {err_msg}"
            )
            
        status = resp_data.get("status", 1)
        demo_audio_b64 = resp_data.get("demo_audio")
        
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceCloned, voice_id)
            if voice:
                if voice.available_training_times is not None and voice.available_training_times > 0:
                    voice.available_training_times -= 1
                if status in (2, 4) and demo_audio_b64 and not voice.demo_audio_url:
                    # Finalize audio if demo audio is returned
                    await cls._save_demo_audio(voice_id, demo_audio_b64)
                else:
                    voice.status = status
                    if status == 3:
                        voice.error_message = resp_data.get("message", "升级失败")
                    session.add(voice)
                    await session.commit()
                    await session.refresh(voice)
                return voice

    @classmethod
    async def upgrade_designed_voice(cls, voice_id: int) -> VoiceDesigned:
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceDesigned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色设计记录")
            if voice.provider == "qwen":
                raise ServiceException(code=400, message="阿里云音色设计不需要且不支持升级操作")
            custom_speaker_id = voice.custom_speaker_id
            
        api_key = cls._get_api_key()
            
        url = "https://openspeech.bytedance.com/api/v3/tts/upgrade_voice"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        
        if custom_speaker_id.startswith("vc_") or custom_speaker_id.startswith("vclone_"):
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id
            }
        else:
            payload = {
                "speaker_id": custom_speaker_id
            }
            
        log.info(f"Upgrading designed custom voice to V3 for custom_speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise ServiceException(
                    code=resp.status_code,
                    message=f"音色升级失败 (HTTP {resp.status_code}): {resp.text}"
                )
            resp_data = resp.json()
            
        if resp_data.get("code") is not None and resp_data.get("code") != 0:
            err_msg = resp_data.get("message", "升级失败")
            raise ServiceException(
                code=400,
                message=f"音色升级服务返回错误: {err_msg}"
            )
            
        status = resp_data.get("status", 1)
        demo_audio_b64 = resp_data.get("demo_audio")
        
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceDesigned, voice_id)
            if voice:
                if voice.available_training_times is not None and voice.available_training_times > 0:
                    voice.available_training_times -= 1
                if status in (2, 4) and demo_audio_b64 and not voice.demo_audio_url:
                    await cls._save_designed_demo_audio(voice_id, demo_audio_b64)
                else:
                    voice.status = status
                    if status == 3:
                        voice.error_message = resp_data.get("message", "升级失败")
                    session.add(voice)
                    await session.commit()
                    await session.refresh(voice)
                return voice

    @classmethod
    async def query_and_update_designed_status(cls, voice_id: int) -> VoiceDesigned:
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceDesigned, voice_id)
            if not voice:
                raise ServiceException(code=404, message="找不到该音色设计记录")
            if voice.provider == "qwen":
                return voice
            custom_speaker_id = voice.custom_speaker_id
            
        api_key = cls._get_api_key()
            
        url = "https://openspeech.bytedance.com/api/v3/tts/get_voice"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-Request-Id": str(uuid.uuid4())
        }
        
        if custom_speaker_id.startswith("vc_") or custom_speaker_id.startswith("vclone_"):
            payload = {
                "speaker_id": "custom_speaker_id",
                "custom_speaker_id": custom_speaker_id
            }
        else:
            payload = {
                "speaker_id": custom_speaker_id
            }
            
        log.info(f"Querying live status from Volcano for designed custom_speaker_id={custom_speaker_id}")
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise ServiceException(
                    code=resp.status_code,
                    message=f"查询音色设计状态失败 (HTTP {resp.status_code}): {resp.text}"
                )
            resp_data = resp.json()
            
        if resp_data.get("code") is not None and resp_data.get("code") != 0:
            err_msg = resp_data.get("message", "查询失败")
            raise ServiceException(
                code=400,
                message=f"查询服务返回错误: {err_msg}"
            )
            
        status = resp_data.get("status", 1)
        demo_audio_b64 = resp_data.get("demo_audio")
        
        async with mysql_connector.session_scope() as session:
            voice = await session.get(VoiceDesigned, voice_id)
            if voice:
                if status in (2, 4) and demo_audio_b64 and not voice.demo_audio_url:
                    await cls._save_designed_demo_audio(voice_id, demo_audio_b64)
                else:
                    voice.status = status
                    if status == 3:
                        voice.error_message = resp_data.get("message", "训练失败")
                    session.add(voice)
                    await session.commit()
                    await session.refresh(voice)
                return voice
