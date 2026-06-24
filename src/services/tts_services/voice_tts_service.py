import os
import re
import json
import asyncio
import logging
from typing import AsyncGenerator, Optional
from sqlmodel import select

from models.sqlmodel.voice_tts import VoiceTTSTask, VoiceTTSChunk
from infra.storage.mysql_connector import mysql_connector

from infra.logging.logger import logger as log


class VoiceTTSService:
    _resolved_speaker_cache = {}

    def split_text_by_punctuation(self, t: str, max_len: int = 200) -> list[str]:
        if len(t) <= max_len:
            return [t]
        sentences = re.split(r'([。！？!?；;\n]+)', t)
        chunks = []
        current_chunk = ""
        i = 0
        while i < len(sentences):
            part = sentences[i]
            punct = sentences[i+1] if i + 1 < len(sentences) else ""
            sentence = part + punct
            i += 2
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) <= max_len:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) > max_len:
                    sub_parts = re.split(r'([，,、：:]+)', sentence)
                    sub_chunk = ""
                    j = 0
                    while j < len(sub_parts):
                        sub_part = sub_parts[j]
                        sub_punct = sub_parts[j+1] if j + 1 < len(sub_parts) else ""
                        sub_sentence = sub_part + sub_punct
                        j += 2
                        if not sub_sentence.strip():
                            continue
                        if len(sub_chunk) + len(sub_sentence) <= max_len:
                            sub_chunk += sub_sentence
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            if len(sub_sentence) > max_len:
                                for k in range(0, len(sub_sentence), max_len):
                                    chunks.append(sub_sentence[k:k+max_len])
                                sub_chunk = ""
                            else:
                                sub_chunk = sub_sentence
                        current_chunk = sub_chunk
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    async def generate_tts_stream(
        self, bubble_id: str, voice_character: str, text: str, speed: float = 1.0, byte_stream: bool = False
    ) -> AsyncGenerator[str, None]:
        from services.volcovoice_service import VolcoVoiceService
        import time
        start_time = time.time()
        first_packet_logged = False

        start_msg = f"[Metric] TTS Stream Request Started (bubble_id={bubble_id}, voice={voice_character})"
        log.info(start_msg)
        print(start_msg, flush=True)

        def track_first_packet():
            nonlocal first_packet_logged
            if not first_packet_logged:
                latency = (time.time() - start_time) * 1000
                latency_msg = f"[Metric] TTS Stream First Packet Latency: {latency:.2f}ms (bubble_id={bubble_id}, voice={voice_character})"
                log.info(latency_msg)
                print(latency_msg, flush=True)
                first_packet_logged = True

        cleaned_text = VolcoVoiceService.clean_markdown(text)
        if not cleaned_text:
            cleaned_text = "没有可朗读的文本。"

        # 0. 尝试解析自定义克隆音色或设计音色名称为 custom_speaker_id
        cached_speaker = self._resolved_speaker_cache.get(voice_character)
        if cached_speaker:
            voice_character = cached_speaker
        else:
            from models.sqlmodel.voice_cloned import VoiceCloned
            from models.sqlmodel.voice_designed import VoiceDesigned
            try:
                async with mysql_connector.session_scope() as session:
                    stmt_clone = select(VoiceCloned).where(
                        (VoiceCloned.custom_speaker_id == voice_character) |
                        (VoiceCloned.voice_character == voice_character)
                    )
                    res_clone = await session.execute(stmt_clone)
                    clone_record = res_clone.scalar_one_or_none()
                    if clone_record:
                        original_char = voice_character
                        voice_character = clone_record.custom_speaker_id
                        self._resolved_speaker_cache[original_char] = voice_character
                        log.info(f"Resolved custom cloned voice to custom_speaker_id='{voice_character}'")
                        if clone_record.provider == "volcano" and clone_record.status != 4:
                            clone_record.status = 4
                            session.add(clone_record)
                            await session.commit()
                            log.info(f"Custom cloned voice '{voice_character}' has been used for synthesis. Marked as Active (status=4) in DB.")
                    else:
                        stmt_design = select(VoiceDesigned).where(
                            (VoiceDesigned.custom_speaker_id == voice_character) |
                            (VoiceDesigned.voice_character == voice_character)
                        )
                        res_design = await session.execute(stmt_design)
                        design_record = res_design.scalar_one_or_none()
                        if design_record:
                            original_char = voice_character
                            voice_character = design_record.custom_speaker_id
                            self._resolved_speaker_cache[original_char] = voice_character
                            log.info(f"Resolved custom designed voice to custom_speaker_id='{voice_character}'")
                            if design_record.provider == "volcano" and design_record.status != 4:
                                design_record.status = 4
                                session.add(design_record)
                                await session.commit()
                                log.info(f"Custom designed voice '{voice_character}' has been used for synthesis. Marked as Active (status=4) in DB.")
            except Exception as e:
                log.warning(f"Failed to query custom cloned or designed voice in DB: {e}")
        print(f"[Timing] 1. Custom voice resolved: {(time.time() - start_time)*1000:.2f}ms", flush=True)
        # 1. 检查数据库中是否已存在该气泡和音色的完整任务
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceTTSTask).where(
                VoiceTTSTask.bubble_id == bubble_id,
                VoiceTTSTask.voice_character == voice_character
            )
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()

            if task:
                # 缓存命中：已合并的完整音频
                if task.oss_url and task.status == "completed":
                    log.info(f"TTS Task Cache hit (merged): bubble_id={bubble_id}, voice={voice_character} -> {task.oss_url}")
                    track_first_packet()
                    yield f"data: {json.dumps({'event_type': 'voice_chunk', 'data_type': 'url', 'url': task.oss_url, 'text': cleaned_text, 'index': 0}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        print(f"[Timing] 2. Task cache checked: {(time.time() - start_time)*1000:.2f}ms", flush=True)

        # 2. 如果任务不存在，则分句生成
        segments = self.split_text_by_punctuation(cleaned_text, 200)

        async with mysql_connector.session_scope() as session:
            # 二次确认，防止并发请求下冲突
            stmt = select(VoiceTTSTask).where(
                VoiceTTSTask.bubble_id == bubble_id,
                VoiceTTSTask.voice_character == voice_character
            )
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()
            if not task:
                task = VoiceTTSTask(
                    bubble_id=bubble_id,
                    voice_character=voice_character,
                    full_text=cleaned_text,
                    status="pending"
                )
                session.add(task)
                await session.commit()
                await session.refresh(task)
        print(f"[Timing] 3. Task row confirmed in DB: {(time.time() - start_time)*1000:.2f}ms", flush=True)

        from services.tts_services import tts_manager
        import base64

        local_paths = []

        for idx, segment in enumerate(segments):
            segment_text = segment.strip()
            if not segment_text:
                continue

            if idx == 0:
                print(f"[Timing] 4. Starting voice synthesis: {(time.time() - start_time)*1000:.2f}ms", flush=True)

            try:
                if byte_stream:
                    import tempfile
                    project_id = f"segment_{task.id}_{idx}_{int(time.time() * 1000)}"
                    temp_dir = os.path.join("./final", project_id)
                    os.makedirs(temp_dir, exist_ok=True)
                    local_path = os.path.abspath(os.path.join(temp_dir, f"voice_{project_id}.mp3"))

                    # Write to local file incrementally as we receive chunks from stream
                    log.info(f"Starting generate_voice_stream for idx={idx}")
                    with open(local_path, "wb") as f_out:
                        async for audio_chunk in tts_manager.generate_voice_stream(
                            text=segment_text,
                            voice_character=voice_character,
                            speed=speed,
                            volume=1.0,
                            emotion="neutral",
                            disable_segmentation=True
                        ):
                            f_out.write(audio_chunk)
                            base64_data = base64.b64encode(audio_chunk).decode("utf-8")
                            
                            track_first_packet()
                            yield f"data: {json.dumps({'event_type': 'voice_chunk', 'data_type': 'base64', 'data': base64_data, 'text': segment_text, 'index': idx}, ensure_ascii=False)}\n\n"

                    local_paths.append(local_path)
                    yield f"data: {json.dumps({'event_type': 'sentence_end', 'index': idx}, ensure_ascii=False)}\n\n"
                else:
                    platform_url = await tts_manager.generate_voice(
                        text=segment_text,
                        voice_character=voice_character,
                        speed=speed,
                        disable_segmentation=True
                    )

                    track_first_packet()
                    yield f"data: {json.dumps({'event_type': 'voice_chunk', 'data_type': 'url', 'url': platform_url, 'text': segment_text, 'index': idx}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'event_type': 'sentence_end', 'index': idx}, ensure_ascii=False)}\n\n"
            except Exception as e:
                log.error(f"Failed to generate TTS voice for sentence '{segment_text}': {e}")
                raise e

        # 在连接结束前进行拼接合并，将完整版 oss_url 作为最后一个事件推送给前端，确保下一次播放无缝缓存
        try:
            if byte_stream and local_paths:
                oss_url = await self.merge_local_files_and_upload(task.id, local_paths)
            else:
                log.info(f"Generating full merged voice in background for task_id={task.id}")
                oss_url = await tts_manager.generate_voice(
                    text=cleaned_text,
                    voice_character=voice_character,
                    speed=speed,
                    disable_segmentation=True
                )
                async with mysql_connector.session_scope() as session:
                    task_db = await session.get(VoiceTTSTask, task.id)
                    if task_db:
                        task_db.oss_url = oss_url
                        task_db.status = "completed"
                        session.add(task_db)
                        await session.commit()
            
            if oss_url:
                yield f"data: {json.dumps({'event_type': 'merged_audio', 'url': oss_url}, ensure_ascii=False)}\n\n"
        except Exception as merge_err:
            log.error(f"Failed to merge task {task.id} in stream: {merge_err}")

        yield "data: [DONE]\n\n"

    async def merge_local_files_and_upload(self, task_id: int, local_paths: list[str]) -> Optional[str]:
        log.info(f"Direct local merge starting: Merge {len(local_paths)} local paths for task_id={task_id}")
        if not local_paths:
            return None

        import tempfile
        import shutil
        from utils.obs_utils import upload_audio

        temp_dir = tempfile.mkdtemp(prefix=f"tts_local_merge_{task_id}_")
        try:
            # 1. 生成 concat 配置文件
            list_file_path = os.path.join(temp_dir, "concat_list.txt")
            with open(list_file_path, "w", encoding="utf-8") as f:
                for lp in local_paths:
                    safe_path = lp.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            # 2. 使用 FFmpeg copy 拼接所有本地文件
            merged_output_path = os.path.join(temp_dir, "merged.mp3")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file_path, "-c", "copy", merged_output_path
            ]
            log.info(f"Running FFmpeg concat command (copy): {' '.join(cmd)}")
            
            import subprocess
            res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            if res.returncode != 0:
                err_msg = res.stderr.decode('utf-8', errors='ignore')
                log.warning(f"FFmpeg local copy concat failed, trying transcode fallback: {err_msg}")
                
                # Fallback: transcode instead of copy
                cmd_fallback = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_file_path, "-c:a", "libmp3lame", "-ab", "128k", merged_output_path
                ]
                log.info(f"Running FFmpeg concat command (transcode fallback): {' '.join(cmd_fallback)}")
                res = await asyncio.to_thread(subprocess.run, cmd_fallback, capture_output=True)
                if res.returncode != 0:
                    err_msg = res.stderr.decode('utf-8', errors='ignore')
                    log.error(f"FFmpeg local fallback concat failed for task_id={task_id}: {err_msg}")
                    raise RuntimeError(f"FFmpeg local concat failed: {err_msg}")

            # 3. 上传完整拼接音频到对象存储
            log.info(f"Uploading merged audio to OBS: {merged_output_path}")
            project_id = f"bubble_tts_{task_id}"
            oss_url = await upload_audio(merged_output_path, project_id=project_id)
            log.info(f"Local upload complete. URL: {oss_url}")

            # 4. 更新任务状态
            async with mysql_connector.session_scope() as session:
                task_db = await session.get(VoiceTTSTask, task_id)
                if task_db:
                    task_db.oss_url = oss_url
                    task_db.status = "completed"
                    session.add(task_db)
                    await session.commit()
            log.info(f"Task_id={task_id} successfully merged locally and uploaded.")
            return oss_url

        except Exception as e:
            log.error(f"Failed to merge locally and upload task_id={task_id}: {e}", exc_info=True)
            async with mysql_connector.session_scope() as session:
                task_db = await session.get(VoiceTTSTask, task_id)
                if task_db:
                    task_db.status = "failed"
                    session.add(task_db)
                    await session.commit()
            return None
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_err:
                log.warning(f"Failed to clean up local merge temp directory {temp_dir}: {cleanup_err}")
            # 延迟异步删除所有单句的本地临时文件，等待后台上传任务处理完成
            for lp in local_paths:
                try:
                    async def delayed_delete(filepath: str, delay: int = 20):
                        await asyncio.sleep(delay)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                    asyncio.create_task(delayed_delete(lp))
                except Exception as ex:
                    log.warning(f"Failed to trigger delete for {lp}: {ex}")

    async def merge_and_upload_task(self, task_id: int) -> Optional[str]:
        log.info(f"Background task starting: Merge TTS chunks for task_id={task_id}")
        
        async with mysql_connector.session_scope() as session:
            task = await session.get(VoiceTTSTask, task_id)
            if not task:
                log.error(f"Merge error: task_id={task_id} not found.")
                return None
            if task.oss_url and task.status == "completed":
                log.info(f"task_id={task_id} is already merged and completed. Skipping.")
                return task.oss_url

            stmt = select(VoiceTTSChunk).where(VoiceTTSChunk.task_id == task_id).order_by(VoiceTTSChunk.chunk_index)
            res = await session.execute(stmt)
            chunks = res.scalars().all()
            if not chunks:
                log.warning(f"No chunks found for task_id={task_id}. Mark as failed.")
                task.status = "failed"
                session.add(task)
                await session.commit()
                return None

        # 创建临时拼接文件夹
        import tempfile
        import shutil
        from utils.obs_utils import download_url_to_file, upload_audio

        temp_dir = tempfile.mkdtemp(prefix=f"tts_merge_{task_id}_")
        log.info(f"Created temp merge directory: {temp_dir}")

        local_files = []
        try:
            # 1. 下载每一个 chunk 音频
            for idx, chunk in enumerate(chunks):
                local_filename = f"part_{idx:03d}.mp3"
                local_path = os.path.join(temp_dir, local_filename)
                log.info(f"Downloading chunk {idx} from {chunk.platform_url} to {local_path}")
                await download_url_to_file(chunk.platform_url, local_path)
                local_files.append(local_path)

            # 2. 生成 concat 配置文件
            list_file_path = os.path.join(temp_dir, "concat_list.txt")
            with open(list_file_path, "w", encoding="utf-8") as f:
                for lf in local_files:
                    safe_path = lf.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            # 3. 使用 FFmpeg copy 拼接所有文件
            merged_output_path = os.path.join(temp_dir, "merged.mp3")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file_path, "-c", "copy", merged_output_path
            ]
            log.info(f"Running FFmpeg concat command (copy): {' '.join(cmd)}")
            
            import subprocess
            res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            if res.returncode != 0:
                err_msg = res.stderr.decode('utf-8', errors='ignore')
                log.warning(f"FFmpeg copy concat failed, trying transcode fallback: {err_msg}")
                
                # Fallback: transcode instead of copy
                cmd_fallback = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_file_path, "-c:a", "libmp3lame", "-ab", "128k", merged_output_path
                ]
                log.info(f"Running FFmpeg concat command (transcode fallback): {' '.join(cmd_fallback)}")
                res = await asyncio.to_thread(subprocess.run, cmd_fallback, capture_output=True)
                if res.returncode != 0:
                    err_msg = res.stderr.decode('utf-8', errors='ignore')
                    log.error(f"FFmpeg fallback concat failed for task_id={task_id}: {err_msg}")
                    raise RuntimeError(f"FFmpeg concat failed: {err_msg}")

            # 4. 上传完整拼接音频到对象存储
            log.info(f"Uploading merged audio to OBS: {merged_output_path}")
            project_id = f"bubble_tts_{task_id}"
            oss_url = await upload_audio(merged_output_path, project_id=project_id)
            log.info(f"Upload complete. URL: {oss_url}")

            # 5. 更新任务状态
            async with mysql_connector.session_scope() as session:
                task_db = await session.get(VoiceTTSTask, task_id)
                if task_db:
                    task_db.oss_url = oss_url
                    task_db.status = "completed"
                    session.add(task_db)
                    await session.commit()
            log.info(f"Task_id={task_id} successfully merged and uploaded.")
            return oss_url

        except Exception as e:
            log.error(f"Failed to merge and upload task_id={task_id}: {e}", exc_info=True)
            async with mysql_connector.session_scope() as session:
                task_db = await session.get(VoiceTTSTask, task_id)
                if task_db:
                    task_db.status = "failed"
                    session.add(task_db)
                    await session.commit()
            return None
        finally:
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Deleted temp directory: {temp_dir}")
            except Exception as cleanup_err:
                log.warning(f"Failed to clean up temp directory {temp_dir}: {cleanup_err}")

    async def warm_up_voice(self, voice_character: str):
        if not voice_character:
            return
        
        # 1. Resolve custom speaker id if any
        resolved_speaker = self._resolved_speaker_cache.get(voice_character)
        if not resolved_speaker:
            from models.sqlmodel.voice_cloned import VoiceCloned
            from models.sqlmodel.voice_designed import VoiceDesigned
            try:
                async with mysql_connector.session_scope() as session:
                    stmt_clone = select(VoiceCloned).where(
                        (VoiceCloned.custom_speaker_id == voice_character) |
                        (VoiceCloned.voice_character == voice_character)
                    )
                    res_clone = await session.execute(stmt_clone)
                    clone_record = res_clone.scalar_one_or_none()
                    if clone_record:
                        resolved_speaker = clone_record.custom_speaker_id
                        self._resolved_speaker_cache[voice_character] = resolved_speaker
                        log.info(f"[Pre-warm] Resolved custom cloned voice '{voice_character}' to '{resolved_speaker}'")
                    else:
                        stmt_design = select(VoiceDesigned).where(
                            (VoiceDesigned.custom_speaker_id == voice_character) |
                            (VoiceDesigned.voice_character == voice_character)
                        )
                        res_design = await session.execute(stmt_design)
                        design_record = res_design.scalar_one_or_none()
                        if design_record:
                            resolved_speaker = design_record.custom_speaker_id
                            self._resolved_speaker_cache[voice_character] = resolved_speaker
                            log.info(f"[Pre-warm] Resolved custom designed voice '{voice_character}' to '{resolved_speaker}'")
            except Exception as e:
                log.warning(f"[Pre-warm] Failed to resolve voice character in DB: {e}")
        
        lookup_name = resolved_speaker or voice_character
        
        # 2. Call TTSManager._resolve_engine_and_kwargs to populate engine cache
        from services.tts_services import tts_manager
        try:
            engine_name = await tts_manager._resolve_engine_and_kwargs(lookup_name, {})
            log.info(f"[Pre-warm] Cached engine for '{lookup_name}': {engine_name}")
            if resolved_speaker and resolved_speaker != voice_character:
                cached = tts_manager._engine_cache.get(resolved_speaker)
                if cached:
                    tts_manager._engine_cache[voice_character] = cached
                    log.info(f"[Pre-warm] Cached engine for original name '{voice_character}' pointing to '{resolved_speaker}'")
            
            # Warm up the engine connection pool in the background
            if engine_name:
                asyncio.create_task(tts_manager.warm_up_engine(engine_name))
        except Exception as e:
            log.warning(f"[Pre-warm] Failed to cache engine details for '{lookup_name}': {e}")

    async def heal_pending_tasks_on_startup(self):
        log.info("Startup healing: Checking for pending/unmerged VoiceTTSTasks...")
        try:
            # 延时等待数据库与 connector 完全就绪
            await asyncio.sleep(5)
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceTTSTask).where(
                    (VoiceTTSTask.oss_url == None) | (VoiceTTSTask.oss_url == ""),
                    VoiceTTSTask.status != "failed"
                )
                res = await session.execute(stmt)
                tasks = res.scalars().all()
                if not tasks:
                    log.info("No unmerged VoiceTTSTasks found on startup.")
                    return
                
                log.info(f"Found {len(tasks)} unmerged VoiceTTSTasks on startup. Healing them in background...")
                for task in tasks:
                    asyncio.create_task(self.merge_and_upload_task(task.id))
        except Exception as e:
            log.warning(f"Error during startup healing of VoiceTTSTasks: {e}")


# 单例服务导出
voice_tts_service = VoiceTTSService()
