import os
import time
import uuid
import random
import asyncio
import subprocess
import json
import shutil
import re
from datetime import datetime
from pathlib import Path
from config.config import ENV, MY_CONFIG
from exceptions.infra import ServiceException
from models.volcano_online_voice_enums import character_options
from utils.obs_utils import upload_audio
from utils.volcano_utils import volcano_generate_voice, parse_frontend_words
from infra.logging.logger import logger as log


class VolcoVoiceService:
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", file_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(res.stdout)
            return float(data["format"]["duration"])
        except Exception as e:
            log.warning(f"Failed to get audio duration via ffprobe: {e}")
            return 0.0

    @staticmethod
    def transcode_audio(source_audio: str, output_audio: str, file_format: str) -> str:
        codec = "libmp3lame" if file_format == "mp3" else "pcm_s16le"
        cmd = [
            "ffmpeg", "-y", "-i", source_audio, "-c:a", codec, output_audio
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            log.error(f"FFmpeg transcode failed: {e}")
            raise ServiceException(code=500, message=f"音频转码失败: {e}")
        return output_audio

    @staticmethod
    async def delayed_delete(path: str, delay: int = 50):
        await asyncio.sleep(delay)
        try:
            if os.path.isdir(path):
                await asyncio.to_thread(shutil.rmtree, path)
            elif os.path.isfile(path):
                await asyncio.to_thread(os.remove, path)
            log.info(f"Cleaned up temp path: {path}")
        except Exception as e:
            log.warning(f"Failed to delete temp path {path}: {e}")

    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        过滤并清除 Markdown 语法（如标题、列表、粗体、链接、代码块、表情符号等），以便 TTS 朗读更加自然。
        """
        if not text:
            return ""

        # 1. 移除 Markdown 代码块: ```language ... ```
        text = re.sub(r"```[\s\S]*?```", "", text)

        # 2. 移除行内代码: `code` -> code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # 3. 移除标题符号: #, ##, ### 等（行首匹配）
        text = re.sub(r"^\s*#+\s+", "", text, flags=re.MULTILINE)

        # 4. 移除 Markdown 链接: [anchor](url) -> anchor
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # 5. 移除粗体/斜体: **, *, __, _, ***
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

        # 6. 移除删除线: ~~text~~ -> text
        text = re.sub(r"~~([^~]+)~~", r"\1", text)

        # 7. 移除无序列表符号: - , * , + 以及有序列表符号 \d+. （行首匹配）
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # 8. 移除引用符号: > （行首匹配）
        text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)

        # 9. 清理常见表情符号和特殊的补充平面字符，防止火山 TTS 解析异常或念出符号名称
        emoji_pattern = re.compile(
            "["
            "\U00010000-\U0010ffff"
            "\u2600-\u27bf"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r"", text)

        # 10. 整理换行，合并为自然的停顿逗号或句号
        lines = [line.strip() for line in text.splitlines()]
        non_empty_lines = []
        for line in lines:
            if not line:
                continue
            # 若行尾没有标点符号，则补上句号，让 TTS 生成在段落间产生自然的呼吸停顿
            if line[-1] not in "。！？，；、：?.!;:,":
                line += "。"
            non_empty_lines.append(line)
        
        text = "".join(non_empty_lines)

        # 11. 净化连续重复的标点
        text = re.sub(r"[，。！？、]{2,}", lambda m: m.group(0)[0], text)

        return text.strip()

    @classmethod
    async def generate_voice(
        cls,
        text: str,
        voice_character: str = "Vivi",
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        disable_segmentation: bool = True
    ) -> str:
        """
        生成语音并上传到 OBS/TOS，返回在线 URL。
        在传递给火山接口前会对文本进行 Markdown 净化处理。
        """
        # 0. 预处理：过滤 Markdown 字符，使朗读更连贯
        cleaned_text = cls.clean_markdown(text)
        if not cleaned_text:
            cleaned_text = "没有可朗读的文本。"

        # 1. 动态从数据库加载音色代码，如果不存在则回退至硬编码静态映射
        from models.sqlmodel.voice_timbre import VoiceTimbre
        from infra.storage.mysql_connector import mysql_connector
        from sqlmodel import select

        voice_type = None
        try:
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceTimbre).where(VoiceTimbre.voice_character == voice_character, VoiceTimbre.is_enabled == True)
                result = await session.execute(stmt)
                timbre = result.scalar_one_or_none()
                if timbre:
                    voice_type = timbre.voice_code
        except Exception as e:
            log.warning(f"Failed to fetch voice code from database for character '{voice_character}': {e}")

        if not voice_type:
            if voice_character not in character_options:
                log.warning(f"Voice character '{voice_character}' not found, fallback to Vivi")
                voice_character = "Vivi"
            voice_type = character_options[voice_character]

        project_id = f"volco_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        temp_dir = os.path.join("./final", project_id)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            raw_audio_output = os.path.join(temp_dir, f"raw_{project_id}.mp3")
            
            # 2. 生成语音音频 (格式为 mp3)
            log.info(f"Generating voice for: '{cleaned_text[:20]}...' using timbre: {voice_character} ({voice_type})")
            generate_result = await volcano_generate_voice(
                voice_type=voice_type,
                text=cleaned_text,
                filename=raw_audio_output,
                speed=speed,
                volume=volume,
                emotion=emotion,
                emotion_active=False
            )

            # 3. 如果不分句 (disable_segmentation=True)，直接上传音频并返回
            if disable_segmentation:
                final_audio = os.path.join(temp_dir, f"voice_{project_id}.mp3")
                await asyncio.to_thread(cls.transcode_audio, raw_audio_output, final_audio, "mp3")
                audio_url = await upload_audio(final_audio, project_id=project_id)
                log.info(f"Generated single voice URL: {audio_url}")
                return audio_url

            # 4. 如果需要分句 (分句逻辑兼容，但仍最终返回主音频)
            words = parse_frontend_words(generate_result.frontend_payloads)
            STRONG_PUNCTUATION = "\u3002\uff01\uff1f!?\uff1b;"
            COMMA_PUNCTUATION = "\uff0c,\u3001\uff1a:"
            
            segments = []
            current_words = []
            
            def flush_segment():
                nonlocal current_words
                if not current_words:
                    return
                start_ms = float(current_words[0].start_time)
                end_ms = float(current_words[-1].end_time)
                caption_text = "".join(word.word for word in current_words)
                segments.append({
                    "caption_text": caption_text,
                    "start_ms": start_ms,
                    "end_ms": end_ms
                })
                current_words = []

            for idx, word in enumerate(words):
                current_words.append(word)
                if word.word and (word.word[-1] in STRONG_PUNCTUATION or word.word[-1] in COMMA_PUNCTUATION):
                    flush_segment()
            flush_segment()

            final_audio = os.path.join(temp_dir, f"voice_{project_id}.mp3")
            await asyncio.to_thread(cls.transcode_audio, raw_audio_output, final_audio, "mp3")
            audio_url = await upload_audio(final_audio, project_id=project_id)
            return audio_url

        except Exception as e:
            log.error(f"Volcano voice service generation failed: {e}")
            raise ServiceException(code=500, message=f"语音合成失败: {e}")
        finally:
            asyncio.create_task(cls.delayed_delete(temp_dir, delay=30))

    @classmethod
    async def seed_default_timbres_if_empty(cls):
        """
        如果数据库中的 voice_timbre 表没有任何记录，则自动解析并导入
        C:\\Users\\admin\\Downloads\\volcovoice_sample.sql 中的音色配置。
        """
        from models.sqlmodel.voice_timbre import VoiceTimbre
        from infra.storage.mysql_connector import mysql_connector
        from sqlmodel import select

        try:
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceTimbre).limit(1)
                result = await session.execute(stmt)
                if result.scalar_one_or_none() is not None:
                    log.info("voice_timbre 数据表已包含记录，跳过自动初始化。")
                    return
        except Exception as e:
            log.warning(f"无法检查 voice_timbre 表状态，可能表未就绪: {e}")
            return

        sql_path = r"C:\Users\admin\Downloads\volcovoice_sample.sql"
        if not os.path.exists(sql_path):
            log.warning(f"未找到火山音色初始化 SQL 文件 ({sql_path})，跳过音色初始化。")
            return

        log.info(f"开始解析并从 {sql_path} 初始化音色配置表...")
        try:
            with open(sql_path, "r", encoding="utf-8") as f:
                content = f.read()

            inserts = re.findall(
                r"INSERT INTO volcovoice_sample\s+\((.*?)\)\s+VALUES\s+\((.*?)\);",
                content,
                re.DOTALL | re.IGNORECASE
            )

            def parse_sql_values(values_str):
                values = []
                i = 0
                n = len(values_str)
                if values_str.startswith('('):
                    values_str = values_str[1:]
                    n -= 1
                if values_str.endswith(')'):
                    values_str = values_str[:-1]
                    n -= 1
                while i < n:
                    while i < n and (values_str[i].isspace() or values_str[i] == ','):
                        i += 1
                    if i >= n:
                        break
                    if values_str[i] == "'":
                        i += 1
                        char_buf = []
                        while i < n:
                            if values_str[i] == "\\":
                                if i + 1 < n:
                                    char_buf.append(values_str[i+1])
                                    i += 2
                                else:
                                    char_buf.append("\\")
                                    i += 1
                            elif values_str[i] == "'":
                                if i + 1 < n and values_str[i+1] == "'":
                                    char_buf.append("'")
                                    i += 2
                                else:
                                    i += 1
                                    break
                            else:
                                char_buf.append(values_str[i])
                                i += 1
                        values.append("".join(char_buf))
                    else:
                        start = i
                        while i < n and values_str[i] != ',' and values_str[i] != ')':
                            i += 1
                        val_token = values_str[start:i].strip()
                        if val_token.upper() == 'NULL':
                            values.append(None)
                        elif val_token.isdigit():
                            values.append(int(val_token))
                        else:
                            try:
                                values.append(float(val_token))
                            except ValueError:
                                values.append(val_token)
                return values

            timbres_to_insert = []
            for cols_str, vals_str in inserts:
                columns = [c.strip().strip('`') for c in cols_str.split(',')]
                values = parse_sql_values('(' + vals_str + ')')
                if len(columns) != len(values):
                    continue

                record = dict(zip(columns, values))
                
                is_enabled_val = record.get('is_enabled', 1)
                if isinstance(is_enabled_val, str):
                    is_enabled = is_enabled_val.strip() not in ('0', 'False', 'false')
                else:
                    is_enabled = bool(is_enabled_val)

                def parse_dt(dt_str):
                    if not dt_str:
                        return datetime.now()
                    try:
                        return datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return datetime.now()

                sex_val = record.get("sex")
                if sex_val in ("女", "female", "0", "女/0"):
                    normalized_sex = "0"
                elif sex_val in ("男", "male", "1", "男/1"):
                    normalized_sex = "1"
                else:
                    normalized_sex = sex_val

                timbre = VoiceTimbre(
                    voice_character=record.get("voice_character"),
                    voice_code=record.get("voice_code"),
                    voice_model_type=record.get("voice_model_type", "default"),
                    is_enabled=is_enabled,
                    note=record.get("note"),
                    priority=int(record.get("priority", 0)),
                    age_type=record.get("age_type"),
                    sex=normalized_sex,
                    full_voice=record.get("full_voice"),
                    provider="volcano",
                    is_online=True,
                    created_at=parse_dt(record.get("created_at")),
                    updated_at=parse_dt(record.get("updated_at")),
                )
                timbres_to_insert.append(timbre)

            if timbres_to_insert:
                async with mysql_connector.session_scope() as session:
                    session.add_all(timbres_to_insert)
                    await session.commit()
                log.info(f"成功导入 {len(timbres_to_insert)} 个火山音色配置记录到数据库！")
            else:
                log.warning("没有可导入的有效音色配置记录。")
        except Exception as e:
            log.error(f"导入火山音色配置发生异常: {e}", exc_info=True)

    @classmethod
    async def seed_qwen_timbres_if_missing(cls):
        """
        Seed Qwen online voices from config into the voice_timbre database table.
        """
        from models.sqlmodel.voice_timbre import VoiceTimbre
        from infra.storage.mysql_connector import mysql_connector
        from sqlmodel import select
        from models.qwen_online_voice_enums import qwen_online_voice_options
        from models.qwen_online_voice_enums_meta import qwen_online_voice_enums_meta

        try:
            async with mysql_connector.session_scope() as session:
                stmt = select(VoiceTimbre).where(VoiceTimbre.provider == "qwen")
                result = await session.execute(stmt)
                if result.scalars().first() is not None:
                    log.info("Qwen online voice timbres already seeded, skipping.")
                    return

                timbres_to_insert = []
                for name, code in qwen_online_voice_options.items():
                    meta = qwen_online_voice_enums_meta.get(name, {})
                    sex_val = meta.get("gender", "女")
                    if sex_val in ("女", "female", "0", "女/0"):
                        normalized_sex = "0"
                    elif sex_val in ("男", "male", "1", "男/1"):
                        normalized_sex = "1"
                    else:
                        normalized_sex = sex_val

                    timbre = VoiceTimbre(
                        voice_character=name,
                        voice_code=code,
                        voice_model_type="default",
                        is_enabled=True,
                        note=meta.get("description", ""),
                        priority=0,
                        age_type=meta.get("age", "青年"),
                        sex=normalized_sex,
                        full_voice=None,
                        provider="qwen",
                        is_online=True
                    )
                    timbres_to_insert.append(timbre)

                if timbres_to_insert:
                    session.add_all(timbres_to_insert)
                    await session.commit()
                    log.info(f"成功导入 {len(timbres_to_insert)} 个Qwen在线音色配置记录到数据库！")
        except Exception as e:
            log.error(f"导入Qwen在线音色配置发生异常: {e}", exc_info=True)

    @classmethod
    async def seed_volcano_timbres_if_missing(cls):
        """
        Ensure all Volcano online voices from enums are seeded in the database.
        """
        from models.sqlmodel.voice_timbre import VoiceTimbre
        from infra.storage.mysql_connector import mysql_connector
        from sqlmodel import select
        from models.volcano_online_voice_enums import character_options, get_volcano_voice_type
        from models.volcano_online_voice_enums_meta import voice_enums_meta

        try:
            async with mysql_connector.session_scope() as session:
                # Get all existing volcano voice characters in the DB (lowercased for case-insensitive check)
                stmt = select(VoiceTimbre.voice_character).where(VoiceTimbre.provider == "volcano")
                res = await session.execute(stmt)
                existing_chars = {c.lower() for c in res.scalars().all()}

                inserted_chars_lower = set(existing_chars)
                timbres_to_insert = []
                for name, code in character_options.items():
                    name_lower = name.lower()
                    if name_lower in inserted_chars_lower:
                        continue
                    
                    inserted_chars_lower.add(name_lower)
                    meta = voice_enums_meta.get(name, {})
                    try:
                        voice_model_type = get_volcano_voice_type(name)
                    except KeyError:
                        voice_model_type = "big"

                    sex_val = meta.get("gender")
                    if sex_val in ("女", "female", "0", "女/0"):
                        normalized_sex = "0"
                    elif sex_val in ("男", "male", "1", "男/1"):
                        normalized_sex = "1"
                    else:
                        normalized_sex = sex_val

                    timbre = VoiceTimbre(
                        voice_character=name,
                        voice_code=code,
                        voice_model_type=voice_model_type,
                        is_enabled=True,
                        note=None,
                        priority=0,
                        age_type=meta.get("age"),
                        sex=normalized_sex,
                        full_voice=None,
                        provider="volcano",
                        is_online=True
                    )
                    timbres_to_insert.append(timbre)

                if timbres_to_insert:
                    session.add_all(timbres_to_insert)
                    await session.commit()
                    log.info(f"成功导入 {len(timbres_to_insert)} 个新火山音色配置记录到数据库！")
        except Exception as e:
            log.error(f"增量导入火山音色配置发生异常: {e}", exc_info=True)

