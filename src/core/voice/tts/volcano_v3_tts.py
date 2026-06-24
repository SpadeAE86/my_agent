import os
import sys
import time
import uuid
import json
import base64
import logging
import asyncio
import random
import requests
import subprocess
import httpx
from typing import Dict, Optional, Any, AsyncGenerator

# Ensure project src folder is in sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.voice.tts.tts_manager import BaseTTSEngine
from utils.obs_utils import upload_audio
from config.config import MY_CONFIG, ENV
from exceptions.infra import ServiceException

log = logging.getLogger("volcano_v3_tts")


class VolcanoV3TTSEngine(BaseTTSEngine):
    def __init__(self):
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def warm_up(self) -> None:
        """
        Warm up the HTTP connection pool by making a lightweight request to the API host.
        """
        try:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
            
            volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
            url = os.getenv("VOLCANO_V3_SSE_HOST") or volcano_config.get("v3_sse_host", "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
            
            log.info(f"Warming up Volcano V3 connection to host: {url}")
            response = await self._client.get(url, timeout=3.0)
            log.info(f"Volcano V3 connection warmed up. Status code: {response.status_code}")
        except Exception as e:
            log.info(f"Volcano V3 connection warming finished (expected error/status): {e}")

    def _resolve_resource_id(self, voice_type: str) -> str:
        from models.volcano_online_voice_enums import character_options
        is_custom_speaker = (
            voice_type not in character_options.values() or 
            any(voice_type.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_", "S_"])
        )
        if is_custom_speaker:
            return "seed-icl-2.0"

        vt_lower = voice_type.lower()
        if "uranus" in vt_lower or "saturn" in vt_lower or "jupiter" in vt_lower:
            return "seed-tts-2.0"
        elif "mars" in vt_lower or "wvae" in vt_lower:
            return "seed-tts-1.0"
        return "seed-tts-2.0"

    def _get_v3_headers(self, volcano_config: dict, request_id: str, resource_id: str) -> dict:
        api_key = os.getenv("VOLCANO_API_KEY")
        if not api_key:
            api_key = volcano_config.get("api_key")
            if api_key and api_key.startswith("YOUR_"):
                api_key = None

        app_id = os.getenv("VOLCANO_APP_ID")
        if not app_id:
            app_id = volcano_config.get("app_id") or volcano_config.get("appid")
            if app_id and str(app_id).startswith("YOUR_"):
                app_id = None

        access_token = os.getenv("VOLCANO_ACCESS_TOKEN")
        if not access_token:
            access_token = volcano_config.get("access_token")
            if access_token and access_token.startswith("YOUR_"):
                access_token = None

        if api_key:
            return {
                "X-Api-Key": api_key,
                "X-Api-Resource-Id": resource_id,
                "X-Api-Request-Id": request_id,
                "Content-Type": "application/json",
            }

        if app_id and access_token:
            return {
                "X-Api-App-Id": str(app_id),
                "X-Api-Access-Key": access_token,
                "X-Api-Resource-Id": resource_id,
                "X-Api-Request-Id": request_id,
                "Content-Type": "application/json",
            }

        raise ServiceException(
            code=450,
            message="Volcano V3 authentication requires api_key, or app_id + access_token in environment or config.yml",
        )

    def _transcode_audio(self, source_audio: str, output_audio: str, file_format: str) -> str:
        codec = "libmp3lame" if file_format == "mp3" else "pcm_s16le"
        cmd = [
            "ffmpeg", "-y", "-i", source_audio, "-c:a", codec, output_audio
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            log.error(f"FFmpeg transcode failed: {e}")
            raise RuntimeError(f"音频转码失败: {e}")
        return output_audio

    def _sync_generate(
        self,
        text: str,
        voice_type: str,
        resource_id: str,
        speed_rate: int,
        loudness_rate: int,
        instruct: Optional[str],
        temp_dir: str,
        project_id: str,
        **kwargs
    ) -> str:
        volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
        url = os.getenv("VOLCANO_V3_SSE_HOST") or volcano_config.get("v3_sse_host", "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
        request_id = str(uuid.uuid4())

        headers = self._get_v3_headers(volcano_config, request_id, resource_id)

        # Build audio_params
        audio_params = {
            "format": kwargs.get("audio_format", "mp3"),
            "sample_rate": kwargs.get("sample_rate", 24000),
            "speech_rate": speed_rate,
            "loudness_rate": loudness_rate,
        }
        
        # Enable timestamp or subtitle based on model generation
        if "2.0" in resource_id:
            audio_params["enable_subtitle"] = True
        else:
            audio_params["enable_timestamp"] = True

        # Check for model parameter override
        model = kwargs.get("model")
        if not model and "2.0" in resource_id:
            if instruct:
                model = "seed-tts-2.0-expressive"
            else:
                model = "seed-tts-2.0-standard"

        req_params = {
            "text": text,
            "audio_params": audio_params,
            "additions": json.dumps({
                "disable_markdown_filter": kwargs.get("disable_markdown_filter", True)
            })
        }
        req_params["speaker"] = voice_type

        payload = {
            "user": {"uid": f"user_{uuid.uuid4().hex[:8]}"},
            "namespace": "BidirectionalTTS",
            "req_params": req_params
        }

        if model:
            payload["req_params"]["model"] = model

        if instruct:
            payload["req_params"]["context_texts"] = [instruct]

        log.info(f"Volcano V3 SSE Request: url={url}, resource_id={resource_id}, payload={payload}")

        raw_audio_path = os.path.join(temp_dir, f"raw_{project_id}.mp3")
        audio_data = bytearray()
        sentences = []

        t_start = time.time()
        first_packet_time = None

        with requests.Session() as session:
            response = session.post(url, headers=headers, json=payload, stream=True, timeout=(10, 300))
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                body = response.text[:2000]
                log.error(f"Volcano V3 SSE API error body: {body}")
                raise ServiceException(code=450, message=f"Volcano V3 request failed: {exc}", detail=body) from exc

            current_event = None
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                payload_text = line.split(":", 1)[1].strip()
                try:
                    event_payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    log.warning(f"Unexpected non-json volcano event payload: {payload_text[:500]}")
                    continue

                # Log errors
                is_error_code = False
                code = event_payload.get("code")
                if code is not None and code not in (0, 3000, 20000000):
                    is_error_code = True

                if current_event in ("error", "153", "SessionFailed") or is_error_code:
                    err_code = code if code is not None else 500
                    err_msg = event_payload.get("message", "Unknown Error")
                    raise RuntimeError(f"Volcano V3 SSE returned error: code={err_code}, message={err_msg}")

                if event_payload.get("data"):
                    if first_packet_time is None:
                        first_packet_time = time.time()
                        ttft = int((first_packet_time - t_start) * 1000)
                        msg = f"[Metric] Volcano V3 SSE. RequestID: {request_id}, TTFT (后端首包延迟): {ttft}ms"
                        log.info(msg)
                        print(msg, flush=True)
                    audio_data.extend(base64.b64decode(event_payload["data"]))

                # Capture sentence/timestamps event if any
                sentence_data = event_payload.get("sentence")
                if isinstance(sentence_data, dict):
                    sentences.append(sentence_data)

        if not audio_data:
            raise RuntimeError("No audio data received from Volcano V3 SSE API.")

        with open(raw_audio_path, "wb") as f:
            f.write(audio_data)

        log.info(f"Volcano V3 SSE success. Sentences received: {len(sentences)}. Raw audio saved to {raw_audio_path}")
        return raw_audio_path

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
        # Resolve voice type
        from models.volcano_online_voice_enums import character_options
        voice_type = character_options.get(voice_character)
        if not voice_type:
            if voice_character in character_options.values():
                voice_type = voice_character
            elif any(voice_character.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_", "S_"]):
                voice_type = voice_character
            else:
                from models.sqlmodel.voice_cloned import VoiceCloned
                from models.sqlmodel.voice_designed import VoiceDesigned
                from infra.storage.mysql_connector import mysql_connector
                from sqlmodel import select
                
                is_custom = False
                try:
                    async with mysql_connector.session_scope() as session:
                        stmt_clone = select(VoiceCloned).where(
                            (VoiceCloned.custom_speaker_id == voice_character) |
                            (VoiceCloned.voice_character == voice_character)
                        )
                        res_clone = await session.execute(stmt_clone)
                        if res_clone.scalar_one_or_none():
                            is_custom = True
                        else:
                            stmt_design = select(VoiceDesigned).where(
                                (VoiceDesigned.custom_speaker_id == voice_character) |
                                (VoiceDesigned.voice_character == voice_character)
                            )
                            res_design = await session.execute(stmt_design)
                            if res_design.scalar_one_or_none():
                                is_custom = True
                except Exception as db_err:
                    log.warning(f"Failed to check custom speaker '{voice_character}' in DB: {db_err}")
                
                if is_custom:
                    voice_type = voice_character
                else:
                    log.warning(f"Voice character '{voice_character}' not found in enums/DB, fallback to Vivi 2.0")
                    voice_type = character_options.get("Vivi 2.0", "zh_female_vv_uranus_bigtts")

        # Resolve resource_id
        resource_id = kwargs.get("resource_id") or os.getenv("VOLCANO_RESOURCE_ID")
        if not resource_id:
            resource_id = self._resolve_resource_id(voice_type)

        # Map speed & volume to rate offsets [-50, 100] / [-50, 50]
        speech_rate = kwargs.get("speech_rate")
        if speech_rate is None:
            speech_rate = int((speed - 1.0) * 100)
            speech_rate = max(-50, min(100, speech_rate))

        loudness_rate = kwargs.get("loudness_rate")
        if loudness_rate is None:
            loudness_rate = int((volume - 1.0) * 50)
            loudness_rate = max(-50, min(50, loudness_rate))

        instruct = kwargs.get("instruct") or kwargs.get("voice_design_instruct")

        project_id = f"volco_v3_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        temp_dir = os.path.join("./final", project_id)
        os.makedirs(temp_dir, exist_ok=True)

        async with self._lock:
            try:
                # Run blocking request in thread pool
                # Avoid passing duplicate parameters in **kwargs to to_thread
                sync_kwargs = kwargs.copy()
                sync_kwargs.pop("text", None)
                sync_kwargs.pop("voice_type", None)
                sync_kwargs.pop("resource_id", None)
                sync_kwargs.pop("speed_rate", None)
                sync_kwargs.pop("loudness_rate", None)
                sync_kwargs.pop("instruct", None)
                sync_kwargs.pop("voice_design_instruct", None)
                sync_kwargs.pop("temp_dir", None)
                sync_kwargs.pop("project_id", None)

                raw_audio_path = await asyncio.to_thread(
                    self._sync_generate,
                    text=text,
                    voice_type=voice_type,
                    resource_id=resource_id,
                    speed_rate=speech_rate,
                    loudness_rate=loudness_rate,
                    instruct=instruct,
                    temp_dir=temp_dir,
                    project_id=project_id,
                    **sync_kwargs
                )

                # Transcode
                final_mp3_path = os.path.join(temp_dir, f"voice_{project_id}.mp3")
                await asyncio.to_thread(self._transcode_audio, raw_audio_path, final_mp3_path, "mp3")

                # Upload
                audio_url = await upload_audio(final_mp3_path, project_id=project_id)
                return audio_url

            except Exception as e:
                log.error(f"Volcano V3 generation failed: {e}")
                raise ServiceException(code=500, message=f"语音合成失败(V3): {e}")
            finally:
                # Clean up
                async def delayed_delete(path: str, delay: int = 300):
                    await asyncio.sleep(delay)
                    try:
                        import shutil
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.isfile(path):
                            os.remove(path)
                    except Exception as ex:
                        log.warning(f"Failed to clean up V3 temp path {path}: {ex}")

                asyncio.create_task(delayed_delete(temp_dir, delay=300))

    async def generate_voice_stream(
        self,
        text: str,
        voice_character: str,
        speed: float = 1.0,
        volume: float = 1.0,
        emotion: str = "neutral",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        # Resolve voice type
        from models.volcano_online_voice_enums import character_options
        voice_type = character_options.get(voice_character)
        if not voice_type:
            if voice_character in character_options.values():
                voice_type = voice_character
            elif any(voice_character.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_", "S_"]):
                voice_type = voice_character
            else:
                from models.sqlmodel.voice_cloned import VoiceCloned
                from models.sqlmodel.voice_designed import VoiceDesigned
                from infra.storage.mysql_connector import mysql_connector
                from sqlmodel import select
                
                is_custom = False
                try:
                    async with mysql_connector.session_scope() as session:
                        stmt_clone = select(VoiceCloned).where(
                            (VoiceCloned.custom_speaker_id == voice_character) |
                            (VoiceCloned.voice_character == voice_character)
                        )
                        res_clone = await session.execute(stmt_clone)
                        if res_clone.scalar_one_or_none():
                            is_custom = True
                        else:
                            stmt_design = select(VoiceDesigned).where(
                                (VoiceDesigned.custom_speaker_id == voice_character) |
                                (VoiceDesigned.voice_character == voice_character)
                            )
                            res_design = await session.execute(stmt_design)
                            if res_design.scalar_one_or_none():
                                is_custom = True
                except Exception as db_err:
                    log.warning(f"Failed to check custom speaker '{voice_character}' in DB: {db_err}")
                
                if is_custom:
                    voice_type = voice_character
                else:
                    log.warning(f"Voice character '{voice_character}' not found in enums/DB, fallback to Vivi 2.0")
                    voice_type = character_options.get("Vivi 2.0", "zh_female_vv_uranus_bigtts")

        # Resolve resource_id
        resource_id = kwargs.get("resource_id") or os.getenv("VOLCANO_RESOURCE_ID")
        if not resource_id:
            resource_id = self._resolve_resource_id(voice_type)

        speech_rate = kwargs.get("speech_rate")
        if speech_rate is None:
            speech_rate = int((speed - 1.0) * 100)
            speech_rate = max(-50, min(100, speech_rate))

        loudness_rate = kwargs.get("loudness_rate")
        if loudness_rate is None:
            loudness_rate = int((volume - 1.0) * 50)
            loudness_rate = max(-50, min(50, loudness_rate))

        instruct = kwargs.get("instruct") or kwargs.get("voice_design_instruct")

        # Build payload & headers
        volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
        url = os.getenv("VOLCANO_V3_SSE_HOST") or volcano_config.get("v3_sse_host", "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
        request_id = str(uuid.uuid4())
        headers = self._get_v3_headers(volcano_config, request_id, resource_id)

        audio_params = {
            "format": kwargs.get("audio_format", "mp3"),
            "sample_rate": kwargs.get("sample_rate", 24000),
            "speech_rate": speech_rate,
            "loudness_rate": loudness_rate,
        }
        if "2.0" in resource_id:
            audio_params["enable_subtitle"] = True
        else:
            audio_params["enable_timestamp"] = True

        model = kwargs.get("model")
        if not model and "2.0" in resource_id:
            model = "seed-tts-2.0-expressive" if instruct else "seed-tts-2.0-standard"

        req_params = {
            "text": text,
            "audio_params": audio_params,
            "additions": json.dumps({
                "disable_markdown_filter": kwargs.get("disable_markdown_filter", True)
            })
        }
        req_params["speaker"] = voice_type

        payload = {
            "user": {"uid": f"user_{uuid.uuid4().hex[:8]}"},
            "namespace": "BidirectionalTTS",
            "req_params": req_params
        }
        if model:
            payload["req_params"]["model"] = model
        if instruct:
            payload["req_params"]["context_texts"] = [instruct]

        log.info(f"Volcano V3 SSE Stream: url={url}, resource_id={resource_id}")

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

        import base64
        t_start = time.time()
        first_packet_time = None

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                log.error(f"Volcano V3 SSE stream failed: status={response.status_code}, body={body[:500]}")
                raise RuntimeError(f"Volcano V3 SSE stream failed with status {response.status_code}")

            current_event = None
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                payload_text = line.split(":", 1)[1].strip()
                try:
                    event_payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue

                code = event_payload.get("code")
                if code is not None and code not in (0, 3000, 20000000):
                    err_msg = event_payload.get("message", "Unknown Error")
                    raise RuntimeError(f"Volcano V3 SSE stream returned error: code={code}, message={err_msg}")

                if current_event in ("error", "153", "SessionFailed"):
                    err_msg = event_payload.get("message", "Unknown Error")
                    raise RuntimeError(f"Volcano V3 SSE stream event error: {err_msg}")

                if event_payload.get("data"):
                    if first_packet_time is None:
                        first_packet_time = time.time()
                        ttft = int((first_packet_time - t_start) * 1000)
                        msg = f"[Metric] Volcano V3 SSE Stream. RequestID: {request_id}, TTFT (后端首包延迟): {ttft}ms"
                        log.info(msg)
                        print(msg, flush=True)
                    yield base64.b64decode(event_payload["data"])


if __name__ == "__main__":
    import argparse
    import sys
    # Add project src directory to sys.path to allow standalone execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    
    # Configure root logging to output directly to stdout for standalone testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    async def run_standalone_test():
        parser = argparse.ArgumentParser(description="Test Volcano V3 TTS Engine")
        parser.add_argument("--text", type=str, default="那个……请问，我、我可以坐在你旁边吗？我有点害怕……", help="Text to synthesize")
        parser.add_argument("--voice", type=str, default="柔美女友 2.0", help="Voice character name")
        parser.add_argument("--instruct", type=str, default="极其害羞、语速缓慢地颤抖说话", help="Style prompt/instruction")

        args = parser.parse_args()

        print(f"Standalone VolcanoV3TTSEngine Test:")
        print(f"  Voice:    {args.voice}")
        print(f"  Instruct: {args.instruct}")
        print(f"  Text:     {args.text}")

        engine = VolcanoV3TTSEngine()
        try:
            url = await engine.generate_voice(
                text=args.text,
                voice_character=args.voice,
                instruct=args.instruct
            )
            print(f"\n[SUCCESS] Synthesized audio uploaded successfully!")
            print(f"URL: {url}")
        except Exception as e:
            print(f"\n[FAIL] Standalone synthesis failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_standalone_test())
