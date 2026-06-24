import os
import uuid
import json
import logging
import asyncio
import inspect
import websockets
from typing import Dict, Optional, Any, AsyncGenerator

from config.config import MY_CONFIG
from exceptions.infra import ServiceException
from sdk.volcano_protocols import (
    MsgType,
    MsgTypeFlagBits,
    EventType,
    Message,
    receive_message,
    start_connection,
    finish_connection,
    start_session,
    finish_session,
    cancel_session,
    wait_for_event,
)

log = logging.getLogger("volcano_v3_bidirectional")


class VolcanoV3BidirectionalEngine:
    def __init__(self):
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

    def _resolve_resource_id(self, voice_type: str) -> str:
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

    async def connect(self, voice_character: str, **kwargs) -> None:
        """
        Establish V3 Bidirectional WebSocket connection.
        """
        volcano_config = MY_CONFIG.get("audio", {}).get("Volcano", {})
        host = os.getenv("VOLCANO_V3_WS_HOST") or volcano_config.get("v3_ws_host", "wss://openspeech.bytedance.com/api/v3/tts/bidirection")
        
        # Resolve voice type
        from models.volcano_online_voice_enums import character_options
        voice_type = character_options.get(voice_character)
        if not voice_type:
            if voice_character in character_options.values():
                voice_type = voice_character
            elif any(voice_character.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_"]):
                voice_type = voice_character
            else:
                voice_type = character_options.get("Vivi 2.0", "zh_female_vv_uranus_bigtts")

        resource_id = kwargs.get("resource_id") or os.getenv("VOLCANO_RESOURCE_ID") or self._resolve_resource_id(voice_type)
        request_id = str(uuid.uuid4())

        headers = self._get_v3_headers(volcano_config, request_id, resource_id)
        headers.pop("Content-Type", None)
        log.info(f"Connecting to bidirectional WebSocket {host} with headers: {headers}")

        connect_kwargs = {
            "max_size": 10 * 1024 * 1024,
            "open_timeout": float(volcano_config.get("open_timeout", 30)),
        }
        sig = inspect.signature(websockets.connect)
        if "additional_headers" in sig.parameters:
            connect_kwargs["additional_headers"] = headers
        else:
            connect_kwargs["extra_headers"] = headers

        self.websocket = await websockets.connect(host, **connect_kwargs)
        self.connected = True

        log.info("WebSocket connected. Starting connection event handshakes.")
        # Start connection handshake
        await start_connection(self.websocket)
        
        # Wait for connection started ACK
        msg = await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.ConnectionStarted)
        log.info(f"Connection established successfully. Payload: {msg.payload.decode('utf-8', 'ignore')}")

    async def start_session(self, session_id: str, voice_character: str, speed: float = 1.0, volume: float = 1.0, instruct: Optional[str] = None, **kwargs) -> None:
        """
        Start a new dialogue session.
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket is not connected.")

        from models.volcano_online_voice_enums import character_options
        voice_type = character_options.get(voice_character)
        if not voice_type:
            if voice_character in character_options.values():
                voice_type = voice_character
            elif any(voice_character.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_"]):
                voice_type = voice_character
            else:
                voice_type = character_options.get("Vivi 2.0", "zh_female_vv_uranus_bigtts")

        # Map speed & volume to rate offsets
        speech_rate = kwargs.get("speech_rate")
        if speech_rate is None:
            speech_rate = int((speed - 1.0) * 100)
            speech_rate = max(-50, min(100, speech_rate))

        loudness_rate = kwargs.get("loudness_rate")
        if loudness_rate is None:
            loudness_rate = int((volume - 1.0) * 50)
            loudness_rate = max(-50, min(50, loudness_rate))

        audio_params = {
            "format": kwargs.get("audio_format", "mp3"),
            "sample_rate": kwargs.get("sample_rate", 24000),
            "speech_rate": speech_rate,
            "loudness_rate": loudness_rate,
            "enable_subtitle": True
        }

        session_config = {
            "audio_params": audio_params,
        }
        if any(voice_type.startswith(p) for p in ["vc_", "vd_", "vclone_", "vdesign_"]):
            session_config["speaker"] = "custom_speaker_id"
            session_config["custom_speaker_id"] = voice_type
        else:
            session_config["speaker"] = voice_type

        # Handle model override
        model = kwargs.get("model")
        if not model:
            if instruct:
                model = "seed-tts-2.0-expressive"
            else:
                model = "seed-tts-2.0-standard"
        session_config["model"] = model

        if instruct:
            session_config["context_texts"] = [instruct]

        payload = json.dumps(session_config).encode("utf-8")
        log.info(f"Starting session {session_id} with payload: {session_config}")
        
        await start_session(self.websocket, payload, session_id)
        
        # Wait for session started
        msg = await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.SessionStarted)
        log.info(f"Session {session_id} started successfully. Payload: {msg.payload.decode('utf-8', 'ignore')}")

    async def send_text(self, text: str, session_id: str) -> None:
        """
        Send text for speech synthesis under the given session.
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket is not connected.")

        # Construct ChatTTSText message
        msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
        msg.event = EventType.ChatTTSText
        msg.session_id = session_id
        
        payload_data = {
            "text": text
        }
        msg.payload = json.dumps(payload_data).encode("utf-8")
        
        log.info(f"Sending ChatTTSText message. Text: '{text[:20]}...'")
        await self.websocket.send(msg.marshal())

    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """
        Continuously read messages from the WebSocket connection.
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket is not connected.")

        while self.connected:
            try:
                msg = await receive_message(self.websocket)
                yield msg
            except websockets.exceptions.ConnectionClosed:
                log.info("WebSocket connection closed by server.")
                self.connected = False
                break
            except Exception as e:
                log.error(f"Error in receive loop: {e}")
                self.connected = False
                raise

    async def finish_session(self, session_id: str) -> None:
        """
        Finish current session.
        """
        if not self.connected or not self.websocket:
            return

        log.info(f"Finishing session {session_id}")
        await finish_session(self.websocket, session_id)
        # Wait for SessionFinished
        await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.SessionFinished)

    async def close(self) -> None:
        """
        Close the connection cleanly.
        """
        if not self.websocket:
            return

        try:
            if self.connected:
                log.info("Finishing connection")
                await finish_connection(self.websocket)
                await wait_for_event(self.websocket, MsgType.FullServerResponse, EventType.ConnectionFinished)
        except Exception as e:
            log.warning(f"Error during close handshake: {e}")
        finally:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            log.info("WebSocket connection closed.")
