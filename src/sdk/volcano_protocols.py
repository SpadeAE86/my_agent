import io
import json
import logging
import struct
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List

import websockets

logger = logging.getLogger(__name__)


class MsgType(IntEnum):
    Invalid = 0
    FullClientRequest = 0b1
    AudioOnlyClient = 0b10
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111


class MsgTypeFlagBits(IntEnum):
    NoSeq = 0
    PositiveSeq = 0b1
    LastNoSeq = 0b10
    NegativeSeq = 0b11
    WithEvent = 0b100


class VersionBits(IntEnum):
    Version1 = 1


class HeaderSizeBits(IntEnum):
    HeaderSize4 = 1


class SerializationBits(IntEnum):
    Raw = 0
    JSON = 0b1


class CompressionBits(IntEnum):
    None_ = 0


class EventType(IntEnum):
    None_ = 0
    StartConnection = 1
    FinishConnection = 2
    ConnectionStarted = 50
    ConnectionFailed = 51
    ConnectionFinished = 52
    StartSession = 100
    CancelSession = 101
    FinishSession = 102
    TaskRequest = 200
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352


@dataclass
class Message:
    version: VersionBits = VersionBits.Version1
    header_size: HeaderSizeBits = HeaderSizeBits.HeaderSize4
    type: MsgType = MsgType.Invalid
    flag: MsgTypeFlagBits = MsgTypeFlagBits.NoSeq
    serialization: SerializationBits = SerializationBits.JSON
    compression: CompressionBits = CompressionBits.None_
    event: EventType = EventType.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0
    payload: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        if len(data) < 3:
            raise ValueError(f"Data too short: expected at least 3 bytes, got {len(data)}")
        type_and_flag = data[1]
        msg_type = MsgType(type_and_flag >> 4)
        flag = MsgTypeFlagBits(type_and_flag & 0b00001111)
        msg = cls(type=msg_type, flag=flag)
        msg.unmarshal(data)
        return msg

    def marshal(self) -> bytes:
        buffer = io.BytesIO()
        header = [
            (self.version << 4) | self.header_size,
            (self.type << 4) | self.flag,
            (self.serialization << 4) | self.compression,
        ]
        header_size = 4 * self.header_size
        if padding := header_size - len(header):
            header.extend([0] * padding)
        buffer.write(bytes(header))
        writers = self._get_writers()
        for writer in writers:
            writer(buffer)
        return buffer.getvalue()

    def unmarshal(self, data: bytes) -> None:
        buffer = io.BytesIO(data)
        version_and_header_size = buffer.read(1)[0]
        self.version = VersionBits(version_and_header_size >> 4)
        self.header_size = HeaderSizeBits(version_and_header_size & 0b00001111)
        buffer.read(1)
        serialization_compression = buffer.read(1)[0]
        self.serialization = SerializationBits(serialization_compression >> 4)
        self.compression = CompressionBits(serialization_compression & 0b00001111)
        header_size = 4 * self.header_size
        read_size = 3
        if padding_size := header_size - read_size:
            buffer.read(padding_size)
        readers = self._get_readers()
        for reader in readers:
            reader(buffer)
        remaining = buffer.read()
        if remaining:
            raise ValueError(f"Unexpected data after message: {remaining}")

    def _get_writers(self) -> List[Callable[[io.BytesIO], None]]:
        writers = []
        if self.flag == MsgTypeFlagBits.WithEvent:
            writers.extend([self._write_event, self._write_session_id])
        if self.type in [MsgType.FullClientRequest, MsgType.FullServerResponse, MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                writers.append(self._write_sequence)
        elif self.type == MsgType.Error:
            writers.append(self._write_error_code)
        else:
            raise ValueError(f"Unsupported message type: {self.type}")
        writers.append(self._write_payload)
        return writers

    def _get_readers(self) -> List[Callable[[io.BytesIO], None]]:
        readers = []
        if self.type in [MsgType.FullClientRequest, MsgType.FullServerResponse, MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                readers.append(self._read_sequence)
        elif self.type == MsgType.Error:
            readers.append(self._read_error_code)
        else:
            raise ValueError(f"Unsupported message type: {self.type}")
        if self.flag == MsgTypeFlagBits.WithEvent:
            readers.extend([self._read_event, self._read_session_id, self._read_connect_id])
        readers.append(self._read_payload)
        return readers

    def _write_event(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">i", self.event))

    def _write_session_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.StartConnection, EventType.FinishConnection, EventType.ConnectionStarted, EventType.ConnectionFailed]:
            return
        session_id_bytes = self.session_id.encode("utf-8")
        size = len(session_id_bytes)
        buffer.write(struct.pack(">I", size))
        if size > 0:
            buffer.write(session_id_bytes)

    def _write_sequence(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">i", self.sequence))

    def _write_error_code(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">I", self.error_code))

    def _write_payload(self, buffer: io.BytesIO) -> None:
        size = len(self.payload)
        buffer.write(struct.pack(">I", size))
        buffer.write(self.payload)

    def _read_event(self, buffer: io.BytesIO) -> None:
        event_bytes = buffer.read(4)
        if event_bytes:
            self.event = EventType(struct.unpack(">i", event_bytes)[0])

    def _read_session_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.StartConnection, EventType.FinishConnection, EventType.ConnectionStarted, EventType.ConnectionFailed, EventType.ConnectionFinished]:
            return
        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                session_id_bytes = buffer.read(size)
                if len(session_id_bytes) == size:
                    self.session_id = session_id_bytes.decode("utf-8")

    def _read_connect_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.ConnectionStarted, EventType.ConnectionFailed, EventType.ConnectionFinished]:
            size_bytes = buffer.read(4)
            if size_bytes:
                size = struct.unpack(">I", size_bytes)[0]
                if size > 0:
                    self.connect_id = buffer.read(size).decode("utf-8")

    def _read_sequence(self, buffer: io.BytesIO) -> None:
        sequence_bytes = buffer.read(4)
        if sequence_bytes:
            self.sequence = struct.unpack(">i", sequence_bytes)[0]

    def _read_error_code(self, buffer: io.BytesIO) -> None:
        error_code_bytes = buffer.read(4)
        if error_code_bytes:
            self.error_code = struct.unpack(">I", error_code_bytes)[0]

    def _read_payload(self, buffer: io.BytesIO) -> None:
        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                self.payload = buffer.read(size)


async def receive_message(websocket: websockets.WebSocketClientProtocol) -> Message:
    data = await websocket.recv()
    if isinstance(data, str):
        raise ValueError(f"Unexpected text message: {data}")
    if isinstance(data, bytes):
        msg = Message.from_bytes(data)
        logger.info(f"Received: {msg}")
        return msg
    raise ValueError(f"Unexpected message type: {type(data)}")


async def full_client_request(websocket: websockets.WebSocketClientProtocol, payload: bytes) -> None:
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.NoSeq)
    msg.payload = payload
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())
