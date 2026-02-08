import asyncio
from dataclasses import dataclass
from typing import Callable, Optional
from collections.abc import Awaitable

Callback = Callable[..., Optional[Awaitable[None]]]

@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis"""
    text: str
    index: int
    message_id: str
    character_id: str
    character_name: str
    voice_id: str
    is_final: bool = False

@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming"""
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    message_id: str
    character_id: str
    character_name: str
    is_final: bool = False

@dataclass
class ModelSettings:
    model: str
    temperature: float
    top_p: float
    min_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

class PipeQueues:
    """Queue management for pipeline stages"""
    def __init__(self):
        self.stt_queue: asyncio.Queue = asyncio.Queue()
        self.sentence_queue: asyncio.Queue = asyncio.Queue()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
