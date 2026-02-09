import os
import json
import asyncio
import logging
import uvicorn
from typing import Optional
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

from backend.services import STT, ChatLLM, TTS, PipeQueues, ModelSettings, AudioChunk
from backend.services.conversation_pipeline import ConversationPipeline
from backend.database_director import (db, Character, CharacterCreate, CharacterUpdate, Voice, VoiceCreate, VoiceUpdate, Conversation, ConversationCreate, ConversationUpdate, Message as ConversationMessage, MessageCreate)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connection and routes messages.

    Acts as the transport layer: delivers pipeline events and audio
    chunks to the browser over WebSocket. All conversation orchestration
    is handled by ConversationPipeline.
    """

    def __init__(self):
        self.queues = PipeQueues()
        self.websocket: Optional[WebSocket] = None

        # Pipeline components (initialized in initialize())
        self.stt: Optional[STT] = None
        self.chat: Optional[ChatLLM] = None
        self.tts: Optional[TTS] = None
        self.pipeline: Optional[ConversationPipeline] = None

        # Audio streamer task (transport-level)
        self._audio_streamer_task: Optional[asyncio.Task] = None

        self.user_name = "Jay"

    async def initialize(self):
        """Initialize all pipeline components at startup."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.stt = STT(
            on_transcription_update=self.on_transcription_update,
            on_transcription_stabilized=self.on_transcription_stabilized,
            on_transcription_finished=self.on_transcription_finished,
        )
        self.stt.set_event_loop(asyncio.get_event_loop())

        self.chat = ChatLLM(api_key=api_key)

        self.tts = TTS()
        await self.tts.initialize()

        self.pipeline = ConversationPipeline(
            chat=self.chat,
            tts=self.tts,
            queues=self.queues,
            on_event=self.on_pipeline_event,
        )
        self.pipeline.active_characters = await db.get_active_characters()

        logger.info(f"Initialized with {len(self.pipeline.active_characters)} active characters")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection and start pipeline."""
        await websocket.accept()
        self.websocket = websocket

        if self.pipeline:
            await self.pipeline.start_conversation()

        await self.start_pipeline()

        logger.info("WebSocket connected, pipeline started")

    async def disconnect(self):
        if self.stt:
            self.stt.stop_listening()

        await self.stop_pipeline()
        self.websocket = None

    async def shutdown(self):
        await self.disconnect()

    async def start_pipeline(self):
        """Start conversation pipeline + audio streamer."""
        if self.pipeline:
            await self.pipeline.start()

        if self._audio_streamer_task is None or self._audio_streamer_task.done():
            self._audio_streamer_task = asyncio.create_task(self.stream_audio_to_client())

    async def stop_pipeline(self):
        """Stop conversation pipeline + audio streamer."""
        if self.pipeline:
            await self.pipeline.stop()

        if self._audio_streamer_task and not self._audio_streamer_task.done():
            self._audio_streamer_task.cancel()
            try:
                await self._audio_streamer_task
            except asyncio.CancelledError:
                pass

    async def stream_audio_to_client(self):
        """Background task: deliver audio chunks over WebSocket."""
        current_message_id: Optional[str] = None
        last_chunk: Optional[AudioChunk] = None

        while True:
            try:
                chunk: AudioChunk = await asyncio.wait_for(
                    self.queues.tts_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # End-of-response sentinel
            if chunk is None:
                if last_chunk:
                    sample_rate = self.tts.sample_rate if self.tts else 24000
                    await self.on_pipeline_event("audio_stream_stop", {
                        "character_id": last_chunk.character_id,
                        "character_name": last_chunk.character_name,
                        "message_id": last_chunk.message_id,
                        "sample_rate": sample_rate,
                    })
                current_message_id = None
                last_chunk = None
                continue

            # New character/message starting
            if current_message_id != chunk.message_id:
                sample_rate = self.tts.sample_rate if self.tts else 24000
                await self.on_pipeline_event("audio_stream_start", {
                    "character_id": chunk.character_id,
                    "character_name": chunk.character_name,
                    "message_id": chunk.message_id,
                    "sample_rate": sample_rate,
                })
                current_message_id = chunk.message_id

            last_chunk = chunk

            # Send metadata + raw audio bytes
            await self.send_text_to_client({
                "type": "audio_chunk",
                "data": {
                    "character_id": chunk.character_id,
                    "character_name": chunk.character_name,
                    "message_id": chunk.message_id,
                    "sentence_index": chunk.sentence_index,
                    "chunk_index": chunk.chunk_index,
                },
            })

            if self.websocket:
                await self.websocket.send_bytes(chunk.audio_bytes)

    async def on_pipeline_event(self, event_type: str, data: dict):
        """Forward pipeline events to WebSocket client."""
        await self.send_text_to_client({"type": event_type, "data": data})

    # ──────────────────────────────────────────────
    #  STT callbacks
    # ──────────────────────────────────────────────

    async def on_transcription_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})

    async def on_transcription_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})

    async def on_transcription_finished(self, user_message: str):
        await self.queues.stt_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_final", "text": user_message})

    # ──────────────────────────────────────────────
    #  Message handling
    # ──────────────────────────────────────────────

    async def handle_text_message(self, message: str):
        """Handle incoming text messages from WebSocket client."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            payload = data.get("data", {})

            if message_type == "ping":
                await self.send_text_to_client({"type": "pong"})

            elif message_type == "user_message":
                user_message = payload.get("text", "")
                await self.handle_user_message(user_message)

            elif message_type == "start_listening":
                if self.stt:
                    self.stt.start_listening()

            elif message_type == "stop_listening":
                if self.stt:
                    self.stt.stop_listening()

            elif message_type == "model_settings":
                settings_data = payload
                model_settings = ModelSettings(
                    model=settings_data.get("model", "meta-llama/llama-3.1-8b-instruct"),
                    temperature=float(settings_data.get("temperature", 0.7)),
                    top_p=float(settings_data.get("top_p", 0.9)),
                    min_p=float(settings_data.get("min_p", 0.0)),
                    top_k=int(settings_data.get("top_k", 40)),
                    frequency_penalty=float(settings_data.get("frequency_penalty", 0.0)),
                    presence_penalty=float(settings_data.get("presence_penalty", 0.0)),
                    repetition_penalty=float(settings_data.get("repetition_penalty", 1.0))
                )
                if self.chat:
                    self.chat.set_model_settings(model_settings)
                logger.info(f"Model settings updated: {model_settings.model}")

            elif message_type in ("refresh_characters", "refresh_active_characters"):
                await self.refresh_active_characters()

            elif message_type == "clear_history":
                if self.pipeline:
                    self.pipeline.clear_conversation()
                    await self.pipeline.start_conversation()
                await self.send_text_to_client({"type": "history_cleared"})

            elif message_type == "interrupt":
                await self.handle_interrupt()

            # ==========================================
            # Database Operations - Characters
            # ==========================================
            elif message_type == "get_characters":
                characters = await db.get_all_characters()
                await self.send_text_to_client({
                    "type": "characters_data",
                    "data": [c.model_dump() for c in characters]
                })

            elif message_type == "get_character":
                character_id = payload.get("id")
                try:
                    character = await db.get_character(character_id)
                    await self.send_text_to_client({
                        "type": "character_data",
                        "data": character.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "create_character":
                try:
                    character = await db.create_character(CharacterCreate(**payload))
                    await self.send_text_to_client({
                        "type": "character_created",
                        "data": character.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "update_character":
                character_id = payload.pop("id", None)
                if not character_id:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": "Character ID required"
                    })
                else:
                    try:
                        character = await db.update_character(character_id, CharacterUpdate(**payload))
                        await self.send_text_to_client({
                            "type": "character_updated",
                            "data": character.model_dump()
                        })
                        # Refresh active characters if is_active changed
                        if "is_active" in payload:
                            await self.refresh_active_characters()
                    except HTTPException as e:
                        await self.send_text_to_client({
                            "type": "db_error",
                            "error": e.detail
                        })

            elif message_type == "delete_character":
                character_id = payload.get("id")
                try:
                    await db.delete_character(character_id)
                    await self.send_text_to_client({
                        "type": "character_deleted",
                        "data": {"id": character_id}
                    })
                    await self.refresh_active_characters()
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            # ==========================================
            # Database Operations - Voices
            # ==========================================
            elif message_type == "get_voices":
                voices = await db.get_all_voices()
                await self.send_text_to_client({
                    "type": "voices_data",
                    "data": [v.model_dump() for v in voices]
                })

            elif message_type == "get_voice":
                voice_name = payload.get("voice")
                try:
                    voice = await db.get_voice(voice_name)
                    await self.send_text_to_client({
                        "type": "voice_data",
                        "data": voice.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "create_voice":
                try:
                    voice = await db.create_voice(VoiceCreate(**payload))
                    await self.send_text_to_client({
                        "type": "voice_created",
                        "data": voice.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "update_voice":
                voice_name = payload.pop("voice", None)
                if not voice_name:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": "Voice name required"
                    })
                else:
                    try:
                        voice = await db.update_voice(voice_name, VoiceUpdate(**payload))
                        await self.send_text_to_client({
                            "type": "voice_updated",
                            "data": voice.model_dump()
                        })
                        await self.refresh_active_characters()
                    except HTTPException as e:
                        await self.send_text_to_client({
                            "type": "db_error",
                            "error": e.detail
                        })

            elif message_type == "delete_voice":
                voice_name = payload.get("voice")
                try:
                    await db.delete_voice(voice_name)
                    await self.send_text_to_client({
                        "type": "voice_deleted",
                        "data": {"voice": voice_name}
                    })
                    await self.refresh_active_characters()
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            # ==========================================
            # Database Operations - Conversations
            # ==========================================
            elif message_type == "get_conversations":
                limit = payload.get("limit")
                offset = payload.get("offset", 0)
                conversations = await db.get_all_conversations(limit=limit, offset=offset)
                await self.send_text_to_client({
                    "type": "conversations_data",
                    "data": [c.model_dump() for c in conversations]
                })

            elif message_type == "get_conversation":
                conversation_id = payload.get("conversation_id")
                try:
                    conversation = await db.get_conversation(conversation_id)
                    await self.send_text_to_client({
                        "type": "conversation_data",
                        "data": conversation.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "create_conversation":
                try:
                    conversation = await db.create_conversation(ConversationCreate(**payload))
                    await self.send_text_to_client({
                        "type": "conversation_created",
                        "data": conversation.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "update_conversation":
                conversation_id = payload.pop("conversation_id", None)
                if not conversation_id:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": "Conversation ID required"
                    })
                else:
                    try:
                        conversation = await db.update_conversation(conversation_id, ConversationUpdate(**payload))
                        await self.send_text_to_client({
                            "type": "conversation_updated",
                            "data": conversation.model_dump()
                        })
                    except HTTPException as e:
                        await self.send_text_to_client({
                            "type": "db_error",
                            "error": e.detail
                        })

            elif message_type == "delete_conversation":
                conversation_id = payload.get("conversation_id")
                try:
                    await db.delete_conversation(conversation_id)
                    await self.send_text_to_client({
                        "type": "conversation_deleted",
                        "data": {"conversation_id": conversation_id}
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            # ==========================================
            # Database Operations - Messages
            # ==========================================
            elif message_type == "get_messages":
                conversation_id = payload.get("conversation_id")
                limit = payload.get("limit")
                offset = payload.get("offset", 0)
                try:
                    messages = await db.get_messages(conversation_id, limit=limit, offset=offset)
                    await self.send_text_to_client({
                        "type": "messages_data",
                        "data": [m.model_dump() for m in messages]
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "create_message":
                try:
                    message = await db.create_message(MessageCreate(**payload))
                    await self.send_text_to_client({
                        "type": "message_created",
                        "data": message.model_dump()
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

            elif message_type == "delete_message":
                message_id = payload.get("message_id")
                try:
                    await db.delete_message(message_id)
                    await self.send_text_to_client({
                        "type": "message_deleted",
                        "data": {"message_id": message_id}
                    })
                except HTTPException as e:
                    await self.send_text_to_client({
                        "type": "db_error",
                        "error": e.detail
                    })

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_bytes: bytes):
        """Feed audio for transcription."""
        if self.stt:
            self.stt.feed_audio(audio_bytes)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message."""
        await self.queues.stt_queue.put(user_message)

    async def handle_interrupt(self):
        """Interrupt current generation. (To be implemented.)"""
        pass

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client."""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))

    async def refresh_active_characters(self):
        """Refresh active characters from database."""
        if self.pipeline:
            self.pipeline.active_characters = await db.get_active_characters()
            logger.info(f"Refreshed to {len(self.pipeline.active_characters)} active characters")

########################################
##--           FastAPI App          --##
########################################

ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up services...")
    await db.init_database()
    await ws_manager.initialize()
    print("All services initialized!")
    yield
    print("Shutting down services...")
    await ws_manager.shutdown()
    print("All services shut down!")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
##--       WebSocket Endpoint       --##
########################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                await ws_manager.handle_text_message(message["text"])

            elif "bytes" in message:
                await ws_manager.handle_audio_message(message["bytes"])

    except WebSocketDisconnect:
        await ws_manager.disconnect()

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect()

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)