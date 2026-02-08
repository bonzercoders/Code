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
from backend.database_director import (db, Character, CharacterCreate, CharacterUpdate, Voice, VoiceCreate, VoiceUpdate, Conversation, ConversationCreate, ConversationUpdate, Message as ConversationMessage, MessageCreate)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket and Routes Messages"""

    def __init__(self):
        # Pipeline queues - shared between all components
        self.queues = PipeQueues()

        # WebSocket connection
        self.websocket: Optional[WebSocket] = None

        # Pipeline components (initialized in initialize())
        self.stt: Optional[STT] = None
        self.chat: Optional[ChatLLM] = None
        self.tts: Optional[TTS] = None

        # Background tasks
        self.user_message_task: Optional[asyncio.Task] = None
        self.audio_streamer_task: Optional[asyncio.Task] = None

        self.user_name = "Jay"

    async def initialize(self): # <============================= boom
        """Initialize all pipeline components at startup"""

        api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.stt = STT(on_transcription_update=self.on_transcription_update,
                                     on_transcription_stabilized=self.on_transcription_stabilized,
                                     on_transcription_finished=self.on_transcription_finished)

        self.stt.set_event_loop(asyncio.get_event_loop())

        self.chat = ChatLLM(queues=self.queues, api_key=api_key)
        self.chat.active_characters = await self.chat.get_active_characters()

        self.tts = TTS(queues=self.queues)
        await self.tts.initialize()

        logger.info(f"Initialized with {len(self.chat.active_characters)} active characters")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection and start pipeline"""
        await websocket.accept()
        self.websocket = websocket

        # Start a new conversation for this session
        if self.chat:
            await self.chat.start_new_conversation()

        await self.start_pipeline()

        logger.info("WebSocket connected, pipeline started")

    async def disconnect(self):
        if self.stt:
            self.stt.stop_listening()

        await self.stop_pipeline()
        self.websocket = None

    async def shutdown(self):
        await self.disconnect()

    async def handle_text_message(self, message: str):
        """Handle incoming text messages from WebSocket client"""
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
                if self.chat:
                    self.chat.clear_conversation_history()
                    await self.chat.start_new_conversation()  # Start fresh conversation
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
        """Feed audio for transcription"""
        if self.stt:
            self.stt.feed_audio(audio_bytes)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.stt_queue.put(user_message)

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))

    async def on_transcription_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})

    async def on_transcription_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})

    async def on_transcription_finished(self, user_message: str):
        await self.queues.stt_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_final", "text": user_message})

    async def on_llm_text_chunk(self, text: str, character: Character, message_id: str):
        await self.send_text_to_client({
            "type": "text_chunk",
            "data": {
                "text": text,
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
                "is_final": False,
            },
        })

    async def on_text_stream_start(self, character: Character, message_id: str):
        await self.send_text_to_client({
            "type": "text_stream_start",
            "data": {
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
            },
        })

    async def on_text_stream_stop(self, character: Character, message_id: str, full_text: str):
        await self.send_text_to_client({
            "type": "text_chunk",
            "data": {
                "text": "",
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
                "is_final": True,
            },
        })
        await self.send_text_to_client({
            "type": "text_stream_stop",
            "data": {
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
                "text": full_text,
            },
        })

    async def on_audio_stream_start(self, chunk: AudioChunk):
        sample_rate = self.tts.sample_rate if self.tts else 24000
        await self.send_text_to_client({
            "type": "audio_stream_start",
            "data": {
                "character_id": chunk.character_id,
                "character_name": chunk.character_name,
                "message_id": chunk.message_id,
                "sample_rate": sample_rate,
            },
        })

    async def on_audio_stream_stop(self, chunk: AudioChunk):
        await self.send_text_to_client({
            "type": "audio_stream_stop",
            "data": {
                "character_id": chunk.character_id,
                "character_name": chunk.character_name,
                "message_id": chunk.message_id,
            },
        })

    async def refresh_active_characters(self):
        """Refresh active characters from database (call when characters change)"""
        if self.chat:
            self.chat.active_characters = await self.chat.get_active_characters()
            logger.info(f"Refreshed to {len(self.chat.active_characters)} active characters")

    async def start_pipeline(self):
        """Start the voice transcription → LLM → TTS pipeline as background tasks."""
        self._clear_queue(self.queues.stt_queue)
        self._clear_queue(self.queues.sentence_queue)
        self._clear_queue(self.queues.tts_queue)

        if self.tts and not self.tts.is_running:
            await self.tts.start()

        if self.user_message_task is None or self.user_message_task.done():
            self.user_message_task = asyncio.create_task(self.get_user_message())

        if self.audio_streamer_task is None or self.audio_streamer_task.done():
            self.audio_streamer_task = asyncio.create_task(self.stream_audio_to_client())

    async def stop_pipeline(self):
        """Stop background pipeline tasks and TTS worker."""
        if self.user_message_task and not self.user_message_task.done():
            self.user_message_task.cancel()
            try:
                await self.user_message_task
            except asyncio.CancelledError:
                pass

        if self.audio_streamer_task and not self.audio_streamer_task.done():
            self.audio_streamer_task.cancel()
            try:
                await self.audio_streamer_task
            except asyncio.CancelledError:
                pass

        if self.tts and self.tts.is_running:
            await self.tts.stop()

    def _clear_queue(self, queue_obj: asyncio.Queue):
        while True:
            try:
                queue_obj.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stream_audio_to_client(self):
        """Background task: stream synthesized audio chunks to the client."""
        current_message_id: Optional[str] = None

        while True:
            try:
                chunk: AudioChunk = await asyncio.wait_for(self.queues.tts_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if chunk.is_final:
                await self.on_audio_stream_stop(chunk)
                current_message_id = None
                continue

            if current_message_id != chunk.message_id:
                await self.on_audio_stream_start(chunk)
                current_message_id = chunk.message_id

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

    async def get_user_message(self):
        """Background task: get user message from stt queue and process."""
        while True:
            try:
                user_message: str = await self.queues.stt_queue.get()

                if user_message and user_message.strip():
                    await self.chat.generate_character_response(
                        user_message=user_message,
                        sentence_queue=self.queues.sentence_queue,
                        on_text_chunk=self.on_llm_text_chunk,
                        on_text_stream_start=self.on_text_stream_start,
                        on_text_stream_stop=self.on_text_stream_stop,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing user message: {e}")

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
