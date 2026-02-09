import uuid
import asyncio
import logging
from typing import Callable, Optional, Dict, List, Set, Tuple, Awaitable

from backend.stream2sentence import generate_sentences_async
from backend.database_director import db, Character, MessageCreate, ConversationCreate
from backend.services.pipeline_types import (
    TTSSentence, AudioChunk, CharacterResponse, PipeQueues,
)
from backend.services.chat_service import ChatLLM
from backend.services.tts_service import TTS

logger = logging.getLogger(__name__)


########################################
##--     Conversation Commando      --##
########################################

class ConversationPipeline:
    """Orchestrates one conversation session.

    Owns conversation state, turn lifecycle, and the two internal
    background tasks (user-message consumer and sentence-to-audio worker).
    """

    def __init__(
            self,
            chat: ChatLLM,
            tts: TTS,
            queues: PipeQueues,
            on_event: Callable[[str, dict], Awaitable[None]],
            ):
        
        self.chat = chat
        self.tts = tts
        self.queues = queues
        self._on_event = on_event

        # Conversation state
        self.conversation_id: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.active_characters: List[Character] = []

        # Background tasks
        self._user_message_task: Optional[asyncio.Task] = None
        self._process_sentences_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start pipeline background tasks."""

        self._flush_queues()

        self._user_message_task = asyncio.create_task(self.get_user_message())
        self._process_sentences_task = asyncio.create_task(self.process_sentences())

        logger.info("[Pipeline] Background tasks started")

    async def stop(self):
        """Cancel pipeline background tasks."""

        for task in (self._user_message_task, self._process_sentences_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("[Pipeline] Background tasks stopped")

    async def start_conversation(self):
        """Begin a new conversation. Clears history, creates DB record."""

        self.conversation_history = []
        active_char_data = [{"id": c.id, "name": c.name} for c in self.active_characters]

        self.conversation_id = await db.create_conversation_background(ConversationCreate(active_characters=active_char_data))

        logger.info(f"[Pipeline] New conversation: {self.conversation_id}")

    def clear_conversation(self):
        """Reset conversation history."""

        self.conversation_history = []

    async def get_user_message(self):
        """Background task: pull user messages from stt_queue, start turns."""

        while True:
            try:
                user_message: str = await self.queues.stt_queue.get()

                if user_message and user_message.strip():
                    await self.start_turn(user_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Pipeline] Error processing user message: {e}")

    async def start_turn(self, user_message: str):
        """One full round: user message → character response chain."""

        self.conversation_history.append({"role": "user", "name": "Jay", "content": user_message})

        if self.conversation_id:
            db.create_message_background(MessageCreate(
                conversation_id=self.conversation_id,
                role="user",
                name="Jay",
                content=user_message,
            ))

        # Response chain loop
        responded_to: Set[Tuple[str, str]] = set()
        last_message = user_message
        last_speaker_id = "user"

        while True:
            character = self.route_response(
                last_message, last_speaker_id, responded_to
            )
            if character is None:
                break

            response = await self.generate_character_response(character)

            responded_to.add((character.id, last_speaker_id))
            last_message = response.text
            last_speaker_id = character.id

    def route_response(
            self,
            message: str,
            speaker_id: str,
            responded_to: Set[Tuple[str, str]],
            ) -> Optional[Character]:
        """Determine the next responding character, or None to end the chain."""

        exclude_id = None if speaker_id == "user" else speaker_id

        character = self.chat.find_first_mentioned_character(message, self.active_characters, exclude_id)

        if character is None:
            if speaker_id == "user" and self.active_characters:
                return sorted(self.active_characters, key=lambda c: c.name)[0]
            return None

        if speaker_id != "user":
            if (character.id, speaker_id) in responded_to:
                return None

        return character

    async def generate_character_response(self, character: Character) -> CharacterResponse:
        """Generate one character's full response within the current turn."""

        message_id = str(uuid.uuid4())

        await self.emit("text_stream_start",character_id=character.id,character_name=character.name,message_id=message_id)

        response = await self.stream_response(character, message_id)

        await self.emit("text_stream_stop",character_id=character.id,character_name=character.name,message_id=message_id,text=response)

        if response:
            wrapped = self.chat.wrap_character_tags(response, character.name)

            self.conversation_history.append({"role": "assistant","name": character.name,"content": wrapped})

            if self.conversation_id:
                db.create_message_background(MessageCreate(
                    conversation_id=self.conversation_id,
                    role="assistant",
                    name=character.name,
                    content=response,
                    character_id=character.id,
                ))

        return CharacterResponse(
            message_id=message_id,
            conversation_id=self.conversation_id or "",
            character_id=character.id,
            character_name=character.name,
            voice_id=character.voice,
            text=response,
        )

    async def stream_response(self, character: Character, message_id: str) -> str:
        """Stream LLM text, emit chunk events, extract sentences, queue for TTS. Returns the full accumulated response text."""

        messages = self.chat.build_messages_for_character(character, self.conversation_history)

        model_settings = self.chat.get_model_settings()

        self.chat.save_conversation_context(messages, character, model_settings)

        response = ""
        sentence_index = 0

        try:
            async def chunk_generator():
                """Wrap LLM stream: accumulate text and emit chunk events."""
                nonlocal response
                async for text in self.chat.create_completion_stream(
                    messages, model_settings
                ):
                    response += text
                    await self.emit("text_chunk",
                        text=text,
                        character_id=character.id,
                        character_name=character.name,
                        message_id=message_id,
                    )
                    yield text

            async for sentence in generate_sentences_async(
                chunk_generator(),
                minimum_first_fragment_length=4,
                minimum_sentence_length=25,
                tokenizer="nltk",
                quick_yield_single_sentence_fragment=True,
                sentence_fragment_delimiters=".?!;:,\n…)]}。-",
                full_sentence_delimiters=".?!\n…。",
            ):
                text = sentence.strip()
                if text:
                    await self.queues.sentence_queue.put(TTSSentence(
                        text=text,
                        index=sentence_index,
                        message_id=message_id,
                        character_id=character.id,
                        character_name=character.name,
                        voice_id=character.voice,
                    ))
                    logger.info(f"[LLM] {character.name} sentence {sentence_index}: {text[:50]}...")
                    sentence_index += 1

        except Exception as e:
            logger.error(f"[LLM] Error streaming for {character.name}: {e}")

        await self.queues.sentence_queue.put(None)
        logger.info(f"[LLM] {character.name} complete: {sentence_index} sentences")

        return response

    async def process_sentences(self):
        """Background task: pull sentences from queue, synthesize audio, queue chunks."""
        while True:
            try:
                sentence: TTSSentence = await asyncio.wait_for(self.queues.sentence_queue.get(), timeout=0.1)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if sentence is None:
                await self.queues.tts_queue.put(None)
                logger.info("[TTS] End sentinel passed through")
                continue

            logger.info(f"[TTS] Generating audio for sentence {sentence.index}")
            chunk_index = 0

            try:
                async for pcm_bytes in self.tts.generate_audio_for_sentence(
                    sentence.text, sentence.voice_id
                ):
                    await self.queues.tts_queue.put(AudioChunk(
                        audio_bytes=pcm_bytes,
                        sentence_index=sentence.index,
                        chunk_index=chunk_index,
                        message_id=sentence.message_id,
                        character_id=sentence.character_id,
                        character_name=sentence.character_name,
                    ))
                    chunk_index += 1

                logger.info(
                    f"[TTS] {sentence.character_name} #{sentence.index}: {chunk_index} chunks")
            except Exception as e:
                logger.error(f"[TTS] Error generating audio: {e}")

    async def emit(self, event_type: str, **data):
        """Emit a pipeline event to the transport layer."""
        await self._on_event(event_type, data)

    def _flush_queues(self):
        """Drain all pipeline queues."""
        for q in (self.queues.stt_queue,
                  self.queues.sentence_queue,
                  self.queues.tts_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break