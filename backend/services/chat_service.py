import os
import re
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional, Dict, List, Awaitable, AsyncGenerator
from openai import AsyncOpenAI

from backend.stream2sentence import generate_sentences_async
from backend.database_director import db, Character, MessageCreate, ConversationCreate
from backend.services.pipeline_types import TTSSentence, ModelSettings, PipeQueues

logger = logging.getLogger(__name__)

class ChatLLM:
    """"""

    def __init__(self, queues: PipeQueues, api_key: str):
        """Initialize conversation"""
        self.conversation_history: List[Dict] = []
        self.conversation_id: Optional[str] = None
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_settings: Optional[ModelSettings] = None
        self.active_characters: List[Character] = []

    async def start_new_conversation(self):
        """Start a new chat session and persist to database"""
        self.conversation_history = []
        # Create conversation in background, get ID immediately
        active_char_data = [{"id": c.id, "name": c.name} for c in self.active_characters]
        self.conversation_id = await db.create_conversation_background(
            ConversationCreate(active_characters=active_char_data)
        )

    async def get_active_characters(self) -> List[Character]:
        """Get active characters from database"""
        return await db.get_active_characters()

    def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests"""

        self.model_settings = model_settings

    def clear_conversation_history(self):
        """Clear the conversation history"""

        self.conversation_history = []

    def wrap_character_tags(self, text: str, character_name: str) -> str:
        """Wrap response text with character name XML tags for conversation history."""

        return f"<{character_name}>{text}</{character_name}>"

    def character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""

        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else.'
        }

    def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""

        mentioned_characters = []
        processed_characters = set()

        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        name_mentions.sort(key=lambda x: x['position'])

        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters

    def get_model_settings(self) -> ModelSettings:
        """Get current model settings for the LLM request"""
        if self.model_settings is None:
            # Return default settings if not set
            return ModelSettings(
                model="meta-llama/llama-3.1-8b-instruct",
                temperature=0.7,
                top_p=0.9,
                min_p=0.0,
                top_k=40,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0
            )
        return self.model_settings

    def build_messages_for_character(self, character: Character) -> List[Dict[str, str]]:
        """Build the message list for OpenRouter API call."""

        messages = []

        # Character's system prompt
        if character.system_prompt:
            messages.append({"role": "system", "content": character.system_prompt})

        # Conversation history
        messages.extend(self.conversation_history)

        # Instruction for this character
        messages.append(self.character_instruction_message(character))

        return messages

    def save_conversation_context(self, messages: List[Dict[str, str]], character: Character, model_settings: ModelSettings) -> str:
        """Save the full conversation context to a JSON file for debugging.

        Returns the filepath of the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"conversation_context_{timestamp}.json"
        filepath = os.path.join("backend", filename)

        # Build the full request payload for inspection
        context_data = {
            "timestamp": datetime.now().isoformat(),
            "character": {
                "id": character.id,
                "name": character.name,
            },
            "model_settings": {
                "model": model_settings.model,
                "temperature": model_settings.temperature,
                "top_p": model_settings.top_p,
                "min_p": model_settings.min_p,
                "top_k": model_settings.top_k,
                "frequency_penalty": model_settings.frequency_penalty,
                "presence_penalty": model_settings.presence_penalty,
                "repetition_penalty": model_settings.repetition_penalty,
            },
            "messages": [
                {
                    "role": msg.get("role", ""),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", "")
                }
                for msg in messages
            ],
            "message_count": len(messages),
            "conversation_history_count": len(self.conversation_history),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[DEBUG] Saved conversation context to: {filepath}")
        return filepath

    async def generate_character_response(self, user_message: str, sentence_queue: asyncio.Queue,
                                    on_text_chunk: Optional[Callable[[str, Character, str], Awaitable[None]]] = None,
                                    on_text_stream_start: Optional[Callable[[Character, str], Awaitable[None]]] = None,
                                    on_text_stream_stop: Optional[Callable[[Character, str, str], Awaitable[None]]] = None) -> None:

        if not user_message or not user_message.strip():
            return

        self.conversation_history.append({"role": "user", "name": "Jay", "content": user_message})

        # Save user message to database in background (non-blocking)
        if self.conversation_id:
            db.create_message_background(MessageCreate(
                conversation_id=self.conversation_id,
                role="user",
                name="Jay",
                content=user_message
            ))

        responding_characters = self.parse_character_mentions(message=user_message, active_characters=self.active_characters)

        model_settings = self.get_model_settings()

        for character in responding_characters:
            message_id = str(uuid.uuid4())
            messages = self.build_messages_for_character(character)

            if on_text_stream_start:
                await on_text_stream_start(character, message_id)

            full_response = await self.stream_character_response(messages=messages, character=character, message_id=message_id,
                                                                 model_settings=model_settings, sentence_queue=sentence_queue,
                                                                 on_text_chunk=on_text_chunk)

            if on_text_stream_stop:
                await on_text_stream_stop(character, message_id, full_response)

            if full_response:
                response_wrapped = self.wrap_character_tags(full_response, character.name)

                self.conversation_history.append({"role": "assistant", "name": character.name, "content": response_wrapped})

                # Save assistant message to database in background (non-blocking)
                if self.conversation_id:
                    db.create_message_background(MessageCreate(
                        conversation_id=self.conversation_id,
                        role="assistant",
                        name=character.name,
                        content=full_response,
                        character_id=character.id
                    ))

    async def stream_character_response(self, messages: List[Dict[str, str]], character: Character, message_id: str,
                                        model_settings: ModelSettings, sentence_queue: asyncio.Queue,
                                        on_text_chunk: Optional[Callable[[str, Character, str], Awaitable[None]]] = None) -> str:
        """Stream LLM response for a character, extract sentences, queue for TTS."""

        sentence_index = 0
        full_response = ""

        # Save conversation context to JSON file for debugging
        self.save_conversation_context(messages, character, model_settings)

        try:
            stream = await self.client.chat.completions.create(
                model=model_settings.model,
                messages=messages,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
                stream=True,
                extra_body={
                    "top_k": model_settings.top_k,
                    "min_p": model_settings.min_p,
                    "repetition_penalty": model_settings.repetition_penalty,
                }
            )

            async def chunk_generator() -> AsyncGenerator[str, None]:
                nonlocal full_response
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            if on_text_chunk:
                                await on_text_chunk(content, character, message_id)
                            yield content

            async for sentence in generate_sentences_async(
                chunk_generator(),
                minimum_first_fragment_length=4,
                minimum_sentence_length=25,
                tokenizer="nltk",
                quick_yield_single_sentence_fragment=True,
                sentence_fragment_delimiters=".?!;:,\n…)]}。-",
                full_sentence_delimiters=".?!\n…。",
            ):
                sentence_text = sentence.strip()
                if sentence_text:
                    await sentence_queue.put(TTSSentence(
                        text=sentence_text,
                        index=sentence_index,
                        message_id=message_id,
                        character_id=character.id,
                        character_name=character.name,
                        voice_id=character.voice,
                        is_final=False,
                    ))
                    logger.info(f"[LLM] {character.name} sentence {sentence_index}: {sentence_text[:50]}...")
                    sentence_index += 1

        except Exception as e:
            logger.error(f"[LLM] Error streaming for {character.name}: {e}")

        await sentence_queue.put(TTSSentence(
            text="",
            index=sentence_index,
            message_id=message_id,
            character_id=character.id,
            character_name=character.name,
            voice_id=character.voice,
            is_final=True,
        ))
        logger.info(f"[LLM] {character.name} complete: {sentence_index} sentences")

        return full_response
