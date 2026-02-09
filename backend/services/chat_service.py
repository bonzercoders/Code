import os
import re
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, AsyncGenerator
from openai import AsyncOpenAI

from backend.database_director import Character
from backend.services.pipeline_types import ModelSettings

logger = logging.getLogger(__name__)


class ChatLLM:
    """LLM helpers: streaming, message building, and routing utilities.

    Conversation state (history, conversation_id, active_characters)
    lives in ConversationPipeline — this class is stateless aside from
    the OpenAI client and model settings.
    """

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_settings: Optional[ModelSettings] = None

    # ──────────────────────────────────────────────
    #  LLM streaming
    # ──────────────────────────────────────────────

    async def create_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[str, None]:
        """Open an LLM completion stream and yield raw text chunks.

        Encapsulates all OpenRouter / OpenAI API details so the caller
        only sees plain strings.
        """
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
            },
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    # ──────────────────────────────────────────────
    #  Message building
    # ──────────────────────────────────────────────

    def build_messages_for_character(
        self,
        character: Character,
        conversation_history: List[Dict],
    ) -> List[Dict[str, str]]:
        """Build the message list for an OpenRouter API call."""
        messages = []

        if character.system_prompt:
            messages.append({"role": "system", "content": character.system_prompt})

        messages.extend(conversation_history)
        messages.append(self.character_instruction_message(character))

        return messages

    def character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""
        return {
            'role': 'system',
            'content': (
                f'Based on the conversation history above provide the next reply '
                f'as {character.name}. Your response should include only '
                f'{character.name}\'s reply. Do not respond for/as anyone else.'
            ),
        }

    def wrap_character_tags(self, text: str, character_name: str) -> str:
        """Wrap response text with character name XML tags for conversation history."""
        return f"<{character_name}>{text}</{character_name}>"

    # ──────────────────────────────────────────────
    #  Routing helper
    # ──────────────────────────────────────────────

    def find_first_mentioned_character(
        self,
        message: str,
        active_characters: List[Character],
        exclude_id: Optional[str] = None,
    ) -> Optional[Character]:
        """Find the first character mentioned in a message.

        Skips the character whose id matches exclude_id (the current speaker)
        to prevent self-mention loops. Returns None if no mention found.
        """
        candidates = [c for c in active_characters if c.id != exclude_id]

        mentions = []
        for character in candidates:
            name_parts = character.name.lower().split()
            for part in name_parts:
                pattern = r'\b' + re.escape(part) + r'\b'
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    mentions.append((match.start(), character))
                    break  # one match per character is enough

        if not mentions:
            return None

        mentions.sort(key=lambda x: x[0])
        return mentions[0][1]

    # ──────────────────────────────────────────────
    #  Config
    # ──────────────────────────────────────────────

    def get_model_settings(self) -> ModelSettings:
        """Get current model settings, with sensible defaults."""
        if self.model_settings is None:
            return ModelSettings(
                model="meta-llama/llama-3.1-8b-instruct",
                temperature=0.7,
                top_p=0.9,
                min_p=0.0,
                top_k=40,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )
        return self.model_settings

    def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests."""
        self.model_settings = model_settings

    # ──────────────────────────────────────────────
    #  Debug
    # ──────────────────────────────────────────────

    def save_conversation_context(
        self,
        messages: List[Dict[str, str]],
        character: Character,
        model_settings: ModelSettings,
    ) -> str:
        """Save the full conversation context to a JSON file for debugging.

        Returns the filepath of the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"conversation_context_{timestamp}.json"
        filepath = os.path.join("backend", filename)

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
                    "content": msg.get("content", ""),
                }
                for msg in messages
            ],
            "message_count": len(messages),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[DEBUG] Saved conversation context to: {filepath}")
        return filepath
