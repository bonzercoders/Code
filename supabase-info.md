
# supabase.py
## class Database:

```

    def _generate_character_id(self, name: str) -> str:
        """Generate a sequential ID from the character name."""

        base_id = name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)
        base_id = re.sub(r'\s+', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        base_id = base_id.strip('-')

        try:
            response = self.supabase.table("characters")\
                .select("id")\
                .like("id", f"{base_id}-%")\
                .execute()

            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")

            for row in response.data:
                match = pattern.match(row["id"])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)

            next_num = highest_num + 1
            character_id = f"{base_id}-{next_num:03d}"

            logger.info(f"Generated character id: {character_id}")
            return character_id

        except Exception as e:
            logger.error(f"Error generating character id: {e}")
            return f"{base_id}-001"

```

### Characters

generate_character_id
get_all_characters
get_active_characters
get_character
create_character
update_character
delete_character
search_characters

```

class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    global_roleplay: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False
    last_message: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class CharacterCreate(BaseModel):
    name: str
    voice: str = ""
    global_roleplay: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False

class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    voice: Optional[str] = None
    global_roleplay: Optional[str] = None
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None
    last_message: Optional[str] = None

```

### Voices

get_all_voices
get_voice
create_voice
update_voice
delete_voice

```

class Voice(BaseModel):
    voice: str  # Primary key
    method: str = ""
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""
    audio_tokens: Optional[Any] = None
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class VoiceCreate(BaseModel):
    voice: str
    method: str = ""
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""

class VoiceUpdate(BaseModel):
    method: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    speaker_desc: Optional[str] = None
    scene_prompt: Optional[str] = None
    audio_tokens: Optional[Any] = None

```

### Conversations

generate_conversation_title
create_conversation
get_conversation
get_all_conversations
update_conversation
update_conversation_title
add_character_to_conversation
remove_character_from_conversation
delete_conversation

```

class Conversation(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ConversationCreate(BaseModel):
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    active_characters: Optional[List[Dict[str, Any]]] = None

```

### Messages

create_message
get_messages
get_last_message
delete_message

```

class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    name: Optional[str] = None
    content: str
    character_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    name: Optional[str] = None
    character_id: Optional[str] = None
    
class MessageUpdate(BaseModel):
    conversation_id: str
    role: str
    content: str
    name: Optional[str] = None
    character_id: Optional[str] = None

```