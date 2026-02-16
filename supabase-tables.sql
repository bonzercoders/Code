SUPABASE_URL = 'https://wnqozfubwnwqksjkksfy.supabase.co';
Publishable Key = 'sb_publishable_T28Uvx3MVYHo0opH0Eweqw_kCCNFuvC';

create table public.characters (
  name text not null,
  id text not null,
  voice text null,
  global_roleplay text null,
  system_prompt text null,
  is_active boolean null default False,
  image_url text null,
  images jsonb null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint characters_pkey primary key (id),
  constraint characters_character_id_key unique (id)
) TABLESPACE pg_default;

create index IF not exists idx_characters_character_id on public.characters using btree (id) TABLESPACE pg_default;

create trigger update_characters_updated_at BEFORE
update on characters for EACH row
execute FUNCTION update_updated_at_column ();

create table public.voices (
  voice text not null,
  voice_id text not null,
  method text null,
  ref_audio text null,
  transcript text null,
  speaker_desc text null,
  scene_prompt text null,
  audio_ids jsonb null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint voices_pkey primary key (voice_id)
) TABLESPACE pg_default;

create trigger update_voices_updated_at BEFORE
update on voices for EACH row
execute FUNCTION update_updated_at_column ();

create table public.conversations (
  conversation_id uuid not null default extensions.uuid_generate_v4 (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  title text null,
  active_characters jsonb[] null,
  constraint conversations_pkey primary key (conversation_id)
) TABLESPACE pg_default;

create trigger update_conversations_updated_at BEFORE
update on conversations for EACH row
execute FUNCTION update_updated_at_column ();

create table public.messages (
  message_id uuid not null default extensions.uuid_generate_v4 (),
  conversation_id uuid not null,
  role text not null,
  name text null,
  content text not null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  character_id text null,
  constraint messages_pkey primary key (message_id),
  constraint messages_character_id_fkey foreign KEY (character_id) references characters (id),
  constraint messages_conversation_id_fkey foreign KEY (conversation_id) references conversations (conversation_id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_messages_conversation_id on public.messages using btree (conversation_id) TABLESPACE pg_default;
create index IF not exists idx_messages_character_name on public.messages using btree (name) TABLESPACE pg_default;
create index IF not exists idx_messages_created_at on public.messages using btree (created_at) TABLESPACE pg_default;

create trigger update_messages_updated_at BEFORE
update on messages for EACH row
execute FUNCTION update_updated_at_column ();

create table public.user_settings (
  id uuid not null default gen_random_uuid (),
  setting_key text not null,
  setting_value jsonb not null,
  updated_at timestamp with time zone null default now(),
  constraint user_settings_pkey primary key (id),
  constraint user_settings_setting_key_key unique (setting_key)
) TABLESPACE pg_default;