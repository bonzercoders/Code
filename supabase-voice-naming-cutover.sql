begin;

-- 1) Rename ambiguous columns to explicit names.
do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'voices'
      and column_name = 'voice'
  ) then
    alter table public.voices rename column voice to voice_name;
  end if;
end $$;

do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'characters'
      and column_name = 'voice'
  ) then
    alter table public.characters rename column voice to voice_id;
  end if;
end $$;

-- 2) Canonicalize ref_text on voices.
alter table if exists public.voices
  add column if not exists ref_text text;

update public.voices
set ref_text = transcript
where (ref_text is null or btrim(ref_text) = '')
  and transcript is not null
  and btrim(transcript) <> '';

alter table if exists public.voices
  drop column if exists transcript;

-- 3) Normalize empty character voice IDs to NULL.
update public.characters
set voice_id = null
where voice_id is not null
  and btrim(voice_id) = '';

-- 4) Enforce character -> voice reference integrity.
create index if not exists idx_characters_voice_id
  on public.characters using btree (voice_id);

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'characters_voice_id_fkey'
      and conrelid = 'public.characters'::regclass
  ) then
    alter table public.characters
      add constraint characters_voice_id_fkey
      foreign key (voice_id)
      references public.voices (voice_id)
      on update cascade
      on delete set null;
  end if;
end $$;

commit;
