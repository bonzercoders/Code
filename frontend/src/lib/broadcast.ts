import { supabase } from '@/lib/supabase'

/**
 * Notify the backend that a character was created, updated, or deleted.
 * The backend subscribes to 'db-characters' and refreshes ChatLLM.active_characters.
 */
export async function broadcastCharacterChange(
  action: 'created' | 'updated' | 'deleted',
  characterId: string
) {
  const channel = supabase.channel('db-characters')
  await channel.send({
    type: 'broadcast',
    event: 'character-changed',
    payload: { action, characterId },
  })
}

/**
 * Notify the backend that a voice was created, updated, or deleted.
 * The backend subscribes to 'db-voices' and clears its voice cache.
 */
export async function broadcastVoiceChange(
  action: 'created' | 'updated' | 'deleted',
  voiceName: string
) {
  const channel = supabase.channel('db-voices')
  await channel.send({
    type: 'broadcast',
    event: 'voice-changed',
    payload: { action, voiceName },
  })
}
