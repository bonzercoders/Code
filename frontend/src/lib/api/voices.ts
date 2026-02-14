import { supabase } from '@/lib/supabase'
import type { Voice } from '@/components/speech/types'

/**
 * Map a Supabase row (DB columns) → frontend Voice type.
 */
function fromDb(row: Record<string, unknown>): Voice {
  return {
    voiceId: row.voice_id as string,        // DB PK
    voice: row.voice as string,             // Display name
    method: (row.method as Voice['method']) === 'profile' ? 'profile' : 'clone',
    scenePrompt: (row.scene_prompt as string) ?? '',
    referenceText: (row.transcript as string) ?? '',
    referenceAudio: (row.ref_audio as string) ?? '',
    speakerDescription: (row.speaker_desc as string) ?? '',
  }
}

/**
 * Map a frontend Voice → Supabase row for insert/update.
 */
function toDb(v: Voice) {
  return {
    voice_id: v.voiceId,      // PK
    voice: v.voice,           // Display name
    method: v.method,
    scene_prompt: v.scenePrompt,
    transcript: v.referenceText,
    ref_audio: v.referenceAudio,
    speaker_desc: v.speakerDescription,
  }
}

export async function fetchVoices(): Promise<Voice[]> {
  const { data, error } = await supabase
    .from('voices')
    .select('*')
    .order('created_at', { ascending: false })

  if (error) throw error
  return (data ?? []).map(fromDb)
}

export async function createVoice(v: Voice): Promise<Voice> {
  const { data, error } = await supabase
    .from('voices')
    .insert(toDb(v))
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function updateVoice(v: Voice): Promise<Voice> {
  const { data, error } = await supabase
    .from('voices')
    .update(toDb(v))
    .eq('voice_id', v.voiceId)
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function deleteVoice(voiceId: string): Promise<void> {
  const { error } = await supabase
    .from('voices')
    .delete()
    .eq('voice_id', voiceId)

  if (error) throw error
}
