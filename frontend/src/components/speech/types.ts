export type VoiceMethod = 'clone' | 'profile'

export type Voice = {
  voiceId: string         // Maps to DB 'voice_id' (PK)
  voice: string           // Maps to DB 'voice' (human-readable display name)
  method: VoiceMethod
  scenePrompt: string
  referenceText: string
  referenceAudio: string
  speakerDescription: string
}
