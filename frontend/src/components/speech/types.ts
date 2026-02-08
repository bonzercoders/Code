export type VoiceMethod = 'clone' | 'profile'

export type Voice = {
  id: string
  name: string
  method: VoiceMethod
  scenePrompt: string
  referenceText: string
  referenceAudio: string
  speakerDescription: string
}
