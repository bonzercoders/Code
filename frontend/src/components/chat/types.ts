export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  name: string | null
  characterId: string | null
  content: string
  isStreaming: boolean
}
