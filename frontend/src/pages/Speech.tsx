import { useEffect, useMemo, useState } from 'react'

import VoiceBuilderForm from '@/components/speech/VoiceBuilderForm'
import VoiceDirectory from '@/components/speech/VoiceDirectory'
import { type Voice } from '@/components/speech/types'
import { cn } from '@/lib/utils'

const STORAGE_KEY = 'bosun.voices'
const EDITOR_ENTER_MS = 280
const EDITOR_EXIT_MS = 200

type DraftState = {
  draft: Voice
  isNew: boolean
}

const createVoice = (id: string): Voice => ({
  id,
  name: '',
  method: 'clone',
  scenePrompt: '',
  referenceText: '',
  referenceAudio: '',
  speakerDescription: '',
})

const createId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `voice-${Date.now()}`
}

const loadStoredVoices = (): Voice[] => {
  if (typeof window === 'undefined') {
    return []
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (!stored) {
      return []
    }
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed
      .filter((item): item is Partial<Voice> & { id: string } => {
        return (
          typeof item === 'object' &&
          item !== null &&
          'id' in item &&
          typeof item.id === 'string'
        )
      })
      .map((item) => ({
        id: item.id,
        name: item.name ?? '',
        method: item.method === 'profile' ? 'profile' : 'clone',
        scenePrompt: item.scenePrompt ?? '',
        referenceText: item.referenceText ?? '',
        referenceAudio: item.referenceAudio ?? '',
        speakerDescription: item.speakerDescription ?? '',
      }))
  } catch {
    return []
  }
}

function SpeechPage() {
  const [voices, setVoices] = useState<Voice[]>(() => loadStoredVoices())
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeDraft, setActiveDraft] = useState<DraftState | null>(null)
  const [displayedDraft, setDisplayedDraft] = useState<DraftState | null>(null)
  const [transitionState, setTransitionState] = useState<
    'idle' | 'enter' | 'exit'
  >('idle')

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(voices))
  }, [voices])

  useEffect(() => {
    if (selectedId && !voices.some((voice) => voice.id === selectedId)) {
      setSelectedId(null)
    }
  }, [voices, selectedId])

  useEffect(() => {
    const activeId = activeDraft?.draft.id ?? null
    const displayedId = displayedDraft?.draft.id ?? null

    if (activeId === displayedId) {
      return
    }

    if (!activeDraft) {
      if (!displayedDraft) {
        return
      }
      setTransitionState('exit')
      const timeout = window.setTimeout(() => {
        setDisplayedDraft(null)
        setTransitionState('idle')
      }, EDITOR_EXIT_MS)
      return () => window.clearTimeout(timeout)
    }

    if (!displayedDraft) {
      setDisplayedDraft(activeDraft)
      setTransitionState('enter')
      return
    }

    setTransitionState('exit')
    const timeout = window.setTimeout(() => {
      setDisplayedDraft(activeDraft)
      setTransitionState('enter')
    }, EDITOR_EXIT_MS)
    return () => window.clearTimeout(timeout)
  }, [activeDraft, displayedDraft])

  useEffect(() => {
    if (transitionState !== 'enter') {
      return
    }
    const timeout = window.setTimeout(() => {
      setTransitionState('idle')
    }, EDITOR_ENTER_MS)
    return () => window.clearTimeout(timeout)
  }, [transitionState])

  useEffect(() => {
    if (!activeDraft || !displayedDraft) {
      return
    }
    if (transitionState === 'exit') {
      return
    }
    if (
      activeDraft.draft.id === displayedDraft.draft.id &&
      (activeDraft.draft !== displayedDraft.draft ||
        activeDraft.isNew !== displayedDraft.isNew)
    ) {
      setDisplayedDraft(activeDraft)
    }
  }, [activeDraft, displayedDraft, transitionState])

  const displayedVoice = useMemo(() => {
    return displayedDraft?.draft ?? null
  }, [displayedDraft])

  const handleCreate = () => {
    const id = createId()
    const newVoice = createVoice(id)
    setSelectedId(null)
    setActiveDraft({ draft: newVoice, isNew: true })
  }

  const handleSelect = (id: string) => {
    const selected = voices.find((voice) => voice.id === id)
    if (!selected) {
      return
    }
    setSelectedId(id)
    setActiveDraft({ draft: { ...selected }, isNew: false })
  }

  const handleDraftChange = (updates: Partial<Voice>) => {
    setActiveDraft((previous) => {
      if (!previous) {
        return previous
      }
      const next = { ...previous, draft: { ...previous.draft, ...updates } }
      setDisplayedDraft((current) => {
        if (!current || current.draft.id !== previous.draft.id) {
          return current
        }
        return { ...current, draft: { ...current.draft, ...updates } }
      })
      return next
    })
  }

  const handleDelete = () => {
    if (!activeDraft) {
      return
    }
    if (activeDraft.isNew) {
      setActiveDraft(null)
      return
    }

    const id = activeDraft.draft.id
    setVoices((previous) => previous.filter((voice) => voice.id !== id))
    if (selectedId === id) {
      setSelectedId(null)
    }
    setActiveDraft(null)
  }

  const handleSave = () => {
    if (!activeDraft) {
      return
    }

    const { draft, isNew } = activeDraft
    if (isNew) {
      setVoices((previous) => [draft, ...previous])
    } else {
      setVoices((previous) =>
        previous.map((voice) => (voice.id === draft.id ? { ...draft } : voice))
      )
    }
    setSelectedId(null)
    setActiveDraft(null)
  }

  const editorAnimationClass = cn(
    transitionState === 'enter'
      ? 'editor-fade-in'
      : transitionState === 'exit'
        ? 'editor-fade-out'
        : '',
    transitionState === 'exit' ? 'pointer-events-none' : ''
  )

  return (
    <div className="flex h-full w-full flex-col p-8">
      <div className="flex min-h-0 flex-1 flex-col gap-6 xl:flex-row">
        <div className="flex w-full flex-1 flex-col xl:w-[600px] xl:flex-none panel-fade-in">
          <VoiceDirectory
            voices={voices}
            selectedId={selectedId}
            onSelect={handleSelect}
            onCreate={handleCreate}
          />
        </div>
        <div className="flex w-full flex-1 flex-col">
          {displayedVoice ? (
            <div className={cn('h-full w-full', editorAnimationClass)}>
              <VoiceBuilderForm
                key={displayedVoice.id}
                voice={displayedVoice}
                onChange={handleDraftChange}
                onDelete={handleDelete}
                onSave={handleSave}
              />
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-[#2b3139] bg-[#13161a]/40 text-sm text-[#7a828c]">
              Select a voice to edit or create a new one.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default SpeechPage
