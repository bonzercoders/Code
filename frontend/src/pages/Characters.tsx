import { useEffect, useMemo, useState } from 'react'

import CharacterDirectory from '@/components/characters/CharacterDirectory'
import CharacterEditor from '@/components/characters/CharacterEditor'
import { type Character } from '@/components/characters/types'
import { cn } from '@/lib/utils'

const STORAGE_KEY = 'bosun.characters'
const EDITOR_ENTER_MS = 280
const EDITOR_EXIT_MS = 200

type DraftState = {
  draft: Character
  isNew: boolean
}

const createCharacter = (id: string): Character => ({
  id,
  name: '',
  systemPrompt: '',
  globalPrompt: '',
  voice: undefined,
  imageDataUrl: undefined,
})

const createId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `character-${Date.now()}`
}

const loadStoredCharacters = (): Character[] => {
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
      .filter((item): item is Partial<Character> & { id: string } => {
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
        systemPrompt: item.systemPrompt ?? '',
        globalPrompt: item.globalPrompt ?? '',
        voice: item.voice,
        imageDataUrl: item.imageDataUrl,
      }))
  } catch {
    return []
  }
}

function CharactersPage() {
  const [characters, setCharacters] = useState<Character[]>(() =>
    loadStoredCharacters()
  )
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeDraft, setActiveDraft] = useState<DraftState | null>(null)
  const [displayedDraft, setDisplayedDraft] = useState<DraftState | null>(null)
  const [transitionState, setTransitionState] = useState<
    'idle' | 'enter' | 'exit'
  >('idle')

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(characters))
  }, [characters])

  useEffect(() => {
    if (
      selectedId &&
      !characters.some((character) => character.id === selectedId)
    ) {
      setSelectedId(null)
    }
  }, [characters, selectedId])

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

  const displayedCharacter = useMemo(() => {
    return displayedDraft?.draft ?? null
  }, [displayedDraft])

  const handleCreate = () => {
    const id = createId()
    const newCharacter = createCharacter(id)
    setSelectedId(null)
    setActiveDraft({ draft: newCharacter, isNew: true })
  }

  const handleSelect = (id: string) => {
    const selected = characters.find((character) => character.id === id)
    if (!selected) {
      return
    }
    setSelectedId(id)
    setActiveDraft({ draft: { ...selected }, isNew: false })
  }

  const handleDraftChange = (updates: Partial<Character>) => {
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
    setCharacters((previous) =>
      previous.filter((character) => character.id !== id)
    )
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
      setCharacters((previous) => [draft, ...previous])
    } else {
      setCharacters((previous) =>
        previous.map((character) =>
          character.id === draft.id ? { ...draft } : character
        )
      )
    }
    setSelectedId(null)
    setActiveDraft(null)
  }

  const handleChat = () => {}

  const handleClose = () => {
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
    <div className="h-full w-full p-6">
      <div className="grid h-full min-h-0 w-full gap-8 lg:grid-cols-[1.25fr_3fr]">
        <div className="flex h-full min-h-0 w-full flex-col panel-fade-in">
          <CharacterDirectory
            characters={characters}
            selectedId={selectedId}
            onSelect={handleSelect}
            onCreate={handleCreate}
          />
        </div>
        <div className="flex h-full min-h-0 w-full flex-col">
          {displayedCharacter ? (
            <div className={cn('h-full w-full', editorAnimationClass)}>
              <CharacterEditor
                key={displayedCharacter.id}
                character={displayedCharacter}
                onChat={handleChat}
                onChange={handleDraftChange}
                onClose={handleClose}
                onDelete={handleDelete}
                onSave={handleSave}
              />
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-[#2b3139] bg-[#13161a]/40 text-sm text-[#7a828c]">
              Select a character to edit or create a new one.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default CharactersPage
