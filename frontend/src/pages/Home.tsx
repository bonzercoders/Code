import type { ComponentType, ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  Bold,
  Check,
  ChevronDown,
  ChevronsUpDown,
  Code,
  Highlighter,
  Italic,
  List,
  Mic,
  Paperclip,
  Plus,
  Quote,
  Redo2,
  Send,
  Undo2,
} from 'lucide-react'

import arrowLeft from '@/assets/arrow-left.png'
import arrowRight from '@/assets/arrow-right.png'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

type ToolbarButtonProps = {
  label: string
  icon?: ComponentType<{ className?: string }>
  children?: ReactNode
}

function ToolbarButton({ label, icon: Icon, children }: ToolbarButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-7 items-center justify-center gap-1 rounded-md px-2 text-[11px] text-[#a7aeb7]',
        'hover:bg-[#22262d] hover:text-[#e3e7ec]'
      )}
    >
      {Icon ? <Icon className="h-3.5 w-3.5" /> : children}
    </button>
  )
}

type ModelOption = {
  value: string
  label: string
}

type ModelGroup = {
  label: string
  options: ModelOption[]
}

const modelGroups: ModelGroup[] = [
  {
    label: 'Creative',
    options: [
      { value: 'mistral-small-creative', label: 'Mistral Small Creative' },
      { value: 'claude-sonnet', label: 'Claude 3.5 Sonnet' },
      { value: 'gpt-4o', label: 'GPT-4o' },
    ],
  },
  {
    label: 'Balanced',
    options: [
      { value: 'mistral-medium', label: 'Mistral Medium' },
      { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini' },
      { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
    ],
  },
  {
    label: 'Fast',
    options: [
      { value: 'mistral-nemo', label: 'Mistral Nemo' },
      { value: 'gpt-4.1-nano', label: 'GPT-4.1 Nano' },
      { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash' },
    ],
  },
]

function ModelCombobox() {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState(modelGroups[0].options[0].value)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const [triggerWidth, setTriggerWidth] = useState<number | null>(null)

  const selected = modelGroups
    .flatMap((group) => group.options)
    .find((option) => option.value === value)

  useEffect(() => {
    if (!open) {
      return
    }

    const width = triggerRef.current?.offsetWidth ?? null
    setTriggerWidth(width)
  }, [open])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          ref={triggerRef}
          className={cn(
            'flex w-full items-center justify-between rounded-lg border border-[#2c323a] bg-[#171a1f] px-3 py-2 text-xs text-[#c9cfd6]',
            'transition-[border-color,color,box-shadow] hover:border-[#77bef554] hover:text-white hover:shadow-[0_0_0_1px_rgba(0,122,204,0.45)]'
          )}
        >
          <span className="truncate text-left">
            {selected ? selected.label : 'Select model...'}
          </span>
          <ChevronsUpDown className="h-4 w-4 text-[#7b838d]" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="border border-[#2c323a] bg-[#171a1f] p-0 text-[#d7dce2]"
        align="start"
        style={triggerWidth ? { width: `${triggerWidth}px` } : undefined}
      >
        <Command
          className={cn(
            'bg-transparent text-[#d7dce2]',
            '[&_[data-slot=command-input-wrapper]]:border-[#2c323a]',
            '[&_[data-slot=command-input-wrapper]_svg]:text-[#7b838d]',
            '[&_[data-slot=command-item][data-selected=true]]:bg-[#242a31]',
            '[&_[data-slot=command-item][data-selected=true]]:text-white'
          )}
        >
          <CommandInput
            placeholder="Search models..."
            className="text-sm text-[#d7dce2] placeholder:text-[#7b838d]"
          />
          <CommandList className="model-combobox-list max-h-[280px]">
            <CommandEmpty className="text-[#8b93a0]">
              No models found.
            </CommandEmpty>
            {modelGroups.map((group, index) => (
              <div key={group.label}>
                <CommandGroup heading={group.label}>
                  {group.options.map((option) => (
                    <CommandItem
                      key={option.value}
                      value={option.value}
                      onSelect={(currentValue) => {
                        setValue(currentValue)
                        setOpen(false)
                      }}
                      className="text-sm text-[#c9cfd6] data-[selected=true]:text-white"
                    >
                      {option.label}
                      <Check
                        className={cn(
                          'ml-auto h-4 w-4 text-[#6fc3ff] transition-opacity',
                          value === option.value ? 'opacity-100' : 'opacity-0'
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
                {index < modelGroups.length - 1 ? (
                  <CommandSeparator className="bg-[#242a31]" />
                ) : null}
              </div>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

function HomePage() {
  const [leftOpen, setLeftOpen] = useState(false)
  const [rightOpen, setRightOpen] = useState(false)
  const leftDrawerWidth = 320
  const rightDrawerWidth = 320

  return (
    <div className="relative flex h-full w-full overflow-hidden">
      <div
        className="pointer-events-none absolute inset-y-0 left-0 z-10 flex"
        style={{ width: leftDrawerWidth }}
      >
        <div
          className={cn(
            'pointer-events-auto h-full w-full rounded-2xl border border-[#2d3138] bg-[#191c21]',
            leftOpen && 'shadow-[0_20px_45px_rgba(0,0,0,0.4)]',
            'transition-transform duration-300 ease-out'
          )}
          style={{
            transform: `translateX(${leftOpen ? 0 : -leftDrawerWidth}px)`,
          }}
        >
          <div className="flex h-full flex-col gap-4 p-6 text-sm text-[#cbd1d8]">
            <div className="text-base font-semibold text-white">Info Panel</div>
            <div className="h-px w-full bg-[#2a2f36]" />
            <div className="text-xs text-[#9aa2ab]">
              Placeholder content for the left drawer.
            </div>
          </div>
        </div>
        <button
          type="button"
          aria-label={leftOpen ? 'Collapse left drawer' : 'Expand left drawer'}
          onClick={() => setLeftOpen((value) => !value)}
          className="pointer-events-auto absolute left-0 top-1/2 z-20 transition-transform duration-300 ease-out"
          style={{
            transform: `translate(${leftOpen ? leftDrawerWidth : 0}px, -50%)`,
          }}
        >
          <img
            src={arrowRight}
            alt=""
            className={cn(
              'h-8 w-8 transition-transform duration-300 ease-out',
              leftOpen && 'rotate-180'
            )}
          />
        </button>
      </div>

      <div className="flex-1" />

      <div
        className="pointer-events-none absolute inset-y-0 right-0 z-10 flex justify-end"
        style={{ width: rightDrawerWidth }}
      >
        <div
          className={cn(
            'pointer-events-auto h-full w-full rounded-2xl border border-[#2d3138] bg-[#191c21]',
            rightOpen && 'shadow-[0_20px_45px_rgba(0,0,0,0.4)]',
            'transition-transform duration-300 ease-out'
          )}
          style={{
            transform: `translateX(${rightOpen ? 0 : rightDrawerWidth}px)`,
          }}
        >
          <div className="flex h-full flex-col gap-4 p-6 text-sm text-[#cbd1d8]">
            <div className="text-base font-semibold text-white">
              Model Settings
            </div>
            <div className="h-px w-full bg-[#2a2f36]" />
            <div className="space-y-3">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-[#8b929b]">
                Model
              </div>
              <ModelCombobox />
            </div>
          </div>
        </div>
        <button
          type="button"
          aria-label={rightOpen ? 'Collapse right drawer' : 'Expand right drawer'}
          onClick={() => setRightOpen((value) => !value)}
          className="pointer-events-auto absolute right-0 top-1/2 z-20 transition-transform duration-300 ease-out"
          style={{
            transform: `translate(${rightOpen ? -rightDrawerWidth : 0}px, -50%)`,
          }}
        >
          <img
            src={arrowLeft}
            alt=""
            className={cn(
              'h-8 w-8 transition-transform duration-300 ease-out',
              rightOpen && 'rotate-180'
            )}
          />
        </button>
      </div>

      <div className="absolute bottom-6 left-1/2 w-[min(780px,92%)] -translate-x-1/2">
        <div className="overflow-hidden rounded-2xl border border-[#333] bg-[#1b1e23] shadow-[0_20px_50px_rgba(0,0,0,0.55)]">
          <div className="flex items-center gap-1 border-b border-[#333] px-3 py-2">
            <ToolbarButton label="Undo" icon={Undo2} />
            <ToolbarButton label="Redo" icon={Redo2} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <div className="relative">
              <select
                className={cn(
                  'h-7 w-[110px] appearance-none rounded-md border border-transparent bg-transparent px-2 text-[11px] text-[#a7aeb7]',
                  'hover:bg-[#22262d] hover:text-[#e3e7ec] focus:border-[#3c424b] focus:outline-none'
                )}
                defaultValue="Paragraph"
                aria-label="Text style"
              >
                <option>Paragraph</option>
                <option>H1</option>
                <option>H2</option>
                <option>H3</option>
                <option>H4</option>
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-[#6f7782]" />
            </div>
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="List" icon={List} />
            <ToolbarButton label="Quote" icon={Quote} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="Bold" icon={Bold} />
            <ToolbarButton label="Italic" icon={Italic} />
            <ToolbarButton label="Code" icon={Code} />
            <ToolbarButton label="Highlight" icon={Highlighter} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="Align left" icon={AlignLeft} />
            <ToolbarButton label="Align center" icon={AlignCenter} />
            <ToolbarButton label="Align right" icon={AlignRight} />
            <div className="ml-auto flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 rounded-full border border-[#3f4650] bg-[#2a2e34]"
                aria-label="Connection status"
                title="Connection status"
              />
            </div>
          </div>

          <textarea
            className={cn(
              'min-h-[140px] w-full resize-none bg-transparent px-4 py-3 text-sm text-[#dfe3e8] outline-none',
              'placeholder:text-[#6c7480]'
            )}
            placeholder="Type your message..."
          />

          <div className="flex items-center justify-between px-3 py-3">
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#2f353d] bg-[#22252a] text-[#c8cdd3] hover:border-[#3f4650] hover:text-white"
                aria-label="Attach file"
                title="Attach file"
              >
                <Paperclip className="h-4 w-4 text-[#7fd2ff]" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#2f353d] bg-[#22252a] text-[#c8cdd3] hover:border-[#3f4650] hover:text-white"
                aria-label="Add options"
                title="Add options"
              >
                <Plus className="h-4 w-4" />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-[#2f353d] bg-[#22252a] text-[#c8cdd3] hover:border-[#3f4650] hover:text-white"
                aria-label="Microphone"
                title="Microphone"
              >
                <Mic className="h-4 w-4" />
              </button>
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-lg bg-[#007acc] px-4 py-2 text-xs font-semibold text-white shadow-[0_8px_18px_rgba(0,122,204,0.35)] hover:bg-[#1087d9]"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HomePage
