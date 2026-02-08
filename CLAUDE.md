# AGENTS.md

## Project Overview

Low-latency voice chat application with a FastAPI (Python) backend and vanilla JavaScript frontend. Single-user application — do not over-engineer for enterprise scale or multi-tenancy.

## Principles/Style

- Adhere to KISS, YAGNI principles.
- Functional programming style.
- Write code a human can read and maintain.

## Programming Languages

**Python**

  *Preferred patterns:*
  - `asyncio.create_task()` for fire-and-forget or managed background work
  - Producer/consumer with `asyncio.Queue`
  - `asyncio.as_completed()` when processing results as they arrive

  *Avoid:*
  - Overly nested async context managers
  - Complex task orchestration frameworks when simple queues suffice

**JavaScript** (vanilla)
  Functional programming preferred. Avoid classes and OOP patterns; use pure functions and closures.
  ES Modules: Keep imports/exports simple and flat. Avoid circular dependencies or complex re-exports that break loading.

### Key Libraries

- RealtimeSTT (faster-whisper) — speech-to-text
- Higgs Audio — TTS service
- SQLite — database
- FastAPI — backend server
- stream2sentence — sentence boundary detection

## Code Organization

Code
├── backend
│   ├── bosun_multimodal
│   ├── RealtimeSTT
│   ├── database.db
│   ├── database_director.py
│   ├── fastserver.py
│   └── stream2sentence.py
├── frontend
│   ├── characterCache.js
│   ├── characters.css
│   ├── characters.js
│   ├── chat.js
│   ├── db.js
│   ├── editor.css
│   ├── editor.js
│   ├── index.html
│   ├── main.js
│   ├── speech.css
│   ├── speech.js
│   ├── stt-audio.js
│   ├── stt-processor.js
│   ├── tts-audio.js
│   └── websocket.js
├── CLAUDE.md
├── AGENTS.md
├── requirements_higgs.txt
└── setup.sh

## What NOT To Do

1. **Don't over-engineer.** No abstract base classes, factory patterns, or dependency injection unless explicitly asked.
5. **Don't wrap simple operations in unnecessary classes.** A function is usually fine.