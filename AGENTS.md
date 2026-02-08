# AGENTS.md
## Project Overview
Low-latency voice chat application with a FastAPI (Python) backend and JavaScript (React TypeScript) frontend. Single-user application — do not over-engineer for enterprise scale or multi-tenancy.

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


### Key Libraries
- RealtimeSTT (faster-whisper) — speech-to-text
- Higgs Audio — TTS service
- SQLite — database
- FastAPI — backend server
- stream2sentence — sentence boundary detection
- shadcn ui

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
├── CLAUDE.md
├── AGENTS.md
├── requirements_higgs.txt
└── setup.sh