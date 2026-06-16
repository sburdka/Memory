# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemoryOS is a vendor-independent memory infrastructure layer that decouples memory from AI reasoning. It allows users to retain and retrieve context across different AI providers (OpenAI, Claude, Gemini) so that memory persists when switching models.

```
User → MemoryOS → GPT / Claude / Gemini
```

## Planned Stack

**Backend:** FastAPI, Python 3.12, SQLAlchemy, PostgreSQL, Qdrant (vector DB)
**Frontend:** Next.js 15, TypeScript, Tailwind CSS
**AI Providers:** OpenAI, Anthropic Claude
**Deployment:** Docker Compose

## Architecture

```
Frontend (Next.js)
    │
    ▼
FastAPI
    ├── Memory Service       — store/retrieve memories (episodic, semantic, reflective)
    ├── Retrieval Service    — semantic search via Qdrant
    ├── Reflection Service   — generates higher-level insights every 20 memories
    └── LLM Router           — routes to OpenAI or Claude, injects enriched prompt
```

### Memory Types

- **Episodic** — events ("User started learning Python")
- **Semantic** — facts ("User knows FastAPI")
- **Reflective** — generated insights ("User is pursuing backend engineering") — triggered every 20 stored memories

### Memory Lifecycle (per request)

1. Retrieve relevant memories from Qdrant (top 10 by semantic similarity)
2. Inject memories into the prompt before calling the LLM
3. Call the selected LLM (OpenAI or Claude)
4. Extract new memory facts from the response
5. Store extracted memories in PostgreSQL + Qdrant

## Database Schema

**`memories` table:** `id`, `user_id`, `content`, `memory_type`, `importance_score`, `created_at`, `updated_at`

**`insights` table:** `id`, `user_id`, `content`, `confidence_score`, `created_at`

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Main entry point — retrieves memory, enriches prompt, calls LLM, extracts and saves new memory |
| `POST` | `/memory` | Create a memory directly |
| `GET` | `/memory/{user_id}` | Return all memories for a user |
| `POST` | `/memory/search` | Semantic search over stored memories |

### `/chat` request shape
```json
{ "user_id": "123", "model": "openai", "message": "Suggest a project" }
```

## Memory Extraction

The extraction pipeline parses LLM input/output for structured facts:

```json
[
  { "type": "skill", "value": "Python" },
  { "type": "goal", "value": "Backend Job" }
]
```

Extract only: `skill`, `goal`, `preference`, `interest`, `experience`. Ignore generic conversation.

## Prompt Enrichment Format

Prepend this block to every LLM call (top 10 most relevant memories by vector similarity):

```
Relevant User Memory:
- Learning Python
- Interested in SaaS
- Uses FastAPI

User Query:
<user message>
```

## Frontend Pages

1. **Chat** — select GPT or Claude, chat interface
2. **Memory Dashboard** — view stored memories and generated insights
3. **Model Switcher** — switch active LLM mid-session

## MVP Success Criteria

The MVP is complete when this full flow works end-to-end:

1. User chats with GPT → memory is saved
2. User switches to Claude → Claude receives stored memory automatically
3. Memory dashboard shows stored memories
4. Reflection engine generates an insight after 20+ memories
