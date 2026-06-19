# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Project Overview

MemoryOS is a vendor-independent memory infrastructure layer that decouples memory from AI reasoning. It allows users to retain and retrieve context across different AI providers (OpenAI, Anthropic) so that memory persists when switching models.

```
User → MemoryOS → GPT-4o / Anthropic
```

## Stack

**Backend:** FastAPI, Python 3.12, SQLAlchemy (async), PostgreSQL, Qdrant (vector DB)
**Frontend:** Next.js 15, TypeScript, Tailwind CSS
**AI Providers:** OpenAI (GPT-4o + text-embedding-3-small), Anthropic
**Deployment:** Docker Compose

## Running the Project

### Docker (recommended)
```bash
cp .env.example .env          # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
docker compose up --build
```

Services: backend on :8000, frontend on :3000, PostgreSQL on :5432, Qdrant on :6333.

### Backend locally
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Requires PostgreSQL and Qdrant running. Override DATABASE_URL and QDRANT_URL via `.env`.

### Frontend locally
```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
npm run build        # production build
npm run lint         # ESLint
```

## Architecture

### Backend structure (`backend/app/`)
```
main.py              — FastAPI app, CORS, lifespan (DB + Qdrant init)
config.py            — Pydantic Settings from env vars
database.py          — Async SQLAlchemy engine + session factory
models/memory.py     — ORM: Memory, Insight tables
schemas/memory.py    — Pydantic request/response models
routers/
  chat.py            — POST /chat  (main entry point)
  memory.py          — POST /memory, GET /memory/{user_id}, POST /memory/search
services/
  retrieval_service.py  — Qdrant client, embed(), upsert_memory(), search_memories()
  memory_service.py     — PostgreSQL CRUD: store_memory(), get_all(), get_count()
  llm_router.py         — call_llm(), build_system_prompt(), extract_memories()
  reflection_service.py — should_reflect(), generate_insight()
```

### Frontend structure (`frontend/`)
```
app/layout.tsx           — Root layout with <Navigation />
app/page.tsx             — Chat page
app/dashboard/page.tsx   — Memory Dashboard page
components/
  Navigation.tsx         — Top nav (Chat | Memory Dashboard)
  ChatInterface.tsx      — Client component: model selector, messages, send
  MemoryDashboard.tsx    — Client component: load + display memories by type
lib/api.ts               — All fetch() calls to the backend API
```

## Memory Lifecycle (per `/chat` request)

1. `retrieval_service.search_memories()` — embed query, search Qdrant top-10 by user_id filter
2. `llm_router.build_system_prompt()` — prepend memory block to system prompt
3. `llm_router.call_llm()` — route to OpenAI or Anthropic
4. `llm_router.extract_memories()` — extract structured facts via `gpt-4o-mini` with `response_format: json_object`
5. `memory_service.store_memory()` — write to PostgreSQL, then upsert embedding to Qdrant
6. `reflection_service.should_reflect()` — trigger every `REFLECTION_THRESHOLD` (default 20) memories; generate insight and store as `reflective` type

## Memory Types

- **episodic** — events, stored directly or via manual `/memory` endpoint
- **semantic** — structured facts extracted from conversations (`Skill: Python`, `Goal: Backend Job`)
- **reflective** — LLM-generated insights triggered every 20 memories, `importance_score=1.5`

## Prompt Enrichment Format

```
Relevant User Memory:
- Skill: Python
- Goal: Backend engineering role
- Interest: AI infrastructure

[User message follows]
```

## Memory Extraction

`llm_router.extract_memories()` calls `gpt-4o-mini` and expects:
```json
{"memories": [{"type": "skill", "value": "Python"}, {"type": "goal", "value": "Backend Job"}]}
```
Allowed types: `skill`, `goal`, `preference`, `interest`, `experience`.

## Database Schema

**`memories`**: `id` (UUID PK), `user_id`, `content`, `memory_type`, `importance_score`, `created_at`, `updated_at`

**`insights`**: `id` (UUID PK), `user_id`, `content`, `confidence_score`, `created_at`

Tables are auto-created on startup via `Base.metadata.create_all`.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Main flow: retrieve → enrich → call LLM → extract → store |
| `POST` | `/memory` | Create a memory directly |
| `GET` | `/memory/{user_id}` | All memories for a user |
| `POST` | `/memory/search` | Semantic search (returns `list[str]` of content) |
| `GET` | `/health` | Health check |

### `/chat` request
```json
{ "user_id": "123", "model": "openai", "message": "Suggest a project" }
```

## Environment Variables

All config lives in `backend/app/config.py` (Pydantic Settings). Key vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | postgres+asyncpg://... | PostgreSQL connection |
| `QDRANT_URL` | http://localhost:6333 | Qdrant vector DB |
| `OPENAI_API_KEY` | — | Required for chat + embeddings + extraction |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic model |
| `EMBEDDING_MODEL` | text-embedding-3-small | 1536-dim vectors |
| `REFLECTION_THRESHOLD` | 20 | Memories before insight generation |

## Qdrant Details

- Collection `memories` is auto-created on startup (1536-dim COSINE distance)
- Each point: `id=memory_uuid`, `vector=embedding`, `payload={user_id, content}`
- User isolation uses a Qdrant `Filter` on `payload.user_id` at search time
- The async client (`AsyncQdrantClient`) is a module-level singleton in `retrieval_service.py`

## Reference Project

The `MemGovern` research project (uploaded as reference) uses the same architectural pattern — a sidecar Flask server with ChromaDB for semantic search — but targets SWE-Agent/code-agent benchmarks rather than cross-provider user memory. Key patterns borrowed:
- Separate retrieval server with `/search` + `/get_experience` endpoints
- ChromaDB → replaced with Qdrant in this project
- SentenceTransformers embeddings → replaced with OpenAI `text-embedding-3-small`
- Experience cards (structured JSON) → replaced with the simpler `{type, value}` memory extraction format

## Quantization Module (`quantization/`)

MiLo implementation — ultra-low-bit MoE quantization with adaptive low-rank compensators (MLSys 2025).

### Algorithm

1. **Group quantization** — asymmetric min-max quantization per `group_size` elements (3/4/8-bit)
2. **Low-rank compensator** — SVD of quantization error `E = W - dequant(quant(W))`, keep top-k singular values: `U_k @ V_k`
3. **Proximal optimization** — iteratively refine scale/zero to minimize `||E||`
4. **Rank strategy** — Kurtosis-based (Mixtral) or Frequency-based (DeepSeek) adaptive rank assignment

### Forward pass

```
output = F.linear(x, dequant(W_q)) + F.linear(F.linear(x, V_k), U_k) + bias
```

The compensator is applied in two cheap matmuls (rank ≪ hidden dim) instead of materializing `U @ V`.

### Module structure

```
quantization/
  bitpack.py        — BitPack: pack/unpack 2/3/4/8-bit integers
  optimize.py       — proximal optimization, grid search, AdamW scale tuning
  quantize.py       — Quantizer, MiLoLinear, BaseCompressConfig
  compensator.py    — rank_generate() (Kurtosis/Frequency/Uniform), load_compensators()
  compress.py       — CLI: python -m quantization.compress --model ... --arch mixtral
  evaluate.py       — CLI: python -m quantization.evaluate --tasks wikitext2 arc_easy
  models/
    base.py         — BasePatch (layer replacement), BaseMiLoModel (compress/save/load)
    mixtral.py      — MixtralMiLo
    deepseek.py     — DeepSeekMiLo
```

### Compress a model

```bash
pip install -r quantization/requirements.txt
python -m quantization.compress \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --output /path/to/output \
    --arch mixtral \
    --nbits 3 --group-size 64 \
    --sparse-rank 16 --dense-rank 512 \
    --rank-strategy Kurtosis
```

### Evaluate

```bash
python -m quantization.evaluate \
    --model-dir /path/to/output \
    --base-model mistralai/Mixtral-8x7B-v0.1 \
    --arch mixtral \
    --tasks wikitext2 arc_easy hellaswag
```

### Key design decisions

- **`MiLoLinear`** stores `W_q` (packed ints) + `meta` (scale/zero) + `U, V` parameters. `state_dict()` is safetensors-compatible (all tensors, no Python objects).
- **`BitPack.pack_3bit_32`** packs 10 × 3-bit values into one `int32` word.
- **`BaseCompressConfig`** is a plain `dict` subclass so it can be passed directly as `patch_params`.
- **Compensators** are saved separately (`compensators.pt`) from weights (`qmodel.pt`) to allow rank changes without re-quantizing.
- Ranks are persisted to `ranks.json` alongside the model.

## MVP Success Criteria

1. User chats with GPT-4o → memories extracted and stored
2. User switches to Anthropic → Anthropic model receives stored memories automatically
3. Memory dashboard shows memories grouped by type (episodic / semantic / reflective)
4. After 20 memories, reflection engine generates an insight visible in the dashboard
