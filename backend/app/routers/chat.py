from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.memory import ChatRequest, ChatResponse
from app.services import memory_service, retrieval_service, reflection_service
from app.services import llm_router

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 1. Retrieve relevant memories from Qdrant
    memories = await retrieval_service.search_memories(request.user_id, request.message)

    # 2. Build enriched prompt and call LLM
    system = llm_router.build_system_prompt(memories)
    response = await llm_router.call_llm(request.model, system, request.message)

    # 3. Extract new memory facts from the conversation
    extracted = await llm_router.extract_memories(request.message, response)

    # 4. Store extracted memories in PostgreSQL + Qdrant
    for mem in extracted:
        mem_type = mem.get("type", "semantic")
        value = mem.get("value", "")
        if value:
            content = f"{mem_type.capitalize()}: {value}"
            await memory_service.store_memory(db, request.user_id, content, "semantic")

    # 5. Trigger reflection if memory count crosses threshold
    count = await memory_service.get_count(db, request.user_id)
    if reflection_service.should_reflect(count):
        all_memories = await memory_service.get_all(db, request.user_id)
        contents = [m.content for m in all_memories]
        insight = await reflection_service.generate_insight(contents)
        await memory_service.store_memory(db, request.user_id, insight, "reflective", importance_score=1.5)
        await memory_service.store_insight(db, request.user_id, insight)

    return ChatResponse(response=response, memories_stored=len(extracted))
