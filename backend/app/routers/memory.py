from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.memory import CreateMemoryRequest, MemoryOut, SearchRequest
from app.services import memory_service, retrieval_service

router = APIRouter(prefix="/memory")


@router.post("", response_model=MemoryOut)
async def create_memory(request: CreateMemoryRequest, db: AsyncSession = Depends(get_db)):
    return await memory_service.store_memory(
        db, request.user_id, request.content, request.memory_type, request.importance_score
    )


@router.get("/{user_id}", response_model=list[MemoryOut])
async def get_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    return await memory_service.get_all(db, user_id)


@router.post("/search", response_model=list[str])
async def search(request: SearchRequest):
    return await retrieval_service.search_memories(request.user_id, request.query, request.top_k)
