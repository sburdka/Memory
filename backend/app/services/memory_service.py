import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.memory import Memory, Insight
from app.services import retrieval_service


async def store_memory(
    db: AsyncSession,
    user_id: str,
    content: str,
    memory_type: str,
    importance_score: float = 1.0,
) -> Memory:
    memory = Memory(
        id=uuid.uuid4(),
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance_score=importance_score,
    )
    db.add(memory)
    await db.commit()
    await db.refresh(memory)
    await retrieval_service.upsert_memory(str(memory.id), user_id, content)
    return memory


async def get_all(db: AsyncSession, user_id: str) -> list[Memory]:
    result = await db.execute(select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc()))
    return list(result.scalars().all())


async def get_count(db: AsyncSession, user_id: str) -> int:
    result = await db.execute(select(func.count()).where(Memory.user_id == user_id))
    return result.scalar_one()


async def store_insight(db: AsyncSession, user_id: str, content: str, confidence_score: float = 0.8) -> Insight:
    insight = Insight(id=uuid.uuid4(), user_id=user_id, content=content, confidence_score=confidence_score)
    db.add(insight)
    await db.commit()
    return insight


async def get_insights(db: AsyncSession, user_id: str) -> list[Insight]:
    result = await db.execute(select(Insight).where(Insight.user_id == user_id).order_by(Insight.created_at.desc()))
    return list(result.scalars().all())
