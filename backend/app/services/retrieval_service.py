from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings

_qdrant: AsyncQdrantClient | None = None
_openai: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
    return _qdrant


async def init_collection():
    client = get_qdrant()
    existing = [c.name for c in await client.get_collections()]
    if settings.QDRANT_COLLECTION not in existing:
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )


async def embed(text: str) -> list[float]:
    resp = await _get_openai().embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


async def upsert_memory(memory_id: str, user_id: str, content: str) -> None:
    vector = await embed(content)
    await get_qdrant().upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[PointStruct(id=memory_id, vector=vector, payload={"user_id": user_id, "content": content})],
    )


async def search_memories(user_id: str, query: str, top_k: int = 10) -> list[str]:
    vector = await embed(query)
    results = await get_qdrant().search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=top_k,
    )
    return [r.payload["content"] for r in results]
