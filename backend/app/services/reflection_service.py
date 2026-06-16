from app.config import settings
from app.services import llm_router


def should_reflect(count: int) -> bool:
    """Trigger reflection every REFLECTION_THRESHOLD memories."""
    return count > 0 and count % settings.REFLECTION_THRESHOLD == 0


async def generate_insight(memories: list[str]) -> str:
    memory_block = "\n".join(f"- {m}" for m in memories[:50])
    prompt = (
        "Based on these user memories, generate one concise insight (1-2 sentences) "
        "about the user's goals, skills, or trajectory. Be specific and actionable.\n\n"
        f"Memories:\n{memory_block}"
    )
    return await llm_router.call_llm("openai", "", prompt)
