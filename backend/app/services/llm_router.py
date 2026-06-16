import json
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from app.config import settings

_openai: AsyncOpenAI | None = None
_anthropic: AsyncAnthropic | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai


def _get_anthropic() -> AsyncAnthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic


def build_system_prompt(memories: list[str]) -> str:
    if not memories:
        return "You are a helpful assistant."
    memory_block = "\n".join(f"- {m}" for m in memories[:10])
    return f"Relevant User Memory:\n{memory_block}\n\nYou are a helpful assistant. Use the memory context above to personalize your responses."


async def call_llm(model: str, system: str, user_message: str) -> str:
    if model == "openai":
        resp = await _get_openai().chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_message}],
        )
        return resp.choices[0].message.content or ""

    if model == "claude":
        resp = await _get_anthropic().messages.create(
            model=settings.ANTHROPIC_CHAT_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    raise ValueError(f"Unknown model: {model}")


async def extract_memories(user_message: str, ai_response: str) -> list[dict]:
    """Parse conversation for structured memory facts using a cheap model."""
    prompt = (
        "Extract factual memories from this conversation. "
        "Return JSON: {\"memories\": [{\"type\": \"skill\", \"value\": \"Python\"}]}. "
        "Allowed types: skill, goal, preference, interest, experience. "
        "Return {\"memories\": []} if nothing to extract. No other keys.\n\n"
        f"User: {user_message[:1500]}\nAssistant: {ai_response[:1500]}"
    )
    resp = await _get_openai().chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        return data.get("memories", [])
    except Exception:
        return []
