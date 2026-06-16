import uuid
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    model: Literal["openai", "claude"]
    message: str


class ChatResponse(BaseModel):
    response: str
    memories_stored: int


class CreateMemoryRequest(BaseModel):
    user_id: str
    content: str
    memory_type: Literal["episodic", "semantic", "reflective"] = "episodic"
    importance_score: float = 1.0


class MemoryOut(BaseModel):
    id: uuid.UUID
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    created_at: datetime

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 10
