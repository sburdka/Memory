from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://memoryos:memoryos@localhost:5432/memoryos"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "memories"
    EMBEDDING_DIM: int = 1536  # text-embedding-3-small

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-6"
    EXTRACTION_MODEL: str = "gpt-4o-mini"

    REFLECTION_THRESHOLD: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
