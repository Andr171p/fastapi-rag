from typing import Final

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DB_PATH = BASE_DIR / "db.sqlite3"
DB_DRIVER = "aiosqlite"
SQLALCHEMY_URL = f"sqlite+{DB_DRIVER}:///{DB_PATH}"

load_dotenv(ENV_PATH)


class GigaChatSettings(BaseSettings):
    apikey: str = ""
    scope: str = ""
    model_name: str = "GigaChat:latest"

    model_config = SettingsConfigDict(env_prefix="GIGACHAT_")


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    ttl: int = 3600  # По умолчанию 1 чс

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/0"


class EmbeddingsSettings(BaseSettings):
    normalize: bool = False
    batch_size: int = 32
    base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")


class ElasticsearchSettings(BaseSettings):
    username: str = ""
    password: str = ""
    host: str = "localhost"
    port: int = 9200

    model_config = SettingsConfigDict(env_prefix="ELASTICSEARCH_")

    @property
    def auth(self) -> tuple[str, str]:
        return self.username, self.password

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RAGSettings(BaseSettings):
    chunk_size: int = 1000
    chunk_overlap: int = 20
    system_prompt: str = ""
    max_conversation_history_length: int = 10

    model_config = SettingsConfigDict(env_prefix="RAG_")


class Settings(BaseSettings):
    gigachat: GigaChatSettings = GigaChatSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    redis: RedisSettings = RedisSettings()
    elasticsearch: ElasticsearchSettings = ElasticsearchSettings()
    rag: RAGSettings = RAGSettings()


settings: Final[Settings] = Settings()
