from typing import Final

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
TEMP_DIR = BASE_DIR / ".tmp"
PROMPTS_DIR = BASE_DIR / "prompts"

load_dotenv(ENV_PATH)


class GigaChatSettings(BaseSettings):
    api_key: str = ""
    scope: str = ""
    model_name: str = "GigaChat:latest"

    model_config = SettingsConfigDict(env_prefix="GIGACHAT_")


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/0"


class EmbeddingsSettings(BaseSettings):
    model_name: str = "deepvk/USER-bge-m3"
    model_kwargs: dict[str, str] = {"device": "cpu"}
    encode_kwargs: dict[str, bool] = {"normalize_embeddings": False}
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


class Settings(BaseSettings):
    gigachat: GigaChatSettings = GigaChatSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    redis: RedisSettings = RedisSettings()
    elasticsearch: ElasticsearchSettings = ElasticsearchSettings()


settings: Final[Settings] = Settings()
