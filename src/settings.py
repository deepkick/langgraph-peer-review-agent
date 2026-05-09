"""アプリケーション共通設定モジュール。"""

import os
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API キー
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str

    # LangSmith（新形式 LANGSMITH_*）
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "langgraph-peer-review-agent"

    # アプリケーション設定
    openai_smart_model: str = "gpt-5.4"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_smart_model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    default_reflection_db_path: str = "tmp/reflection_db.json"

    def __init__(self, **values):
        super().__init__(**values)
        self._set_env_variables()

    def _set_env_variables(self):
        """大文字キーを環境変数に登録する。
        LangSmith は新旧両方の env 名で動くため、互換性のため両方設定する。
        """
        for key in self.__annotations__.keys():
            if key.isupper():
                os.environ[key] = getattr(self, key)

        # LangSmith 旧形式へのエイリアスも設定
        os.environ["LANGCHAIN_TRACING_V2"] = self.LANGSMITH_TRACING
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT


def get_llm(
    provider: Literal["openai", "anthropic"] = "anthropic",
    settings: Settings | None = None,
) -> BaseChatModel:
    if settings is None:
        settings = Settings()

    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_smart_model,
            temperature=settings.temperature,
            timeout=120,
            max_retries=2,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=settings.anthropic_smart_model,
            temperature=settings.temperature,
            timeout=120,
            max_retries=2,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
