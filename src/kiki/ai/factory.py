"""Build the active LLM provider from settings + secrets."""

from __future__ import annotations

from kiki.ai.kiki_harness import KikiHarnessProvider
from kiki.ai.ollama import OllamaProvider
from kiki.ai.openai_compatible import OpenAICompatibleProvider
from kiki.ai.provider import LLMProvider
from kiki.config.settings import Settings
from kiki.storage.secrets import OPENAI_API_KEY, SecretStore, SecretStoreError


def create_provider(settings: Settings, secrets: SecretStore) -> LLMProvider:
    if settings.ai.provider == "ollama":
        return OllamaProvider(
            settings.ai.ollama.base_url,
            num_ctx=settings.ai.ollama.num_ctx,
            think=settings.ai.ollama.think,
            suppress_thinking=settings.ai.ollama.suppress_thinking,
        )
    if settings.ai.provider == "kiki_harness":
        return KikiHarnessProvider(settings.ai.kiki_harness.base_url)
    if settings.ai.provider == "openai_compatible":
        try:
            key = secrets.get(OPENAI_API_KEY)
        except SecretStoreError:
            key = None
        return OpenAICompatibleProvider(settings.ai.openai_compatible.base_url, api_key=key)
    raise ValueError(f"unknown provider {settings.ai.provider}")


def active_model(settings: Settings) -> str:
    if settings.ai.provider == "kiki_harness":
        return settings.ai.kiki_harness.model
    if settings.ai.provider == "openai_compatible":
        return settings.ai.openai_compatible.model
    return settings.ai.ollama.model
