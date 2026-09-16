"""Shared provider -> embedder resolution used by the CLI, the watcher and the MCP server."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .indexer import PROVIDERS, Embedder, create_embedder, unknown_provider_message

MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY not set. Configure via environment variable, config.toml [openai] api_key, or macOS Keychain."
)


class EmbedderConfigError(ValueError):
    """The embedder cannot be built from the current configuration."""


class UnknownProviderError(EmbedderConfigError):
    """The configured provider name is not one of PROVIDERS."""


class MissingApiKeyError(EmbedderConfigError):
    """The OpenAI provider is selected but no API key can be resolved."""


@dataclass(frozen=True)
class EmbedderSettings:
    """Everything needed to build an embedder, fully resolved."""

    provider: str
    model: str | None
    base_url: str | None
    api_key: str | None = field(repr=False)

    def create(self) -> Embedder:
        """Instantiate the embedder described by these settings."""
        return create_embedder(provider=self.provider, model=self.model, base_url=self.base_url, api_key=self.api_key)


def resolve_embedder_settings(
    config: Config,
    *,
    provider: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    lmstudio_url: str | None = None,
) -> EmbedderSettings:
    """Resolve provider, model, base URL and API key from ``config`` plus optional overrides.

    Overrides win over the config file; unset overrides fall back to the config
    values for the selected provider. ``config`` is never mutated.

    Raises:
        UnknownProviderError: provider is not one of PROVIDERS.
        MissingApiKeyError: OpenAI selected but no key in config, environment or Keychain.
    """
    resolved_provider = provider or config.provider
    if resolved_provider not in PROVIDERS:
        raise UnknownProviderError(unknown_provider_message(resolved_provider))

    defaults = {
        "openai": (config.openai_model, None),
        "ollama": (config.ollama_model, ollama_url or config.ollama_url),
        "lmstudio": (config.lmstudio_model, lmstudio_url or config.lmstudio_url),
    }
    default_model, base_url = defaults[resolved_provider]

    api_key: str | None = None
    if resolved_provider == "openai":
        api_key = config.get_openai_api_key()
        if not api_key:
            raise MissingApiKeyError(MISSING_API_KEY_MESSAGE)

    return EmbedderSettings(
        provider=resolved_provider, model=model or default_model, base_url=base_url, api_key=api_key
    )
