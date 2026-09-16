"""Tests for the shared provider -> embedder resolution."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from obsidian_rag.config import Config
from obsidian_rag.embedders import (
    EmbedderConfigError,
    EmbedderSettings,
    MissingApiKeyError,
    UnknownProviderError,
    resolve_embedder_settings,
)

OLLAMA = Config(provider="ollama", ollama_url="http://ollama:11434", ollama_model="nomic")
LMSTUDIO = Config(provider="lmstudio", lmstudio_url="http://lm:1234", lmstudio_model="nomic-v1.5")
OPENAI = Config(provider="openai", openai_model="text-embedding-3-large", openai_api_key="sk-test")


class TestResolveFromConfig:
    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (OPENAI, EmbedderSettings("openai", "text-embedding-3-large", None, "sk-test")),
            (OLLAMA, EmbedderSettings("ollama", "nomic", "http://ollama:11434", None)),
            (LMSTUDIO, EmbedderSettings("lmstudio", "nomic-v1.5", "http://lm:1234", None)),
        ],
        ids=["openai", "ollama", "lmstudio"],
    )
    def test_uses_the_selected_providers_config_values(self, config: Config, expected: EmbedderSettings):
        assert resolve_embedder_settings(config) == expected

    def test_openai_without_key_fails_with_clear_message(self):
        with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY not set"):
            resolve_embedder_settings(Config(provider="openai"))

    def test_unknown_provider_is_rejected_with_typed_error(self):
        with pytest.raises(UnknownProviderError, match="Unknown provider: weird"):
            resolve_embedder_settings(Config(provider="weird"))

    def test_config_errors_share_one_base_class(self):
        assert issubclass(MissingApiKeyError, EmbedderConfigError)
        assert issubclass(UnknownProviderError, EmbedderConfigError)
        assert issubclass(EmbedderConfigError, ValueError)


class TestOverrides:
    def test_provider_override_switches_model_and_url_defaults(self):
        config = replace(OLLAMA, provider="openai")
        settings = resolve_embedder_settings(config, provider="ollama")
        assert settings == EmbedderSettings("ollama", "nomic", "http://ollama:11434", None)

    def test_explicit_model_and_url_win_over_config(self):
        settings = resolve_embedder_settings(OLLAMA, model="mxbai", ollama_url="http://other:1")
        assert settings.model == "mxbai"
        assert settings.base_url == "http://other:1"

    def test_lmstudio_url_override_ignored_for_other_providers(self):
        assert resolve_embedder_settings(OLLAMA, lmstudio_url="http://lm:1234").base_url == "http://ollama:11434"

    def test_does_not_mutate_config(self):
        config = Config(provider="openai", openai_api_key="sk-test")
        snapshot = replace(config)
        resolve_embedder_settings(config, provider="ollama", model="x", ollama_url="http://o:1")
        assert config == snapshot


class TestEmbedderSettings:
    def test_repr_hides_the_api_key(self):
        settings = EmbedderSettings("openai", "m", None, "sk-secret")
        assert "sk-secret" not in repr(settings)
        assert settings.api_key == "sk-secret"

    def test_create_forwards_settings_to_factory(self, monkeypatch: pytest.MonkeyPatch):
        factory = MagicMock(return_value="embedder")
        monkeypatch.setattr("obsidian_rag.embedders.create_embedder", factory)
        settings = EmbedderSettings("ollama", "nomic", "http://ollama:11434", None)
        assert settings.create() == "embedder"
        factory.assert_called_once_with(provider="ollama", model="nomic", base_url="http://ollama:11434", api_key=None)
