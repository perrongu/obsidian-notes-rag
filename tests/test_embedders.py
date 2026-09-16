"""Tests for the shared provider -> embedder resolution."""

from unittest.mock import MagicMock

import pytest

from obsidian_rag.config import Config
from obsidian_rag.embedders import EmbedderSettings, MissingApiKeyError, resolve_embedder_settings


@pytest.fixture(autouse=True)
def no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the macOS Keychain from tests; the key comes from Config only."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("obsidian_rag.config._get_keychain_value", lambda *_: None)


class TestResolveFromConfig:
    def test_openai_uses_config_model_and_key(self):
        config = Config(provider="openai", openai_model="text-embedding-3-large", openai_api_key="sk-test")
        settings = resolve_embedder_settings(config)
        assert settings == EmbedderSettings(
            provider="openai", model="text-embedding-3-large", base_url=None, api_key="sk-test"
        )

    def test_openai_without_key_fails_with_clear_message(self):
        with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY not set"):
            resolve_embedder_settings(Config(provider="openai"))

    def test_ollama_uses_config_url_and_model(self):
        config = Config(provider="ollama", ollama_url="http://ollama:11434", ollama_model="nomic")
        settings = resolve_embedder_settings(config)
        assert settings == EmbedderSettings(
            provider="ollama", model="nomic", base_url="http://ollama:11434", api_key=None
        )

    def test_lmstudio_uses_config_url_and_model(self):
        config = Config(provider="lmstudio", lmstudio_url="http://lm:1234", lmstudio_model="nomic-v1.5")
        settings = resolve_embedder_settings(config)
        assert settings == EmbedderSettings(
            provider="lmstudio", model="nomic-v1.5", base_url="http://lm:1234", api_key=None
        )

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider: weird"):
            resolve_embedder_settings(Config(provider="weird"))


class TestOverrides:
    def test_provider_override_switches_model_and_url_defaults(self):
        config = Config(provider="openai", ollama_url="http://ollama:11434", ollama_model="nomic")
        settings = resolve_embedder_settings(config, provider="ollama")
        assert settings == EmbedderSettings(
            provider="ollama", model="nomic", base_url="http://ollama:11434", api_key=None
        )

    def test_explicit_model_and_url_win_over_config(self):
        config = Config(provider="ollama", ollama_url="http://ollama:11434", ollama_model="nomic")
        settings = resolve_embedder_settings(config, model="mxbai", ollama_url="http://other:1")
        assert settings.model == "mxbai"
        assert settings.base_url == "http://other:1"

    def test_lmstudio_url_override_ignored_for_other_providers(self):
        config = Config(provider="ollama", ollama_url="http://ollama:11434")
        settings = resolve_embedder_settings(config, lmstudio_url="http://lm:1234")
        assert settings.base_url == "http://ollama:11434"

    def test_does_not_mutate_config(self):
        config = Config(provider="openai", openai_api_key="sk-test")
        resolve_embedder_settings(config, provider="ollama", model="x")
        assert config.provider == "openai"
        assert config.ollama_model == "nomic-embed-text"


class TestCreate:
    def test_create_forwards_settings_to_factory(self, monkeypatch: pytest.MonkeyPatch):
        factory = MagicMock(return_value="embedder")
        monkeypatch.setattr("obsidian_rag.embedders.create_embedder", factory)
        settings = EmbedderSettings(provider="ollama", model="nomic", base_url="http://ollama:11434", api_key=None)
        assert settings.create() == "embedder"
        factory.assert_called_once_with(provider="ollama", model="nomic", base_url="http://ollama:11434", api_key=None)
