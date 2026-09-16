"""Tests for Config defaults, TOML round-trip and the shared provider defaults."""

import inspect
import tomllib
from pathlib import Path

import pytest

from obsidian_rag import defaults
from obsidian_rag.config import Config, load_config, save_config
from obsidian_rag.indexer import LMStudioEmbedder, OllamaEmbedder


@pytest.fixture(autouse=True)
def isolated_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the config file to tmp_path (conftest.py scrubs the env overrides)."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("obsidian_rag.config.get_config_path", lambda: config_path)
    return config_path


def _saved_toml(config: Config) -> dict:
    """Save ``config`` and parse the TOML that was written."""
    return tomllib.loads(save_config(config).read_text())


def _ctor_defaults(cls: type) -> dict[str, object]:
    return {name: p.default for name, p in inspect.signature(cls.__init__).parameters.items() if name != "self"}


class TestSharedDefaults:
    def test_config_fields_use_the_shared_defaults(self):
        config = Config()
        assert config.provider == defaults.DEFAULT_PROVIDER
        assert config.openai_model == defaults.DEFAULT_OPENAI_MODEL
        assert config.ollama_url == defaults.DEFAULT_OLLAMA_URL
        assert config.ollama_model == defaults.DEFAULT_OLLAMA_MODEL
        assert config.lmstudio_url == defaults.DEFAULT_LMSTUDIO_URL
        assert config.lmstudio_model == defaults.DEFAULT_LMSTUDIO_MODEL

    def test_local_embedders_default_to_the_shared_defaults(self):
        ollama = _ctor_defaults(OllamaEmbedder)
        lmstudio = _ctor_defaults(LMStudioEmbedder)
        assert ollama == {"base_url": defaults.DEFAULT_OLLAMA_URL, "model": defaults.DEFAULT_OLLAMA_MODEL}
        assert lmstudio == {"base_url": defaults.DEFAULT_LMSTUDIO_URL, "model": defaults.DEFAULT_LMSTUDIO_MODEL}


class TestSaveConfig:
    def test_default_config_writes_only_the_provider(self):
        assert _saved_toml(Config()) == {"provider": "openai"}

    def test_default_local_provider_settings_are_omitted(self):
        assert _saved_toml(Config(provider="ollama")) == {"provider": "ollama"}
        assert _saved_toml(Config(provider="lmstudio")) == {"provider": "lmstudio"}

    def test_non_default_provider_settings_are_written(self):
        data = _saved_toml(Config(provider="ollama", ollama_url="http://o:1", ollama_model="mxbai"))
        assert data["ollama"] == {"url": "http://o:1", "model": "mxbai"}

        data = _saved_toml(Config(provider="openai", openai_model="text-embedding-3-large", openai_api_key="k"))
        assert data["openai"] == {"api_key": "k", "model": "text-embedding-3-large"}


class TestLoadConfig:
    def test_missing_file_gives_defaults(self):
        assert load_config() == Config()

    def test_round_trip_preserves_settings(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        original = Config(provider="lmstudio", vault_path=str(vault), lmstudio_url="http://l:2", lmstudio_model="m")

        save_config(original)

        assert load_config() == Config(
            provider="lmstudio", vault_path=str(vault.resolve()), lmstudio_url="http://l:2", lmstudio_model="m"
        )

    def test_openai_section_without_api_key_keeps_model_and_no_key(self, isolated_config_file: Path):
        isolated_config_file.write_text('provider = "openai"\n\n[openai]\nmodel = "text-embedding-3-large"\n')

        config = load_config()

        assert config.openai_model == "text-embedding-3-large"
        assert config.openai_api_key is None
