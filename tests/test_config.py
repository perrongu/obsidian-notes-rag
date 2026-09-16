"""Tests for Config defaults, TOML round-trip and the shared provider defaults."""

import inspect
import tomllib
from pathlib import Path

import pytest

from obsidian_rag import defaults
from obsidian_rag.config import Config, absolute_path, load_config, resolve_path_case, save_config
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

    def test_data_path_is_written_absolute(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _saved_toml(Config(data_path="~/idx"))["data_path"] == str(tmp_path / "idx")

    def test_default_local_provider_settings_are_omitted(self):
        assert _saved_toml(Config(provider="ollama")) == {"provider": "ollama"}
        assert _saved_toml(Config(provider="lmstudio")) == {"provider": "lmstudio"}

    def test_non_default_provider_settings_are_written(self):
        data = _saved_toml(Config(provider="ollama", ollama_url="http://o:1", ollama_model="mxbai"))
        assert data["ollama"] == {"url": "http://o:1", "model": "mxbai"}

        data = _saved_toml(Config(provider="openai", openai_model="text-embedding-3-large", openai_api_key="k"))
        assert data["openai"] == {"api_key": "k", "model": "text-embedding-3-large"}


class TestAbsolutePath:
    def test_tilde_is_expanded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert absolute_path("~/idx") == str(tmp_path / "idx")

    def test_relative_path_is_anchored_to_the_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        assert absolute_path("rel/idx") == str(tmp_path / "rel" / "idx")

    def test_absolute_path_is_unchanged_even_when_missing(self, tmp_path: Path):
        missing = tmp_path / "nope" / "idx"
        assert absolute_path(str(missing)) == str(missing)


class TestResolvePathCase:
    def test_missing_path_is_still_made_absolute(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """A vault that is not mounted yet must not leave a tilde or relative path for launchd."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        assert resolve_path_case("~/missing-vault") == str(tmp_path / "missing-vault")
        assert resolve_path_case("rel/missing-vault") == str(tmp_path / "rel" / "missing-vault")

    def test_existing_symlink_resolves_to_its_target(self, tmp_path: Path):
        target = tmp_path / "Vault"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        assert resolve_path_case(str(link)) == str(target.resolve())


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

    def test_env_data_path_is_made_absolute(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OBSIDIAN_RAG_DATA", "~/idx")

        assert load_config().data_path == str(tmp_path / "idx")

    def test_env_vault_path_is_made_absolute_even_when_missing(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "~/missing-vault")

        assert load_config().vault_path == str(tmp_path / "missing-vault")

    def test_toml_data_path_is_made_absolute(self, isolated_config_file: Path, monkeypatch, tmp_path: Path):
        isolated_config_file.write_text('provider = "openai"\ndata_path = "rel/idx"\n')
        monkeypatch.chdir(tmp_path)

        assert load_config().data_path == str(tmp_path / "rel" / "idx")

    def test_openai_section_without_api_key_keeps_model_and_no_key(self, isolated_config_file: Path):
        isolated_config_file.write_text('provider = "openai"\n\n[openai]\nmodel = "text-embedding-3-large"\n')

        config = load_config()

        assert config.openai_model == "text-embedding-3-large"
        assert config.openai_api_key is None
