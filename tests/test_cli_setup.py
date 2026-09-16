"""Tests for the interactive ``setup`` wizard and the shared watcher-service installer.

Every path the wizard touches (config file, data dir, LaunchAgents, wrapper script, logs)
is redirected to ``tmp_path`` and ``launchctl`` is replaced by a recording fake, so these
tests never read or write the developer's real home directory.
"""

import plistlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result

from obsidian_rag.cli import PLIST_NAME, WRAPPER_SCRIPT_NAME, main
from obsidian_rag.config import PROVIDER_URL_ENV, Config, load_config
from obsidian_rag.defaults import DEFAULT_LMSTUDIO_MODEL
from obsidian_rag.embedders import EmbedderConfigError, resolve_embedder_settings

OK = CompletedProcess([], 0, "", "")


@dataclass(frozen=True)
class WizardEnv:
    config_path: Path
    data_dir: Path
    vault: Path
    plist_path: Path
    wrapper_path: Path
    log_dir: Path
    launchctl_calls: list[tuple[str, Path]]


@pytest.fixture(autouse=True)
def wizard_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WizardEnv:
    config_dir, data_dir, vault = tmp_path / "config", tmp_path / "data", tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# note")

    # config module: get_config_path (also the name imported by cli), save_config and load_config
    monkeypatch.setattr("obsidian_rag.config.get_config_dir", lambda: config_dir)
    monkeypatch.setattr("obsidian_rag.config.get_data_dir", lambda: data_dir)  # Config.get_data_path
    monkeypatch.setattr("obsidian_rag.cli.get_data_dir", lambda: data_dir)  # the wizard's default prompt
    # service paths are computed from Path.home() at import time
    monkeypatch.setattr("obsidian_rag.cli.LAUNCH_AGENTS_DIR", tmp_path / "LaunchAgents")
    monkeypatch.setattr("obsidian_rag.cli.LOG_DIR", tmp_path / "Logs")
    monkeypatch.setattr("obsidian_rag.cli.WRAPPER_SCRIPT_DIR", tmp_path / "bin")
    # network probes, imported by name into cli
    monkeypatch.setattr("obsidian_rag.cli.is_ollama_running", lambda url: False)
    monkeypatch.setattr("obsidian_rag.cli.is_lmstudio_running", lambda url: False)
    monkeypatch.setattr("obsidian_rag.cli.get_ollama_models", lambda url: [])
    monkeypatch.setattr("obsidian_rag.cli.get_lmstudio_models", lambda url: [])

    calls: list[tuple[str, Path]] = []

    def fake_launchctl(action: str, plist_path: Path) -> CompletedProcess[str]:
        calls.append((action, plist_path))
        return OK

    monkeypatch.setattr("obsidian_rag.cli._launchctl", fake_launchctl)
    # Belt and braces: nothing in cli may spawn a real process during these tests
    monkeypatch.setattr("obsidian_rag.cli.subprocess", SimpleNamespace(run=MagicMock(side_effect=AssertionError)))
    monkeypatch.setattr(sys, "platform", "darwin")

    return WizardEnv(
        config_path=config_dir / "config.toml",
        data_dir=data_dir,
        vault=vault,
        plist_path=tmp_path / "LaunchAgents" / PLIST_NAME,
        wrapper_path=tmp_path / "bin" / WRAPPER_SCRIPT_NAME,
        log_dir=tmp_path / "Logs",
        launchctl_calls=calls,
    )


def run_setup(text: str) -> Result:
    result = CliRunner().invoke(main, ["setup"], input=text)
    assert "Aborted" not in result.output, f"the input ran out before the wizard finished:\n{result.output}"
    return result


def _config(env: WizardEnv) -> dict:
    return tomllib.loads(env.config_path.read_text())


def _plist(env: WizardEnv) -> dict:
    return plistlib.loads(env.plist_path.read_bytes())


class TestProviderSelection:
    def test_openai_env_key_not_saved(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "Found OPENAI_API_KEY" in result.output
        config = _config(wizard_env)
        assert config["provider"] == "openai"
        assert "openai" not in config
        assert config["vault_path"] == str(wizard_env.vault.resolve())
        assert "data_path" not in config  # the default is resolved at run time, never pinned in the file
        assert not wizard_env.plist_path.exists()

    def test_openai_keychain_key_is_detected(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("obsidian_rag.config._get_keychain_value", lambda *_: "sk-keychain")

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "Found OPENAI_API_KEY" in result.output
        assert "openai" not in _config(wizard_env)

    def test_openai_prompts_for_key_and_saves_it(self, wizard_env):
        result = run_setup(f"\nsk-typed\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "Choice (1, 2, 3) [1]:" in result.output
        assert "sk-typed" not in result.output
        assert _config(wizard_env)["openai"] == {"api_key": "sk-typed"}

    def test_ollama_server_down_uses_defaults(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        list_models = MagicMock(return_value=[])
        monkeypatch.setattr("obsidian_rag.cli.get_ollama_models", list_models)

        result = run_setup(f"2\n\n\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "not detected" in result.output
        config = _config(wizard_env)
        assert config["provider"] == "ollama"
        assert "ollama" not in config
        list_models.assert_not_called()

    @pytest.mark.parametrize(
        ("answers", "expected_model"),
        [("1\n", "mxbai-embed-large"), ("3\ncustom-model\n", "custom-model")],
        ids=["listed", "other"],
    )
    def test_ollama_lists_models_and_stores_choice(self, wizard_env, monkeypatch, answers, expected_model):
        monkeypatch.setattr("obsidian_rag.cli.is_ollama_running", lambda url: True)
        monkeypatch.setattr("obsidian_rag.cli.get_ollama_models", lambda url: ["mxbai-embed-large", "nomic-embed-text"])

        result = run_setup(f"2\nhttp://ollama.local:11434\n{answers}{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "found 2" in result.output
        assert _config(wizard_env)["ollama"] == {"url": "http://ollama.local:11434", "model": expected_model}

    def test_lmstudio_server_down_offers_default_model(self, wizard_env):
        result = run_setup(f"3\n\n\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert f"[{DEFAULT_LMSTUDIO_MODEL}]" in result.output
        config = _config(wizard_env)
        assert config["provider"] == "lmstudio"
        assert "lmstudio" not in config

    def test_provider_menu_covers_every_provider(self):
        from obsidian_rag.cli import _PROVIDER_PROMPTS, PROVIDER_LABELS
        from obsidian_rag.indexer import PROVIDERS

        assert set(PROVIDER_LABELS) == set(PROVIDERS) == set(_PROVIDER_PROMPTS)
        assert set(PROVIDER_URL_ENV) == set(PROVIDERS) - {"openai"}


class TestVaultAndConfigFile:
    def test_vault_path_retry_then_success(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        result = run_setup(f"1\nn\n/nope\n\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "Directory not found: /nope" in result.output
        assert wizard_env.config_path.exists()

    def test_vault_path_cancel(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        result = run_setup("1\nn\n/nope\nn\n")

        assert result.exit_code == 0, result.output
        assert "Setup cancelled." in result.output
        assert not wizard_env.config_path.exists()

    def test_vault_path_must_be_a_directory(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        note = wizard_env.vault / "note.md"

        result = run_setup(f"1\nn\n{note}\n\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert f"Directory not found: {note}" in result.output
        assert _config(wizard_env)["vault_path"] == str(wizard_env.vault.resolve())

    def test_vault_check_does_not_walk_the_vault(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        """An iCloud vault with thousands of evicted files must not be walked just to print a count."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setattr(Path, "rglob", MagicMock(side_effect=AssertionError("the wizard walked the vault")))

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "✓ Vault found" in result.output
        assert "markdown files" not in result.output

    def test_relative_vault_path_is_canonical_for_the_whole_run(self, wizard_env, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.chdir(tmp_path)

        result = run_setup("1\nn\nvault\n\nn\n\n")

        assert result.exit_code == 0, result.output
        canonical = str(wizard_env.vault.resolve())
        assert _config(wizard_env)["vault_path"] == canonical
        assert _plist(wizard_env)["EnvironmentVariables"]["OBSIDIAN_RAG_VAULT"] == canonical

    def test_existing_config_overwrite_declined(self, wizard_env):
        wizard_env.config_path.parent.mkdir(parents=True)
        wizard_env.config_path.write_text('provider = "ollama"\n')

        result = run_setup("n\n")

        assert result.exit_code == 0, result.output
        assert "Setup cancelled." in result.output
        assert wizard_env.config_path.read_text() == 'provider = "ollama"\n'


class TestEmbedderResolution:
    @pytest.fixture
    def resolve_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[Config]:
        """Record every embedder resolution the wizard performs; each one may cost a Keychain read."""
        calls: list[Config] = []

        def counting_resolve(config: Config, **overrides):
            calls.append(config)
            return resolve_embedder_settings(config, **overrides)

        monkeypatch.setattr("obsidian_rag.cli.resolve_embedder_settings", counting_resolve)
        return calls

    @pytest.mark.parametrize("run_indexing", ["n", "y"], ids=["indexing-refused", "indexing-accepted"])
    def test_wizard_resolves_the_embedder_once(self, wizard_env, monkeypatch, resolve_calls, run_indexing):
        keychain_reads: list[str] = []

        def fake_keychain(service: str, *_):
            keychain_reads.append(service)
            return "sk-keychain"

        monkeypatch.setattr("obsidian_rag.config._get_keychain_value", fake_keychain)
        monkeypatch.setattr("obsidian_rag.embedders.create_embedder", lambda **_: MagicMock())
        monkeypatch.setattr("obsidian_rag.cli.VectorStore", MagicMock())
        indexer = MagicMock()
        indexer.return_value.iter_markdown_files.return_value = []
        monkeypatch.setattr("obsidian_rag.cli.VaultIndexer", indexer)

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\n{run_indexing}\n\n")

        assert result.exit_code == 0, result.output
        assert wizard_env.plist_path.exists()
        assert len(resolve_calls) == 1
        assert len(keychain_reads) == 1  # _prompt_openai's detection; the resolution reuses that key

    def test_embedder_config_error_is_reported_in_one_line(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        def failing_resolve(config: Config, **_):
            raise EmbedderConfigError("bad embedder configuration")

        monkeypatch.setattr("obsidian_rag.cli.resolve_embedder_settings", failing_resolve)

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\n")

        assert result.exit_code == 0, result.output
        assert "Configuration saved" in result.output
        assert "✗ bad embedder configuration" in result.output
        assert "Traceback" not in result.output
        assert "Run initial indexing now?" not in result.output  # both optional steps are skipped, not attempted
        assert not wizard_env.plist_path.exists()
        assert "Setup complete!" in result.output  # the saved configuration is still usable once fixed


class TestServiceInstall:
    def test_install_service_creates_log_dir_and_plist(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        result = run_setup(f"1\nn\n{wizard_env.vault}\n{wizard_env.data_dir}\nn\n\n")

        assert result.exit_code == 0, result.output
        assert wizard_env.log_dir.is_dir()
        assert wizard_env.data_dir.is_dir()  # launchd chdirs into WorkingDirectory before the job starts
        assert wizard_env.wrapper_path.exists() and wizard_env.wrapper_path.stat().st_mode & 0o111
        assert wizard_env.launchctl_calls == [("load", wizard_env.plist_path)]
        plist = _plist(wizard_env)
        assert plist["EnvironmentVariables"] == {
            "OBSIDIAN_RAG_VAULT": str(wizard_env.vault.resolve()),
            "OBSIDIAN_RAG_DATA": str(wizard_env.data_dir),
            "OBSIDIAN_RAG_PROVIDER": "openai",
        }
        assert plist["StandardOutPath"] == str(wizard_env.log_dir / "watcher.log")
        assert plist["WorkingDirectory"] == str(wizard_env.data_dir)
        assert f"Logs: {wizard_env.log_dir}/watcher.log" in result.output
        assert "data_path" not in _config(wizard_env)  # typing the default is the same as accepting it

    def test_accepting_a_relative_default_does_not_pin_it(self, wizard_env, monkeypatch, tmp_path: Path):
        """platformdirs honours a relative XDG_DATA_HOME verbatim; the comparison must normalize both sides."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setattr("obsidian_rag.cli.get_data_dir", lambda: Path("rel-data"))
        monkeypatch.chdir(tmp_path)

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert "data_path" not in _config(wizard_env)

    def test_custom_data_path_is_saved_and_pinned_in_plist(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        custom = wizard_env.data_dir.parent / "custom-index"

        result = run_setup(f"1\nn\n{wizard_env.vault}\n{custom}\nn\n\n")

        assert result.exit_code == 0, result.output
        assert _config(wizard_env)["data_path"] == str(custom)
        plist = _plist(wizard_env)
        assert plist["EnvironmentVariables"]["OBSIDIAN_RAG_DATA"] == str(custom)
        assert plist["WorkingDirectory"] == str(custom)
        assert custom.is_dir()

    @pytest.mark.parametrize(
        ("choice", "provider", "url"),
        [("2", "ollama", "http://ollama.local:11434"), ("3", "lmstudio", "http://lm.local:1234")],
        ids=["ollama", "lmstudio"],
    )
    def test_wizard_pins_the_chosen_provider_url_under_its_own_key(self, wizard_env, choice, provider, url):
        result = run_setup(f"{choice}\n{url}\n\n{wizard_env.vault}\n\nn\n\n")

        assert result.exit_code == 0, result.output
        assert _plist(wizard_env)["EnvironmentVariables"] == {
            "OBSIDIAN_RAG_VAULT": str(wizard_env.vault.resolve()),
            "OBSIDIAN_RAG_DATA": str(wizard_env.data_dir),
            "OBSIDIAN_RAG_PROVIDER": provider,
            PROVIDER_URL_ENV[provider]: url,
        }

    def test_install_service_reloads_existing_plist_and_reports_launchctl_failure(
        self, wizard_env, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        wizard_env.plist_path.parent.mkdir(parents=True)
        wizard_env.plist_path.write_text("stale")

        def failing_launchctl(action: str, plist_path: Path) -> CompletedProcess[str]:
            wizard_env.launchctl_calls.append((action, plist_path))
            return CompletedProcess([], 1, "", "boom") if action == "load" else OK

        monkeypatch.setattr("obsidian_rag.cli._launchctl", failing_launchctl)

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\n\n")

        assert result.exit_code == 0, result.output
        assert wizard_env.launchctl_calls == [("unload", wizard_env.plist_path), ("load", wizard_env.plist_path)]
        assert "Error starting service: boom" in result.output
        assert "Setup complete!" in result.output

    def test_non_darwin_skips_service_prompt(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setattr(sys, "platform", "linux")

        result = run_setup(f"1\nn\n{wizard_env.vault}\n\nn\n")

        assert result.exit_code == 0, result.output
        assert "Background service not yet supported" in result.output
        assert not wizard_env.plist_path.exists()

    def test_install_service_command_uses_shared_helper(self, wizard_env, monkeypatch: pytest.MonkeyPatch):
        config = Config(vault_path=str(wizard_env.vault), data_path=str(wizard_env.data_dir), openai_api_key="k")
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: config)

        result = CliRunner().invoke(main, ["--model", "pinned", "install-service"])

        assert result.exit_code == 0, result.output
        assert f"Created: {wizard_env.wrapper_path}" in result.output
        assert f"Created: {wizard_env.plist_path}" in result.output
        assert "Service installed and started." in result.output
        assert wizard_env.log_dir.is_dir()
        plist = _plist(wizard_env)
        assert plist["EnvironmentVariables"]["OBSIDIAN_RAG_MODEL"] == "pinned"
        assert plist["WorkingDirectory"] == str(wizard_env.data_dir)
        assert wizard_env.launchctl_calls == [("load", wizard_env.plist_path)]

    @pytest.mark.parametrize(
        ("home", "given", "expected"),
        [(None, "rel/dir", "rel/dir"), ("home", "~/idx", "home/idx")],
        ids=["relative", "tilde"],
    )
    def test_install_service_writes_absolute_paths_for_data_option(
        self, wizard_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home, given, expected
    ):
        """launchd resolves a relative WorkingDirectory from /, so the service would never start."""
        config = Config(vault_path=str(wizard_env.vault), openai_api_key="k")
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: config)
        monkeypatch.chdir(tmp_path)
        if home:
            monkeypatch.setenv("HOME", str(tmp_path / home))

        result = CliRunner().invoke(main, ["--data", given, "install-service"])

        assert result.exit_code == 0, result.output
        plist = _plist(wizard_env)
        assert plist["EnvironmentVariables"]["OBSIDIAN_RAG_DATA"] == str(tmp_path / expected)
        assert plist["WorkingDirectory"] == str(tmp_path / expected)
        assert (tmp_path / expected).is_dir()

    def test_install_service_writes_canonical_path_for_vault_option(self, wizard_env, monkeypatch, tmp_path: Path):
        """--vault gets the same realpath treatment as config.toml: watchdog needs the canonical root."""
        config = Config(data_path=str(wizard_env.data_dir), openai_api_key="k")
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: config)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "link").symlink_to(wizard_env.vault)

        result = CliRunner().invoke(main, ["--vault", "link", "install-service"])

        assert result.exit_code == 0, result.output
        assert _plist(wizard_env)["EnvironmentVariables"]["OBSIDIAN_RAG_VAULT"] == str(wizard_env.vault.resolve())

    def test_install_service_command_pins_lmstudio_url_under_its_own_key(self, wizard_env, monkeypatch):
        config = Config(vault_path=str(wizard_env.vault), data_path=str(wizard_env.data_dir))
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: config)

        result = CliRunner().invoke(
            main, ["--provider", "lmstudio", "--lmstudio-url", "http://lm:1234", "install-service"]
        )

        assert result.exit_code == 0, result.output
        assert wizard_env.data_dir.is_dir()
        env_vars = _plist(wizard_env)["EnvironmentVariables"]
        assert env_vars == {
            "OBSIDIAN_RAG_VAULT": str(wizard_env.vault),
            "OBSIDIAN_RAG_DATA": str(wizard_env.data_dir),
            "OBSIDIAN_RAG_PROVIDER": "lmstudio",
            "OBSIDIAN_RAG_LMSTUDIO_URL": "http://lm:1234",
        }
        # The watcher starts from these variables: load_config must read the pinned URL back
        for name, value in env_vars.items():
            monkeypatch.setenv(name, value)
        watcher_config = load_config()
        assert (watcher_config.provider, watcher_config.lmstudio_url) == ("lmstudio", "http://lm:1234")
