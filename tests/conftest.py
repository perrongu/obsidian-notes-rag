"""Shared test fixtures."""

import pytest

# Every OBSIDIAN_RAG_* override read by load_config(); scrubbed for the whole suite.
CONFIG_ENV_VARS = (
    "OBSIDIAN_RAG_PROVIDER",
    "OBSIDIAN_RAG_VAULT",
    "OBSIDIAN_RAG_DATA",
    "OBSIDIAN_RAG_OLLAMA_URL",
    "OBSIDIAN_RAG_LMSTUDIO_URL",
    "OBSIDIAN_RAG_MODEL",
)


@pytest.fixture(autouse=True)
def no_host_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config tests must not depend on OBSIDIAN_RAG_* variables set in the developer's shell."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_host_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never resolve an OpenAI key from the developer's environment or macOS Keychain.

    Tests that need a key set it explicitly on their Config; everything else must
    behave identically on a developer Mac and on the Linux CI runner.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("obsidian_rag.config._get_keychain_value", lambda *_: None)
