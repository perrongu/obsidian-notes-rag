"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def no_host_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never resolve an OpenAI key from the developer's environment or macOS Keychain.

    Tests that need a key set it explicitly on their Config; everything else must
    behave identically on a developer Mac and on the Linux CI runner.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("obsidian_rag.config._get_keychain_value", lambda *_: None)
