"""Tests for CLI commands."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsidian_rag.cli import main
from obsidian_rag.config import Config


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Config:
    """Keep CLI tests independent of the machine's real config file and environment.

    The CLI group calls ``load_config()`` on every invocation; without this fixture the
    tests would read ``~/Library/Application Support/obsidian-notes-rag/config.toml`` and
    the ``OBSIDIAN_RAG_*`` variables, so they pass or fail depending on the host machine.
    """
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    config = Config(vault_path=str(vault_path), data_path=str(tmp_path / "data"), openai_api_key="test-key")
    monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: config)
    for name in ("OBSIDIAN_RAG_VAULT", "OBSIDIAN_RAG_DATA", "OBSIDIAN_RAG_PROVIDER", "OBSIDIAN_RAG_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return config


class TestIndexCommand:
    def test_index_with_path_filter(self):
        """Verify --path-filter option is accepted and passed through."""
        runner = CliRunner()
        with (
            patch("obsidian_rag.embedders.create_embedder") as mock_embedder,
            patch("obsidian_rag.cli.VectorStore") as mock_store,
            patch("obsidian_rag.cli.VaultIndexer") as mock_indexer,
        ):
            mock_embedder.return_value = MagicMock()
            mock_embedder.return_value.close = MagicMock()
            mock_store.return_value = MagicMock()
            mock_store.return_value.get_stats.return_value = {"count": 0}
            mock_indexer.return_value.iter_markdown_files.return_value = []

            result = runner.invoke(main, ["index", "--path-filter", "Daily Notes/"])
            assert result.exit_code == 0


class TestSimilarCommand:
    def test_similar_shows_results(self):
        """Verify similar command accepts note-path and displays results."""
        runner = CliRunner()
        with (
            patch("obsidian_rag.embedders.create_embedder") as mock_embedder,
            patch("obsidian_rag.cli.VectorStore") as mock_store,
        ):
            embedder_instance = MagicMock()
            embedder_instance.embed.return_value = [0.1] * 1536
            embedder_instance.close = MagicMock()
            mock_embedder.return_value = embedder_instance

            store_instance = MagicMock()
            store_instance.get_by_file.return_value = [
                {"content": "Note content", "metadata": {"file_path": "test.md", "heading": ""}}
            ]
            store_instance.search.return_value = [
                {
                    "content": "Similar note",
                    "metadata": {"file_path": "other.md", "heading": "Section"},
                    "distance": 0.2,
                }
            ]
            mock_store.return_value = store_instance

            result = runner.invoke(main, ["similar", "test.md"])
            assert result.exit_code == 0
            assert "other.md" in result.output


class TestContextCommand:
    def test_context_shows_note_and_similar(self):
        """Verify context command shows note content and similar notes."""
        runner = CliRunner()
        with (
            patch("obsidian_rag.embedders.create_embedder") as mock_embedder,
            patch("obsidian_rag.cli.VectorStore") as mock_store,
        ):
            embedder_instance = MagicMock()
            embedder_instance.embed.return_value = [0.1] * 1536
            embedder_instance.close = MagicMock()
            mock_embedder.return_value = embedder_instance

            store_instance = MagicMock()
            store_instance.get_by_file.return_value = [
                {"content": "Note content here", "metadata": {"file_path": "test.md", "heading": ""}}
            ]
            store_instance.search.return_value = [
                {
                    "content": "Related note",
                    "metadata": {"file_path": "related.md", "heading": "Intro"},
                    "distance": 0.3,
                }
            ]
            mock_store.return_value = store_instance

            result = runner.invoke(main, ["context", "test.md"])
            assert result.exit_code == 0
            assert "Note content here" in result.output
            assert "related.md" in result.output


class TestEmbedderResolution:
    EMBEDDING_COMMANDS = [["index"], ["search", "anything"], ["similar", "n.md"], ["context", "n.md"], ["watch"]]

    @pytest.mark.parametrize("args", EMBEDDING_COMMANDS, ids=lambda a: a[0])
    def test_missing_openai_key_exits_with_clear_error(self, isolated_config, monkeypatch, args):
        """Every embedding command shares one resolution path and one readable error."""
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: replace(isolated_config, openai_api_key=None))

        result = CliRunner().invoke(main, args)

        assert result.exit_code == 1
        assert "Error: OPENAI_API_KEY not set" in result.output
        assert "Traceback" not in result.output

    @pytest.mark.parametrize("args", EMBEDDING_COMMANDS, ids=lambda a: a[0])
    def test_unknown_provider_from_config_exits_with_clear_error(self, isolated_config, monkeypatch, args):
        monkeypatch.setattr("obsidian_rag.cli.load_config", lambda: replace(isolated_config, provider="weird"))

        result = CliRunner().invoke(main, args)

        assert result.exit_code == 1
        assert "Error: Unknown provider: weird" in result.output

    def test_cli_overrides_reach_the_resolver(self, monkeypatch):
        captured = {}

        def fake_resolve(config, **overrides):
            captured.update(overrides)
            raise SystemExit(0)

        monkeypatch.setattr("obsidian_rag.cli.resolve_embedder_settings", fake_resolve)
        CliRunner().invoke(
            main, ["--provider", "ollama", "--model", "mxbai", "--ollama-url", "http://o:1", "search", "q"]
        )

        assert captured == {"provider": "ollama", "model": "mxbai", "ollama_url": "http://o:1", "lmstudio_url": None}
