"""Tests for the MCP server tool surface."""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

from obsidian_rag import server
from obsidian_rag.config import Config

EXPECTED_TOOLS = {"search_notes", "get_similar", "get_note_context", "get_stats", "reindex"}


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with fresh lazy-initialized globals."""
    monkeypatch.setattr(server, "_config", None)
    monkeypatch.setattr(server, "_embedder", None)
    monkeypatch.setattr(server, "_store", None)


def _list_tools():
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


def _call_tool(name: str, arguments: dict):
    return asyncio.run(server.mcp.call_tool(name, arguments))


def _json_content(result) -> dict | list:
    return json.loads(result.content[0].text)


class TestToolRegistration:
    def test_exposes_expected_tools(self):
        assert set(_list_tools()) == EXPECTED_TOOLS

    def test_search_notes_input_schema(self):
        schema = _list_tools()["search_notes"].input_schema
        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query", "limit", "note_type"}

    def test_tool_descriptions_come_from_docstrings(self):
        assert _list_tools()["reindex"].description.startswith("Re-index the Obsidian vault.")


class TestGetStats:
    def test_returns_store_stats(self, monkeypatch: pytest.MonkeyPatch):
        store = MagicMock()
        store.get_stats.return_value = {"collection": "obsidian_notes", "count": 3, "data_path": "/tmp/x"}
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("get_stats", {})

        assert result.is_error is False
        assert _json_content(result) == {"collection": "obsidian_notes", "count": 3, "data_path": "/tmp/x"}

    def test_reports_failure_as_payload_instead_of_raising(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(server, "get_store", MagicMock(side_effect=RuntimeError("db locked")))

        result = _call_tool("get_stats", {})

        assert result.is_error is False
        assert _json_content(result) == {"error": "db locked"}


class TestSearchNotes:
    def test_returns_structured_hits(self, monkeypatch: pytest.MonkeyPatch):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        store = MagicMock()
        hit_metadata = {"file_path": "a.md", "heading": "Intro", "type": "note"}
        store.search.return_value = [{"content": "hello world", "metadata": hit_metadata, "distance": 0.2}]
        monkeypatch.setattr(server, "_config", Config())
        monkeypatch.setattr(server, "_embedder", embedder)
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("search_notes", {"query": "hello", "limit": 5, "note_type": "note"})

        assert result.is_error is False
        expected_hit = {
            "file_path": "a.md",
            "heading": "Intro",
            "content": "hello world",
            "similarity": 0.8,
            "type": "note",
        }
        assert result.structured_content == {"result": [expected_hit]}
        embedder.embed.assert_called_once_with("hello", task_type="search_query")
        store.search.assert_called_once_with([0.1, 0.2, 0.3], limit=5, where={"type": "note"})


class TestReindex:
    def test_requires_vault_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(server, "_config", Config(vault_path=None))
        monkeypatch.setattr(server, "_embedder", MagicMock())
        monkeypatch.setattr(server, "_store", MagicMock())

        result = _call_tool("reindex", {})

        assert result.is_error is False
        assert _json_content(result)["error"].startswith("No vault path configured")


class TestLazyInitialization:
    def test_store_is_created_once_under_concurrent_calls(self, monkeypatch: pytest.MonkeyPatch):
        """Tools run on worker threads in mcp 2.x, so lazy init must be thread-safe."""
        monkeypatch.setattr(server, "_config", Config(data_path="/tmp/does-not-matter"))
        factory = MagicMock(side_effect=lambda **_: (threading.Event().wait(0.05), object())[1])
        monkeypatch.setattr(server, "VectorStore", factory)

        stores: list[object] = []
        threads = [threading.Thread(target=lambda: stores.append(server.get_store())) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert factory.call_count == 1
        assert len({id(s) for s in stores}) == 1
