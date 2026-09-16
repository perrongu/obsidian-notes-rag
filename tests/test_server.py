"""Tests for the MCP server tool surface, driven through an in-process MCP client."""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest
from mcp.client import Client

from obsidian_rag import server
from obsidian_rag.config import Config

EXPECTED_TOOLS = {"search_notes", "get_similar", "get_note_context", "get_stats", "reindex"}
THREAD_COUNT = 8
THREAD_TIMEOUT_S = 5.0


def _forbid_host_config() -> Config:
    raise AssertionError("tests must not load the host machine's config; set server._config explicitly")


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with fresh lazy-initialized globals and can never read the real config."""
    monkeypatch.setattr(server, "_config", None)
    monkeypatch.setattr(server, "_embedder", None)
    monkeypatch.setattr(server, "_store", None)
    monkeypatch.setattr(server, "load_config", _forbid_host_config)


def _list_tools():
    async def run():
        async with Client(server.mcp) as client:
            return await client.list_tools()

    return {tool.name: tool for tool in asyncio.run(run()).tools}


def _call_tool(name: str, arguments: dict):
    """Call a tool the way a real client does: through the request handler, not the tool manager."""

    async def run():
        async with Client(server.mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _text(result) -> str:
    return result.content[0].text


def _json_content(result) -> dict | list:
    return json.loads(_text(result))


def _hit(file_path: str, content: str, distance: float, heading: str | None = None) -> dict:
    metadata = {"file_path": file_path, "heading": heading, "type": "note"}
    return {"content": content, "metadata": metadata, "distance": distance}


class TestToolRegistration:
    def test_exposes_expected_tools(self):
        assert set(_list_tools()) == EXPECTED_TOOLS

    def test_search_notes_input_schema_survives_error_wrapper(self):
        schema = _list_tools()["search_notes"].input_schema
        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query", "limit", "note_type"}

    def test_tool_descriptions_come_from_docstrings(self):
        assert _list_tools()["reindex"].description.startswith("Re-index the Obsidian vault.")

    def test_invalid_arguments_are_reported_as_error_result(self):
        result = _call_tool("search_notes", {"query": 1})
        assert result.is_error is True


class TestGetStats:
    def test_returns_store_stats(self, monkeypatch: pytest.MonkeyPatch):
        store = MagicMock()
        store.get_stats.return_value = {"collection": "obsidian_notes", "count": 3, "data_path": "/data"}
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("get_stats", {})

        assert result.is_error is False
        assert _json_content(result) == {"collection": "obsidian_notes", "count": 3, "data_path": "/data"}

    def test_unexpected_failure_becomes_error_result_with_message(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(server, "get_store", MagicMock(side_effect=RuntimeError("db locked")))

        result = _call_tool("get_stats", {})

        assert result.is_error is True
        assert "db locked" in _text(result)


class TestSearchNotes:
    def test_returns_structured_hits(self, monkeypatch: pytest.MonkeyPatch):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        store = MagicMock()
        store.search.return_value = [_hit("a.md", "hello world", 0.2, heading="Intro")]
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


class TestGetSimilar:
    def test_unknown_note_is_an_error_result(self, monkeypatch: pytest.MonkeyPatch):
        store = MagicMock()
        store.get_by_file.return_value = []
        monkeypatch.setattr(server, "_config", Config())
        monkeypatch.setattr(server, "_embedder", MagicMock())
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("get_similar", {"note_path": "missing.md"})

        assert result.is_error is True
        assert "Note not found: missing.md" in _text(result)


class TestGetNoteContext:
    def test_returns_note_content_and_similar_notes_excluding_itself(self, monkeypatch: pytest.MonkeyPatch):
        embedder = MagicMock()
        embedder.embed.return_value = [0.5]
        store = MagicMock()
        store.get_by_file.return_value = [_hit("a.md", "part one", 0.0), _hit("a.md", "part two", 0.0)]
        store.search.return_value = [_hit("a.md", "part one", 0.0), _hit("b.md", "related", 0.3, heading="Why")]
        monkeypatch.setattr(server, "_config", Config())
        monkeypatch.setattr(server, "_embedder", embedder)
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("get_note_context", {"note_path": "a.md", "limit": 2})

        assert result.is_error is False
        assert _json_content(result) == {
            "file_path": "a.md",
            "content": "part one\n\npart two",
            "similar_notes": [{"file_path": "b.md", "heading": "Why", "preview": "related", "similarity": 0.7}],
        }


class TestReindex:
    def test_requires_vault_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(server, "_config", Config(vault_path=None))
        monkeypatch.setattr(server, "_embedder", MagicMock())
        monkeypatch.setattr(server, "_store", MagicMock())

        result = _call_tool("reindex", {})

        assert result.is_error is True
        assert "No vault path configured" in _text(result)

    def test_refuses_to_run_while_another_reindex_is_in_progress(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setattr(server, "_config", Config(vault_path=str(tmp_path)))
        monkeypatch.setattr(server, "_embedder", MagicMock())
        store = MagicMock()
        monkeypatch.setattr(server, "_store", store)

        assert server._reindex_lock.acquire(blocking=False)
        try:
            result = _call_tool("reindex", {"clear": True})
        finally:
            server._reindex_lock.release()

        assert result.is_error is True
        assert "already running" in _text(result)
        store.clear.assert_not_called()

    def test_indexes_files_and_releases_lock(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        (tmp_path / "note.md").write_text("# Title\n\nbody")
        embedder = MagicMock()
        embedder.embed_batch.return_value = [[0.1]]
        embedder.embed.return_value = [0.1]
        store = MagicMock()
        store.get_stats.return_value = {"count": 1}
        monkeypatch.setattr(server, "_config", Config(vault_path=str(tmp_path)))
        monkeypatch.setattr(server, "_embedder", embedder)
        monkeypatch.setattr(server, "_store", store)

        result = _call_tool("reindex", {"clear": True})

        assert result.is_error is False
        payload = _json_content(result)
        assert payload["files_indexed"] == 1
        assert payload["cleared"] is True
        assert payload["total_in_store"] == 1
        store.clear.assert_called_once()
        assert server._reindex_lock.acquire(blocking=False)
        server._reindex_lock.release()


def _run_in_threads(fn, count: int) -> tuple[list, list[BaseException]]:
    results: list = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(fn())
        except BaseException as e:  # noqa: BLE001 - re-raised via assertion below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=THREAD_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "threads did not finish (deadlock?)"
    return results, errors


class TestLazyInitialization:
    def test_store_is_created_once_under_concurrent_calls(self, monkeypatch: pytest.MonkeyPatch):
        """Tools run on worker threads in mcp 2.x, so lazy init must be thread-safe."""
        monkeypatch.setattr(server, "_config", Config(data_path="/data"))
        entered = threading.Event()
        release = threading.Event()

        def slow_factory(**_) -> object:
            entered.set()
            assert release.wait(THREAD_TIMEOUT_S)
            return object()

        factory = MagicMock(side_effect=slow_factory)
        monkeypatch.setattr(server, "VectorStore", factory)

        def collect() -> tuple[list, list[BaseException]]:
            return _run_in_threads(server.get_store, THREAD_COUNT)

        outcome: list = []
        driver = threading.Thread(target=lambda: outcome.append(collect()))
        driver.start()
        assert entered.wait(THREAD_TIMEOUT_S)
        release.set()
        driver.join(timeout=THREAD_TIMEOUT_S)
        assert not driver.is_alive()

        stores, errors = outcome[0]
        assert errors == []
        assert len(stores) == THREAD_COUNT
        assert factory.call_count == 1
        assert all(s is stores[0] for s in stores)

    def test_get_store_does_not_hold_the_embedder_lock(self, monkeypatch: pytest.MonkeyPatch):
        """A slow embedder init must not block tools that only need the store."""
        monkeypatch.setattr(server, "_config", Config(data_path="/data"))
        monkeypatch.setattr(server, "VectorStore", MagicMock(return_value=object()))

        with server._embedder_lock:
            results, errors = _run_in_threads(server.get_store, 1)

        assert errors == []
        assert len(results) == 1
