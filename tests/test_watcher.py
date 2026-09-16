"""Tests for watcher retry behavior: no hot loop, bounded retries, iCloud dataless handling."""

import errno
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.icloud import FileNotMaterializedError
from obsidian_rag.retry_queue import RetryQueue, process_due
from obsidian_rag.watcher import NoteEventHandler


@pytest.fixture
def note(tmp_path: Path) -> Path:
    path = tmp_path / "note.md"
    path.write_text("# hello\n\nworld\n", encoding="utf-8")
    return path


@pytest.fixture
def handler(tmp_path: Path) -> NoteEventHandler:
    return NoteEventHandler(
        vault_path=tmp_path,
        embedder=MagicMock(),
        store=MagicMock(),
        retry_queue=RetryQueue(max_retries=3, base_delay=60.0),
    )


class TestIndexFileEventPath:
    def test_transient_error_is_queued_once(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock(side_effect=RuntimeError("boom"))
        handler._index_file(note)
        handler._index_file(note)
        assert len(handler.retry_queue) == 1
        assert handler.retry_queue.attempts(note) == 0

    def test_success_clears_pending_retry(self, handler: NoteEventHandler, note: Path):
        handler.retry_queue.add(note, now=0.0)
        handler.indexer.index_file = MagicMock(return_value=[])
        with patch("obsidian_rag.watcher.icloud.is_dataless", return_value=False):
            handler._index_file(note)
        assert handler.retry_queue.is_empty()

    def test_permanent_error_is_not_queued(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock(side_effect=sqlite3.OperationalError("dimension mismatch"))
        handler._index_file(note)
        assert handler.retry_queue.is_empty()

    def test_dataless_file_is_queued_and_download_requested(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock()
        with (
            patch("obsidian_rag.watcher.icloud.is_dataless", return_value=True),
            patch("obsidian_rag.watcher.icloud.request_download") as download,
        ):
            handler._index_file(note)
        download.assert_called_once_with(note)
        handler.indexer.index_file.assert_not_called()
        assert len(handler.retry_queue) == 1


class TestTryIndex:
    def test_raises_on_failure(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            handler._try_index(note)

    def test_dataless_raises_not_materialized(self, handler: NoteEventHandler, note: Path):
        with (
            patch("obsidian_rag.watcher.icloud.is_dataless", return_value=True),
            patch("obsidian_rag.watcher.icloud.request_download") as download,
            pytest.raises(FileNotMaterializedError),
        ):
            handler._try_index(note)
        download.assert_called_once_with(note)

    def test_edeadlk_during_read_requests_download_and_reraises(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock(side_effect=OSError(errno.EDEADLK, "Resource deadlock avoided"))
        with (
            patch("obsidian_rag.watcher.icloud.is_dataless", return_value=False),
            patch("obsidian_rag.watcher.icloud.request_download") as download,
            pytest.raises(OSError),
        ):
            handler._try_index(note)
        download.assert_called_once_with(note)

    def test_success_upserts_chunks(self, handler: NoteEventHandler, note: Path):
        handler.indexer.index_file = MagicMock(return_value=[("chunk", [0.1])])
        with patch("obsidian_rag.watcher.icloud.is_dataless", return_value=False):
            handler._try_index(note)
        handler.store.upsert_batch.assert_called_once_with(["chunk"], [[0.1]])


class TestProcessDue:
    def test_one_cycle_calls_failing_item_exactly_once(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(Path("/v/a.md"), now=0.0)
        index_fn = MagicMock(side_effect=RuntimeError("boom"))

        process_due(queue, index_fn, is_permanent=lambda e: False, now=0.0)

        assert index_fn.call_count == 1
        assert len(queue) == 1

    def test_persistent_failure_stops_after_max_retries(self):
        given_up: list[Path] = []
        queue = RetryQueue(max_retries=3, base_delay=60.0, on_give_up=given_up.append)
        path = Path("/v/a.md")
        queue.add(path, now=0.0)
        index_fn = MagicMock(side_effect=RuntimeError("boom"))

        now = 0.0
        for _ in range(20):
            process_due(queue, index_fn, is_permanent=lambda e: False, now=now)
            now += 60.0 * 8

        assert index_fn.call_count == 3
        assert queue.is_empty()
        assert given_up == [path]

    def test_backoff_delays_next_attempt(self):
        queue = RetryQueue(max_retries=5, base_delay=60.0)
        queue.add(Path("/v/a.md"), now=0.0)
        index_fn = MagicMock(side_effect=RuntimeError("boom"))

        process_due(queue, index_fn, is_permanent=lambda e: False, now=0.0)
        process_due(queue, index_fn, is_permanent=lambda e: False, now=30.0)
        assert index_fn.call_count == 1
        process_due(queue, index_fn, is_permanent=lambda e: False, now=60.0)
        assert index_fn.call_count == 2

    def test_permanent_error_is_discarded_without_give_up(self):
        given_up: list[Path] = []
        queue = RetryQueue(max_retries=3, base_delay=60.0, on_give_up=given_up.append)
        queue.add(Path("/v/a.md"), now=0.0)
        index_fn = MagicMock(side_effect=ValueError("bad"))

        process_due(queue, index_fn, is_permanent=lambda e: isinstance(e, ValueError), now=0.0)

        assert queue.is_empty()
        assert given_up == []

    def test_success_removes_item(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(Path("/v/a.md"), now=0.0)
        index_fn = MagicMock()

        process_due(queue, index_fn, is_permanent=lambda e: False, now=0.0)

        assert queue.is_empty()
        assert index_fn.call_count == 1
