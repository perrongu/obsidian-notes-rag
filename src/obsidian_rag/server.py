"""MCP server for obsidian-rag with semantic search tools."""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BeforeValidator

from . import __version__
from .config import Config, load_config
from .embedders import resolve_embedder_settings
from .indexer import Embedder, VaultIndexer
from .store import VectorStore

logger = logging.getLogger(__name__)

# Create MCP server
mcp = MCPServer("obsidian-rag", version=__version__)

# Shared instances, created on first use. mcp 2.x runs synchronous tools on
# worker threads, so each instance has its own lock. get_config() never takes
# another lock, so the store/embedder getters may call it while holding theirs.
_config_lock = threading.Lock()
_embedder_lock = threading.Lock()
_store_lock = threading.Lock()
_config: Config | None = None
_embedder: Embedder | None = None
_store: VectorStore | None = None

# Tool bodies also run concurrently; a second reindex must not clear the store
# while the first one is still upserting.
_reindex_lock = threading.Lock()

_REINDEX_BATCH_SIZE = 50
_SEARCH_CONTENT_CHARS = 500
_SIMILAR_PREVIEW_CHARS = 200
_SIMILAR_EMBED_CHARS = 8000
_SIMILAR_OVERFETCH = 10

P = ParamSpec("P")
R = TypeVar("R")

# Optional tool arguments use concrete sentinel defaults (0 / "") instead of
# ``X | None = None``: the SDK publishes ``X | None`` as ``anyOf [X, null]`` and
# some MCP clients (Claude Code) then reject calls that omit the argument. The
# validators keep an explicit ``null`` working by mapping it to the sentinel.
_OptionalInt = Annotated[int, BeforeValidator(lambda v: 0 if v is None else v)]
_OptionalStr = Annotated[str, BeforeValidator(lambda v: "" if v is None else v)]


def _resolve_limit(limit: int, default: int) -> int:
    """A limit of 0 (or less) means "not provided" and falls back to the config default."""
    return limit if limit > 0 else default


def get_config() -> Config:
    """Get or create config instance."""
    global _config
    with _config_lock:
        if _config is None:
            _config = load_config()
        return _config


def get_embedder() -> Embedder:
    """Get or create embedder instance."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = resolve_embedder_settings(get_config()).create()
        return _embedder


def get_store() -> VectorStore:
    """Get or create store instance."""
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore(data_path=get_config().get_data_path())
        return _store


def _raise_tool_errors(fn: Callable[P, R]) -> Callable[P, R]:
    """Surface failures to the MCP client as ``is_error`` results.

    ``ToolError`` already carries a client-facing message and is re-raised as is.
    Any other exception is logged with its traceback and converted to a
    ``ToolError`` so the message reaches the client instead of the SDK's generic
    "Error executing tool" text.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            logger.exception("%s failed", fn.__name__)
            raise ToolError(str(e)) from e

    return wrapper


def _preview(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _similarity(result: dict) -> float:
    return round(1 - result["distance"], 3)


@mcp.tool()
@_raise_tool_errors
def search_notes(query: str, limit: _OptionalInt = 0, note_type: _OptionalStr = "") -> list[dict]:
    """Search notes using semantic similarity.

    Args:
        query: Search query text
        limit: Maximum number of results (0 = default from config)
        note_type: Filter by note type, "daily" or "note" (empty = no filter)

    Returns:
        List of matching notes with content, file path, and similarity score
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    limit = _resolve_limit(limit, config.indexer.default_search_limit)
    query_embedding = embedder.embed(query, task_type="search_query")
    where = {"type": note_type} if note_type else None
    results = store.search(query_embedding, limit=limit, where=where)
    threshold = config.indexer.similarity_threshold

    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "content": _preview(r["content"], _SEARCH_CONTENT_CHARS),
            "similarity": _similarity(r),
            "type": r["metadata"].get("type", "note"),
        }
        for r in results
        if threshold <= 0 or _similarity(r) >= threshold
    ]


@mcp.tool()
@_raise_tool_errors
def get_similar(note_path: str, limit: _OptionalInt = 0) -> list[dict]:
    """Find notes similar to the given note.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to return (0 = default from config)

    Returns:
        List of similar notes with content preview and similarity score
    """
    limit = _resolve_limit(limit, get_config().indexer.default_similar_limit)
    return _similar_notes(note_path, limit)


def _similar_notes(note_path: str, limit: int) -> list[dict]:
    """Shared body of ``get_similar``; ``limit`` is already resolved (0 = no similar notes)."""
    embedder = get_embedder()
    store = get_store()

    results = store.get_by_file(note_path)
    if not results:
        raise ToolError(f"Note not found: {note_path}")
    if limit <= 0:
        return []

    note_content = "\n\n".join(r["content"] for r in results)
    note_embedding = embedder.embed(note_content[:_SIMILAR_EMBED_CHARS])
    all_results = store.search(note_embedding, limit=limit + _SIMILAR_OVERFETCH)

    similar = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "preview": _preview(r["content"], _SIMILAR_PREVIEW_CHARS),
            "similarity": _similarity(r),
        }
        for r in similar
    ]


@mcp.tool()
@_raise_tool_errors
def get_note_context(note_path: str, limit: _OptionalInt = 0) -> dict:
    """Get a note and its related context.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to include (0 = default from config)

    Returns:
        Note content and list of similar notes for context
    """
    config = get_config()
    store = get_store()
    limit = _resolve_limit(limit, config.indexer.default_context_limit)

    results = store.get_by_file(note_path)
    if not results:
        raise ToolError(f"Note not found: {note_path}")

    note_content = "\n\n".join(r["content"] for r in results)

    return {
        "file_path": note_path,
        "content": note_content,
        "similar_notes": _similar_notes(note_path, limit),
    }


@mcp.tool()
@_raise_tool_errors
def get_stats() -> dict:
    """Get index statistics.

    Returns:
        Statistics about the indexed notes collection
    """
    return get_store().get_stats()


def _index_files(indexer: VaultIndexer, store: VectorStore, files: list[Path]) -> tuple[int, int, list[dict]]:
    """Embed and upsert ``files`` in batches. Returns (files_indexed, chunks_created, errors)."""
    chunk_count = 0
    file_count = 0
    errors: list[dict] = []
    batch_chunks: list = []
    batch_embeddings: list = []

    for file_path in files:
        try:
            for chunk, embedding in indexer.index_file(file_path):
                batch_chunks.append(chunk)
                batch_embeddings.append(embedding)
                chunk_count += 1

                if len(batch_chunks) >= _REINDEX_BATCH_SIZE:
                    store.upsert_batch(batch_chunks, batch_embeddings)
                    batch_chunks = []
                    batch_embeddings = []

            file_count += 1
        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})

    if batch_chunks:
        store.upsert_batch(batch_chunks, batch_embeddings)

    return file_count, chunk_count, errors


@mcp.tool()
@_raise_tool_errors
def reindex(clear: bool = False, path_filter: _OptionalStr = "") -> dict:
    """Re-index the Obsidian vault.

    Only one reindex runs at a time; a second call while one is in progress
    fails immediately instead of clearing the store under the running one.

    Args:
        clear: If True, clear existing index before re-indexing (default: False)
        path_filter: Path prefix to limit indexing, e.g. "Daily Notes/" (empty = whole vault)

    Returns:
        Statistics about the indexing operation
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    if not config.vault_path:
        raise ToolError("No vault path configured. Run 'obsidian-rag setup' first.")

    if not _reindex_lock.acquire(blocking=False):
        raise ToolError("A reindex is already running. Wait for it to finish before starting another.")
    try:
        indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder, config=config.indexer)

        if clear:
            store.clear()

        files = list(indexer.iter_markdown_files())
        if path_filter:
            files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]

        file_count, chunk_count, errors = _index_files(indexer, store, files)
    finally:
        _reindex_lock.release()

    return {
        "files_indexed": file_count,
        "chunks_created": chunk_count,
        "total_in_store": store.get_stats()["count"],
        "errors": errors if errors else None,
        "path_filter": path_filter or None,
        "cleared": clear,
    }


def run_server():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
