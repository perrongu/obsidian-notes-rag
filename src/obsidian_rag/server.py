"""MCP server for obsidian-rag with semantic search tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .config import Config, load_config
from .indexer import Embedder, VaultIndexer, create_embedder
from .store import VectorStore

logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("obsidian-rag")

# Global instances (lazy initialized — server runs single-threaded via stdio)
_config: Config | None = None
_embedder: Embedder | None = None
_store: VectorStore | None = None


def get_config() -> Config:
    """Get or create config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_embedder() -> Embedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        config = get_config()

        # Resolve API key from config, env, or Keychain
        resolved_api_key: str | None = None
        if config.provider == "openai":
            resolved_api_key = config.get_openai_api_key()
            if not resolved_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Configure via environment variable, "
                    "config.toml [openai] api_key, or macOS Keychain."
                )

        # Determine model and base_url based on provider
        if config.provider == "openai":
            model = config.openai_model
            base_url = None
        elif config.provider == "ollama":
            model = config.ollama_model
            base_url = config.ollama_url
        else:  # lmstudio
            model = config.lmstudio_model
            base_url = config.lmstudio_url

        _embedder = create_embedder(
            provider=config.provider,
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
        )
    return _embedder


def get_store() -> VectorStore:
    """Get or create store instance."""
    global _store
    if _store is None:
        config = get_config()
        _store = VectorStore(data_path=config.get_data_path())
    return _store


@mcp.tool()
def search_notes(query: str, limit: int | None = None, note_type: str | None = None) -> list[dict]:
    """Search notes using semantic similarity.

    Args:
        query: Search query text
        limit: Maximum number of results (default: from config)
        note_type: Optional filter - "daily" or "note"

    Returns:
        List of matching notes with content, file path, and similarity score
    """
    try:
        config = get_config()
        embedder = get_embedder()
        store = get_store()

        if limit is None:
            limit = config.indexer.default_search_limit

        query_embedding = embedder.embed(query, task_type="search_query")
        where = {"type": note_type} if note_type else None
        results = store.search(query_embedding, limit=limit, where=where)
        threshold = config.indexer.similarity_threshold

        return [
            {
                "file_path": r["metadata"]["file_path"],
                "heading": r["metadata"].get("heading") or None,
                "content": r["content"][:500] if len(r["content"]) > 500 else r["content"],
                "similarity": round(1 - r["distance"], 3),
                "type": r["metadata"].get("type", "note"),
            }
            for r in results
            if threshold <= 0 or (1 - r["distance"]) >= threshold
        ]
    except Exception as e:
        logger.error("search_notes failed: %s", e)
        return [{"error": str(e)}]


@mcp.tool()
def get_similar(note_path: str, limit: int | None = None) -> list[dict]:
    """Find notes similar to the given note.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to return (default: from config)

    Returns:
        List of similar notes with content preview and similarity score
    """
    try:
        config = get_config()
        embedder = get_embedder()
        store = get_store()

        if limit is None:
            limit = config.indexer.default_similar_limit

        results = store.get_by_file(note_path)
        if not results:
            return [{"error": f"Note not found: {note_path}"}]

        note_content = "\n\n".join(r["content"] for r in results)
        note_embedding = embedder.embed(note_content[:8000])
        all_results = store.search(note_embedding, limit=limit + 10)

        similar = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

        return [
            {
                "file_path": r["metadata"]["file_path"],
                "heading": r["metadata"].get("heading") or None,
                "preview": r["content"][:200] if len(r["content"]) > 200 else r["content"],
                "similarity": round(1 - r["distance"], 3),
            }
            for r in similar
        ]
    except Exception as e:
        logger.error("get_similar failed: %s", e)
        return [{"error": str(e)}]


@mcp.tool()
def get_note_context(note_path: str, limit: int | None = None) -> dict:
    """Get a note and its related context.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to include (default: from config)

    Returns:
        Note content and list of similar notes for context
    """
    try:
        config = get_config()
        store = get_store()

        if limit is None:
            limit = config.indexer.default_context_limit

        results = store.get_by_file(note_path)
        if not results:
            return {"error": f"Note not found: {note_path}"}

        note_content = "\n\n".join(r["content"] for r in results)
        similar = get_similar(note_path, limit=limit)

        return {
            "file_path": note_path,
            "content": note_content,
            "similar_notes": similar if not (similar and "error" in similar[0]) else [],
        }
    except Exception as e:
        logger.error("get_note_context failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
def get_stats() -> dict:
    """Get index statistics.

    Returns:
        Statistics about the indexed notes collection
    """
    try:
        store = get_store()
        return store.get_stats()
    except Exception as e:
        logger.error("get_stats failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
def reindex(clear: bool = False, path_filter: str | None = None) -> dict:
    """Re-index the Obsidian vault.

    Args:
        clear: If True, clear existing index before re-indexing (default: False)
        path_filter: Optional path prefix to limit indexing (e.g., "Daily Notes/")

    Returns:
        Statistics about the indexing operation
    """
    try:
        config = get_config()
        embedder = get_embedder()
        store = get_store()

        if not config.vault_path:
            return {"error": "No vault path configured. Run 'obsidian-rag setup' first."}

        indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder, config=config.indexer)

        if clear:
            store.clear()

        files = list(indexer.iter_markdown_files())
        if path_filter:
            files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]

        chunk_count = 0
        file_count = 0
        errors = []
        batch_chunks = []
        batch_embeddings = []
        batch_size = 50

        for file_path in files:
            try:
                for chunk, embedding in indexer.index_file(file_path):
                    batch_chunks.append(chunk)
                    batch_embeddings.append(embedding)
                    chunk_count += 1

                    if len(batch_chunks) >= batch_size:
                        store.upsert_batch(batch_chunks, batch_embeddings)
                        batch_chunks = []
                        batch_embeddings = []

                file_count += 1
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})

        if batch_chunks:
            store.upsert_batch(batch_chunks, batch_embeddings)

        return {
            "files_indexed": file_count,
            "chunks_created": chunk_count,
            "total_in_store": store.get_stats()["count"],
            "errors": errors if errors else None,
            "path_filter": path_filter,
            "cleared": clear,
        }
    except Exception as e:
        logger.error("reindex failed: %s", e)
        return {"error": str(e)}


def run_server():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
