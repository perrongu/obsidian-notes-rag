# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Fork Context

This is a **personal fork** of [proofgeist/obsidian-notes-rag](https://github.com/proofgeist/obsidian-notes-rag), maintained at [perrongu/obsidian-notes-rag](https://github.com/perrongu/obsidian-notes-rag).

- **Branch**: `custom/main` (all work happens here, based on upstream tag `v1.1.2`)
- **Upstream**: `proofgeist/obsidian-notes-rag` (remote named `upstream`)
- **Do not push to `main`** — it mirrors upstream

Custom modifications vs upstream v1.1.2:
- macOS Keychain integration for API key resolution (`config.py`)
- tiktoken-based token truncation for embedding safety (`indexer.py`)
- Retry with exponential backoff on OpenAI API calls (`indexer.py`)
- Batch embedding with sub-batching, 100 texts max per request (`indexer.py`)
- Structured logging in server and watcher
- Safe AppleScript execution via argv passing (`watcher.py`)
- Permanent error detection to skip non-transient retry loops (`watcher.py`)
- plistlib-based plist generation (`cli.py`)
- Filter column validation in store queries (`store.py`)
- Lazy config loading in watcher (`watcher.py`)

## Commands

```bash
# Install dependencies (development)
uv sync --dev

# Run tests
uv run pytest -v
uv run pytest tests/test_store.py -v      # single file

# Type checking
uv run pyright

# Install the tool from this fork (production)
uv tool install --force --python /opt/homebrew/bin/python3.13 \
  "obsidian-notes-rag @ git+https://github.com/perrongu/obsidian-notes-rag.git@custom/main"

# Watcher service management
launchctl unload ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist  # stop
launchctl load ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist    # start
launchctl list | grep obsidian-notes                                          # status

# CLI (when installed as uv tool)
obsidian-rag stats
obsidian-rag search "query" --limit 5
obsidian-rag index                         # incremental reindex
obsidian-rag index --clear                 # full reindex (only if chunking logic changed)
```

## Critical: Python Requirement

**Always use `--python /opt/homebrew/bin/python3.13`** when installing via `uv tool install`. The python.org Framework Python does not compile sqlite3 with `enable_load_extension`, which breaks sqlite-vec entirely.

## Architecture

```
Obsidian Vault → VaultIndexer → Embedder (OpenAI/Ollama/LMStudio) → VectorStore (sqlite-vec)
                      |               |                                      ↓
                 chunk_markdown   _truncate_for_embedding          MCP Client ← FastMCP Server
                 (Chonkie)        + _call_with_retry
```

### Key Components (src/obsidian_rag/)

- **config.py**: `Config` dataclass with `get_openai_api_key()` (config → env → Keychain chain), `load_config()`/`save_config()` for TOML
- **indexer.py**: `VaultIndexer` scans markdown, `OpenAIEmbedder` with retry/truncation/batching, `create_embedder()` factory, `IndexerConfig` with `_make_defaults()` classmethod
- **store.py**: `VectorStore` wraps sqlite-vec, two tables (chunks + chunks_vec virtual table), thread-safe, filter column validation via `_ALLOWED_FILTER_COLUMNS`
- **server.py**: FastMCP server with 5 tools (`search_notes`, `get_similar`, `get_note_context`, `get_stats`, `reindex`), structured logging, lazy-initialized globals
- **watcher.py**: `VaultWatcher` with watchdog, debouncing (2s), `RetryQueue`, `_is_permanent_error()` classification, macOS notifications via safe AppleScript
- **cli.py**: Click CLI, plistlib-based plist generation, `install-service`/`uninstall-service` commands

### Chunking

Chonkie RecursiveChunker: 1500 tokens max, 50 chars minimum, splits by heading > paragraph > line > sentence > word. Code blocks preserved. Chunks exceeding 8191 tokens are truncated via tiktoken before embedding.

### Embedding Safety

`OpenAIEmbedder` in indexer.py handles:
1. Token truncation to 8191 via `_truncate_for_embedding()` (tiktoken with cl100k_base fallback)
2. Retry with exponential backoff (1s, 4s, 16s) on rate limits, timeouts, 5xx errors
3. Sub-batching at 100 texts per API call to stay under OpenAI's 300k token/request limit

## Deployed Paths

| What | Where |
|------|-------|
| Binary | `~/.local/bin/obsidian-rag` |
| Installed package | `~/.local/share/uv/tools/obsidian-notes-rag/` |
| Config | `~/Library/Application Support/obsidian-notes-rag/config.toml` |
| SQLite databases | `~/Library/Application Support/obsidian-notes-rag/*.db` |
| Logs | `~/Library/Logs/obsidian-notes-rag/` |
| Watcher plist | `~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist` |
| MCP config (Claude Code) | `~/.claude.json` |
| MCP config (Claude Desktop) | `~/Library/Application Support/Claude/claude_desktop_config.json` |

Reinstalling the package never touches databases or config.toml. A full reindex (`--clear`) is only needed if chunking logic changes.

## Upstream Sync

```bash
git fetch upstream
git log --oneline custom/main..upstream/main   # check what's new
git merge upstream/main                         # merge, conflicts likely in indexer.py
git push origin custom/main
# then reinstall (see Commands section)
```

## Testing

- `test_store.py` — VectorStore contract tests
- `test_indexer.py` — frontmatter parsing, chunk_markdown
- `test_indexer_config.py` — IndexerConfig presets, serialization
- `test_cli.py` — CLI commands
