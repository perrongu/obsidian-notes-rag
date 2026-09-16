# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Fork Context

This is a **personal fork** of [proofgeist/obsidian-notes-rag](https://github.com/proofgeist/obsidian-notes-rag), maintained at [perrongu/obsidian-notes-rag](https://github.com/perrongu/obsidian-notes-rag).

- **Branch `dev`**: active development branch — all work happens here
- **Branch `custom/main`**: stable/deployed version — merge from `dev` when ready, then reinstall
- **Branch `main`**: mirrors upstream — do not push to it
- **Upstream**: `proofgeist/obsidian-notes-rag` (remote named `upstream`), forked at tag `v1.1.2`

Custom modifications vs upstream v1.1.2:
- macOS Keychain integration for API key resolution (`config.py`)
- tiktoken-based token truncation for embedding safety (`indexer.py`)
- Retry with exponential backoff on OpenAI API calls (`indexer.py`)
- Batch embedding with sub-batching, 100 texts max per request (`indexer.py`)
- Structured logging in server and watcher
- Safe AppleScript execution via argv passing (`watcher.py`)
- Permanent error detection to skip non-transient retry loops (`watcher.py`)
- Retry queue with per-path attempt tracking and exponential backoff, one pass per health cycle (`retry_queue.py`)
- iCloud dataless-file handling: detect evicted files, request `brctl download`, retry later (`icloud.py`)
- plistlib-based plist generation (`cli.py`)
- Filter column validation in store queries (`store.py`)
- Lazy config loading in watcher (`watcher.py`)
- Shared provider -> embedder resolution (`embedders.py`) used by CLI, watcher and server
- MCP server on mcp 2.x `MCPServer` (upstream pins `mcp<2`): per-singleton locks for lazy init, single-flight `reindex`, failures raised as `ToolError` (`server.py`)

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

**Always use `--python /opt/homebrew/bin/python3.13`** for both `uv tool install` and `uv venv`. The python.org Framework Python does not compile sqlite3 with `enable_load_extension`, which breaks sqlite-vec entirely (runtime and tests).

```bash
# Dev venv setup (if .venv is missing or broken)
uv venv --python /opt/homebrew/bin/python3.13
uv sync --dev
```

## Architecture

```
Obsidian Vault → VaultIndexer → Embedder (OpenAI/Ollama/LMStudio) → VectorStore (sqlite-vec)
                      |               |                                      ↓
                 chunk_markdown   _truncate_for_embedding          MCP Client ← MCPServer (mcp 2.x)
                 (Chonkie)        + _call_with_retry
```

### Key Components (src/obsidian_rag/)

- **defaults.py**: the provider defaults (`DEFAULT_PROVIDER`, `DEFAULT_*_MODEL`, `DEFAULT_*_URL`), a leaf module with no internal imports; every other module reads them from here instead of repeating the literals
- **config.py**: `Config` dataclass with `get_openai_api_key()` (config → env → Keychain chain), `load_config()`/`save_config()` for TOML
- **indexer.py**: `VaultIndexer` scans markdown, `OpenAIEmbedder` with retry/truncation/batching, `create_embedder()` factory, `IndexerConfig` with `_make_defaults()` classmethod (derived from the dataclass field defaults)
- **embedders.py**: `resolve_embedder_settings(config, **overrides)` -> frozen `EmbedderSettings` with `.create()`; the single place provider/model/base_url/API key are resolved (CLI, watcher, server). Raises `MissingApiKeyError` for OpenAI without a key
- **store.py**: `VectorStore` wraps sqlite-vec, two tables (chunks + chunks_vec virtual table), thread-safe, filter column validation via `_ALLOWED_FILTER_COLUMNS`
- **server.py**: `MCPServer` (mcp 2.x) with 5 tools (`search_notes`, `get_similar`, `get_note_context`, `get_stats`, `reindex`), failures raised as `ToolError` (client sees `is_error=True`), lock-guarded lazy-initialized globals and a single-flight `reindex` (mcp 2.x runs sync tools on worker threads)
- **watcher.py**: `VaultWatcher` with watchdog, debouncing (2s), `_is_permanent_error()` classification, macOS notifications via safe AppleScript; `_try_index()` raises, `_index_file()` queues
- **retry_queue.py**: `RetryQueue` (attempts + backoff per path, `pop_due` snapshot) and `process_due()` (one retry pass, never spins)
- **icloud.py**: `is_dataless()`, `is_dataless_error()` (EDEADLK), `request_download()` via `brctl`
- **cli.py**: Click CLI, plistlib-based plist generation, `_install_watcher_service()` shared by `setup` and `install-service` (`launchctl load`/`unload` go through `_launchctl()`, the seam tests replace), `setup` wizard split into `_prompt_*` helpers with the provider menu driven by `PROVIDERS`

### Chunking

Chonkie RecursiveChunker: 1500 tokens max, 50 chars minimum, splits by heading > paragraph > line > sentence > word. Code blocks preserved. Before embedding, texts of at most 8191 UTF-8 bytes are sent as is (every default-size chunk); longer texts are counted with tiktoken and cut at 8191 tokens.

### Embedding Safety

`OpenAIEmbedder` in indexer.py handles:
1. Token truncation to 8191 via `_truncate_for_embedding()`: byte-length short-circuit first, then tiktoken (model encoding, cl100k_base if unknown)
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

## Git Workflow

Work on feature branches off `dev`. When done:

```bash
# 1. Merge PR into dev (via GitHub), close PR
# 2. Locally:
git checkout dev
git pull
git checkout custom/main
git merge dev
git push origin custom/main
# 3. Reinstall (see Commands section)
# 4. Cleanup:
git branch -d <feature-branch>
git push origin --delete <feature-branch>
git checkout dev
```

## Upstream Sync

```bash
git fetch upstream
git log --oneline custom/main..upstream/main   # check what's new
git merge upstream/main                         # merge, conflicts likely in indexer.py, server.py, pyproject.toml
git push origin custom/main
# then reinstall (see Commands section)
```

## Testing

- `test_store.py` — VectorStore contract tests
- `test_config.py` — shared provider defaults, `save_config` minimal output, TOML round-trip
- `test_indexer.py` — frontmatter parsing, chunk_markdown
- `test_indexer_config.py` — IndexerConfig presets, serialization
- `test_cli.py` — CLI commands, shared misconfiguration error across embedding commands
- `test_cli_setup.py` — `setup` wizard driven through `CliRunner` input, watcher-service install with a fake `launchctl`, every path redirected to `tmp_path`
- `test_embedders.py` — provider -> embedder resolution, overrides, typed config errors
- `test_server.py` — MCP tools driven through an in-process client, error results, reindex guard, thread-safe lazy init
- `test_retry_queue.py` — RetryQueue backoff, give-up, snapshot semantics
- `test_icloud.py` — dataless detection, EDEADLK classification, brctl invocation
- `test_watcher.py` — handler retry behavior, `process_due` never hot-loops
