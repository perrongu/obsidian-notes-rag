[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/obsidian-notes-rag)](https://pypi.org/project/obsidian-notes-rag/)

# obsidian-notes-rag

MCP server and CLI for semantic search over your Obsidian vault. Generates embeddings with OpenAI, Ollama, or LM Studio. Stores vectors locally in sqlite-vec (~200KB, no telemetry, no network calls).

## What it does

Search your notes by meaning, not just keywords:

```bash
obsidian-rag search "project architecture decisions" -n 5
obsidian-rag similar "Projects/Platform Hub.md"
obsidian-rag context "Daily Notes/2026-02-14.md"
```

As an MCP server, it gives any compatible AI assistant the same capabilities — searching your notes, finding related content, and pulling context during conversations.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for running and installing)
- One of: `OPENAI_API_KEY`, [Ollama](https://ollama.ai/), or [LM Studio](https://lmstudio.ai/) for embeddings

## Setup

### 1. Run the setup wizard

```bash
uvx obsidian-notes-rag setup
```

This creates a config at `~/.config/obsidian-notes-rag/config.toml` with your vault path, embedding provider, and API key.

### 2. Build the index

```bash
uvx obsidian-notes-rag index
```

Parses your markdown files, chunks them by heading structure (using [Chonkie](https://github.com/chonkie-ai/chonkie) RecursiveChunker), generates embeddings, and stores everything in a local SQLite database.

### 3. Connect to an MCP client

Works with any MCP-compatible client. Examples:

**Claude Code:**

```bash
claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve
```

**Claude Desktop, Cursor, Windsurf, etc. (JSON config):**

Add to your client's MCP config file (e.g. `~/Library/Application Support/Claude/claude_desktop_config.json` for Claude Desktop on macOS):

```json
{
  "mcpServers": {
    "obsidian-notes-rag": {
      "command": "uvx",
      "args": ["obsidian-notes-rag", "serve"]
    }
  }
}
```

### 4. Install the CLI (optional)

If you want `obsidian-rag` available as a standalone command:

```bash
uv tool install obsidian-notes-rag
```

This installs both `obsidian-rag` and `obsidian-notes-rag` to `~/.local/bin/`.

### Using the CLI with AI coding assistants

Instead of running the MCP server, you can have your AI assistant call the CLI directly via shell commands. This avoids loading MCP tool definitions into the context window, freeing up tokens for your actual work.

To do this, create a rule or skill that tells your assistant when and how to use the CLI:

- **Claude Code**: Create a [skill](https://docs.anthropic.com/en/docs/claude-code/skills) with CLI usage instructions
- **Cursor**: Add a [rule](https://docs.cursor.com/context/rules) to `.cursor/rules/`
- **Windsurf**: Add a [rule](https://docs.windsurf.com/windsurf/memories#rules) to `.windsurfrules`

The rule should describe when to use each command (`search`, `similar`, `context`) and any project-specific conventions. This gives the assistant enough context to run the right CLI commands without the overhead of an MCP connection.

## CLI Reference

```bash
# Search
obsidian-rag search "query"                  # semantic search
obsidian-rag search "standup" --type daily   # filter by note type
obsidian-rag search "design" -n 10           # more results

# Explore
obsidian-rag similar "Path/To/Note.md"       # find related notes
obsidian-rag context "Path/To/Note.md"       # show note + related context

# Index
obsidian-rag index                            # re-index vault
obsidian-rag index --clear                    # rebuild from scratch
obsidian-rag index --path-filter "Daily Notes/"  # index subset

# Info
obsidian-rag stats                            # show index size

# Services
obsidian-rag serve                            # start MCP server
obsidian-rag watch                            # watch for changes, auto-reindex
obsidian-rag install-service                  # macOS launchd auto-start
obsidian-rag uninstall-service                # remove service
obsidian-rag service-status                   # check service status
```

## MCP Tools

Once connected, your AI assistant has access to:

| Tool | What it does |
|------|--------------|
| `search_notes` | Find notes matching a query |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get a note with related context |
| `get_stats` | Show index statistics |
| `reindex` | Rebuild the index |

## Keeping the Index Fresh

**Manual:** `obsidian-rag index`

**Auto-reindex on file changes:** `obsidian-rag watch` (run in a terminal or background)

**macOS background service:** `obsidian-rag install-service` (starts on login, appears in System Settings > Login Items)

## Using Ollama (local, no API key)

```bash
ollama pull nomic-embed-text
obsidian-rag --provider ollama index
```

## Using LM Studio (local, no API key)

Load an embedding model in LM Studio, then:

```bash
obsidian-rag --provider lmstudio index
```

## Configuration

The setup wizard writes to `~/.config/obsidian-notes-rag/config.toml`. You can also override with environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OBSIDIAN_RAG_PROVIDER` | `openai` (default), `ollama`, or `lmstudio` |
| `OBSIDIAN_RAG_VAULT` | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | Index storage path (default: platform-specific) |
| `OBSIDIAN_RAG_OLLAMA_URL` | Ollama URL (default: `http://localhost:11434`) |
| `OBSIDIAN_RAG_LMSTUDIO_URL` | LM Studio URL (default: `http://localhost:1234`) |
| `OBSIDIAN_RAG_MODEL` | Override embedding model |

## How it works

1. Parses markdown files, strips YAML frontmatter
2. Chunks content using Chonkie's RecursiveChunker (splits by headings > paragraphs > lines > sentences, max 1500 tokens per chunk)
3. Generates embeddings via your chosen provider
4. Stores metadata in SQLite, vectors in sqlite-vec (KNN search via vec0 virtual tables)
5. MCP server and CLI both query the same local database

## Upgrading

If you installed the CLI with `uv tool install`, upgrade with:

```bash
uv tool upgrade obsidian-notes-rag
```

If you use `uvx` to run commands or the MCP server, it automatically uses the latest version.

### Upgrading to v1.0.0

v1.0.0 replaces ChromaDB with sqlite-vec. After upgrading, rebuild your index:

```bash
obsidian-rag index --clear
```

The old ChromaDB data at `~/.local/share/obsidian-notes-rag/` (or your configured path) can be deleted.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Fork Maintenance (perrongu/obsidian-notes-rag)

This is a fork of [proofgeist/obsidian-notes-rag](https://github.com/proofgeist/obsidian-notes-rag) with custom modifications on the `custom/main` branch.

### Custom modifications (vs upstream v1.1.2)

- **Keychain macOS** — API key resolution: config → env → Keychain (`config.py`)
- **Truncation tiktoken** — respect 8191 token limit for embeddings (`indexer.py`)
- **Retry with backoff** — exponential backoff on OpenAI API calls (`indexer.py`)
- **Batch embedding** — sub-batches of 100 texts max per request (`indexer.py`)
- **Structured logging** — in MCP server and watcher (`server.py`, `watcher.py`)
- **Safe AppleScript** — argv passing instead of f-string interpolation (`watcher.py`)
- **Permanent error detection** — avoid retry loops on non-transient errors (`watcher.py`)
- **plistlib** — safe plist generation (`cli.py`)
- **Filter validation** — allowed columns in store queries (`store.py`)
- **Lazy config** — deferred loading in watcher (`watcher.py`)

### Key paths

| Element | Path |
|---------|------|
| Repo | `~/obsidian-notes-rag/` |
| Binary | `~/.local/bin/obsidian-rag` |
| Installed package | `~/.local/share/uv/tools/obsidian-notes-rag/` |
| Config | `~/Library/Application Support/obsidian-notes-rag/config.toml` |
| Databases | `~/Library/Application Support/obsidian-notes-rag/*.db` |
| Logs | `~/Library/Logs/obsidian-notes-rag/` |
| Watcher plist | `~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist` |
| MCP (Claude Code) | `~/.claude.json` |
| MCP (Claude Desktop) | `~/Library/Application Support/Claude/claude_desktop_config.json` |

### Sync upstream updates

```bash
cd ~/obsidian-notes-rag
git fetch upstream
git log --oneline custom/main..upstream/main   # see what's new
git checkout custom/main
git merge upstream/main                         # resolve conflicts if any
git push origin custom/main
```

### Reinstall after update

```bash
# 1. Stop watcher
launchctl unload ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist

# 2. Reinstall from fork (IMPORTANT: --python to force homebrew)
uv tool install --force --python /opt/homebrew/bin/python3.13 \
  "obsidian-notes-rag @ git+https://github.com/perrongu/obsidian-notes-rag.git@custom/main"

# 3. Restart watcher
launchctl load ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist
```

### Verification checklist

```bash
which obsidian-rag                         # → ~/.local/bin/obsidian-rag
obsidian-rag stats                         # → shows document count
obsidian-rag search "test" --limit 1       # → returns results
launchctl list | grep obsidian-notes       # → PID visible
head -1 ~/.local/share/uv/tools/obsidian-notes-rag/pyvenv.cfg
# → home = /opt/homebrew/...  (NOT /Library/Frameworks)
```

### Rollback to PyPI (if fork breaks)

```bash
launchctl unload ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist
uv tool install --force --python /opt/homebrew/bin/python3.13 obsidian-notes-rag==1.1.2
launchctl load ~/Library/LaunchAgents/com.obsidian-notes-rag.watcher.plist
```

> **Note:** rollback to PyPI loses all custom modifications (no retry, no truncation, etc.)

### Why `--python /opt/homebrew/bin/python3.13`

The Python from `/Library/Frameworks/` (python.org installer) does not compile sqlite3 with
`enable_load_extension`, which prevents sqlite-vec from working. Homebrew Python includes
this capability. Always specify this flag when installing.

### Data safety

- Reinstalling the package **never** touches SQLite databases or `config.toml`
- A full reindex (`obsidian-rag index --clear`) is only needed if chunking logic changes
- Without `--clear`, indexing is additive (upsert)

---

## Support

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ernestkoe)

## License

MIT
