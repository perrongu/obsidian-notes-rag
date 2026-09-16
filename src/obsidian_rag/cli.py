"""Command-line interface for obsidian-notes-rag."""

import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import click

# Suppress noisy HTTP logs from httpx/openai during progress bars
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from .config import PROVIDER_URL_ENV, Config, get_config_path, get_data_dir, load_config, save_config
from .defaults import (
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROVIDER,
)
from .embedders import EmbedderConfigError, EmbedderSettings, resolve_embedder_settings
from .indexer import (
    PROVIDERS,
    Chunk,
    VaultIndexer,
    get_lmstudio_models,
    get_ollama_models,
    is_lmstudio_running,
    is_ollama_running,
)
from .server import run_server
from .store import VectorStore
from .watcher import VaultWatcher


def _embedder_settings(ctx: click.Context) -> EmbedderSettings:
    """Resolve the embedder from config plus the global CLI overrides; fail with a one-line error if misconfigured."""
    try:
        return resolve_embedder_settings(ctx.obj["config"], **ctx.obj["overrides"])
    except EmbedderConfigError as e:
        raise click.ClickException(str(e)) from e


@click.group()
@click.option("--vault", default=None, help="Path to Obsidian vault")
@click.option("--data", default=None, help="Path to vector store data")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(list(PROVIDERS)),
    help=f"Embedding provider (default: {DEFAULT_PROVIDER})",
)
@click.option("--ollama-url", default=None, help="Ollama API URL (only used with --provider ollama)")
@click.option("--lmstudio-url", default=None, help="LM Studio API URL (only used with --provider lmstudio)")
@click.option("--model", default=None, help="Override embedding model name")
@click.pass_context
def main(ctx, vault, data, provider, ollama_url, lmstudio_url, model):
    """Obsidian RAG - Semantic search for your Obsidian vault."""
    ctx.ensure_object(dict)

    # Load config from file, then apply CLI overrides
    config = load_config()

    ctx.obj["vault"] = vault or config.vault_path or ""
    ctx.obj["data"] = data or config.get_data_path()
    # Raw CLI overrides (None when absent); precedence over config is applied once, in resolve_embedder_settings
    ctx.obj["overrides"] = {
        "provider": provider,
        "model": model,
        "ollama_url": ollama_url,
        "lmstudio_url": lmstudio_url,
    }
    ctx.obj["config"] = config


PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI (recommended - requires API key)",
    "ollama": "Ollama (local, offline)",
    "lmstudio": "LM Studio (local, offline)",
}

_INDEX_BATCH_SIZE = 50


def _iter_chunks(indexer: VaultIndexer, files: list[Path]) -> Iterator[tuple[Chunk, list[float]]]:
    """Yield (chunk, embedding) for every file behind a progress bar; a failing file is reported and skipped."""
    with click.progressbar(files, label="Indexing") as bar:
        for file_path in bar:
            try:
                yield from indexer.index_file(file_path)
            except Exception as e:
                click.echo(f"\nError indexing {file_path}: {e}", err=True)


def _index_with_progress(indexer: VaultIndexer, store: VectorStore, files: list[Path]) -> int:
    """Embed and upsert ``files`` in batches; returns the number of chunks stored."""
    chunk_count = 0
    batch_chunks: list[Chunk] = []
    batch_embeddings: list[list[float]] = []
    for chunk, embedding in _iter_chunks(indexer, files):
        batch_chunks.append(chunk)
        batch_embeddings.append(embedding)
        chunk_count += 1
        if len(batch_chunks) >= _INDEX_BATCH_SIZE:
            store.upsert_batch(batch_chunks, batch_embeddings)
            batch_chunks, batch_embeddings = [], []
    if batch_chunks:
        store.upsert_batch(batch_chunks, batch_embeddings)
    return chunk_count


def _confirm_overwrite_existing_config() -> bool:
    config_path = get_config_path()
    if config_path.exists() and not click.confirm(f"Config already exists at {config_path}. Overwrite?"):
        click.echo("Setup cancelled.")
        return False
    return True


def _prompt_provider() -> str:
    click.echo("Select embedding provider:")
    for number, provider in enumerate(PROVIDERS, 1):
        click.echo(f"  {number}. {PROVIDER_LABELS[provider]}")
    choices = [str(number) for number in range(1, len(PROVIDERS) + 1)]
    choice = click.prompt("Choice", type=click.Choice(choices), default="1")
    return PROVIDERS[int(choice) - 1]


def _prompt_openai() -> dict[str, str | None]:
    """Config fields for OpenAI: the key is only stored when the user asks for it."""
    existing_key = Config().get_openai_api_key()  # environment, then macOS Keychain
    if existing_key:
        click.echo("\n✓ Found OPENAI_API_KEY (environment or Keychain)")
        keep = click.confirm("Save API key to config file?", default=False)
        return {"openai_api_key": existing_key if keep else None}
    click.echo("\nNo OPENAI_API_KEY found in environment or Keychain.")
    return {"openai_api_key": click.prompt("Enter your OpenAI API key", hide_input=True)}


def _choose_model(models: list[str], other_label: str, other_prompt: str) -> str:
    """Numbered menu over ``models`` plus an "Other" entry that asks for a free-form name."""
    click.echo("\nSelect embedding model:")
    for number, model in enumerate(models, 1):
        click.echo(f"  {number}. {model}")
    click.echo(f"  {len(models) + 1}. {other_label}")

    choices = [str(number) for number in range(1, len(models) + 2)]
    index = int(click.prompt("Choice", type=click.Choice(choices), default="1")) - 1
    return models[index] if index < len(models) else click.prompt(other_prompt)


@dataclass(frozen=True)
class _LocalProvider:
    """How the wizard talks to a locally hosted embedding server."""

    name: str
    url_field: str
    model_field: str
    default_url: str
    default_model: str
    is_running: Callable[[str], bool]
    list_models: Callable[[str], list[str]]
    model_noun: str
    install_hint: tuple[str, ...] = ()


# The probes are looked up at call time so tests can replace them on the module.
_OLLAMA = _LocalProvider(
    name="Ollama",
    url_field="ollama_url",
    model_field="ollama_model",
    default_url=DEFAULT_OLLAMA_URL,
    default_model=DEFAULT_OLLAMA_MODEL,
    is_running=lambda url: is_ollama_running(url),
    list_models=lambda url: get_ollama_models(url),
    model_noun="model name",
    install_hint=(f"Install {DEFAULT_OLLAMA_MODEL}:", f"  ollama pull {DEFAULT_OLLAMA_MODEL}"),
)
_LMSTUDIO = _LocalProvider(
    name="LM Studio",
    url_field="lmstudio_url",
    model_field="lmstudio_model",
    default_url=DEFAULT_LMSTUDIO_URL,
    default_model=DEFAULT_LMSTUDIO_MODEL,
    is_running=lambda url: is_lmstudio_running(url),
    list_models=lambda url: get_lmstudio_models(url),
    model_noun="model identifier",
)


def _prompt_local_provider(spec: _LocalProvider) -> dict[str, str]:
    """Ask for the server URL and the embedding model; returns the Config fields for ``spec``."""
    url = click.prompt(f"\n{spec.name} API URL", default=spec.default_url)
    model_prompt = f"\nEnter embedding {spec.model_noun}"

    click.echo(f"Checking {spec.name} server...", nl=False)
    if not spec.is_running(url):
        click.echo(" not detected (server may still work)")
        click.echo("Could not auto-detect models.")
        return {spec.url_field: url, spec.model_field: click.prompt(model_prompt, default=spec.default_model)}

    click.echo(" ✓ connected")
    click.echo("Fetching available embedding models...", nl=False)
    models = spec.list_models(url)
    if models:
        click.echo(f" found {len(models)}")
        model = _choose_model(models, f"Other (enter {spec.model_noun})", model_prompt)
    else:
        click.echo(" none found")
        for line in ("\nNo embedding models detected.", *spec.install_hint):
            click.echo(line)
        model = click.prompt(model_prompt, default=spec.default_model)
    return {spec.url_field: url, spec.model_field: model}


_PROVIDER_PROMPTS: dict[str, Callable[[], dict[str, Any]]] = {
    "openai": _prompt_openai,
    "ollama": partial(_prompt_local_provider, _OLLAMA),
    "lmstudio": partial(_prompt_local_provider, _LMSTUDIO),
}


def _prompt_vault_path() -> str | None:
    """Ask until an existing directory is given; ``None`` when the user gives up."""
    while True:
        vault_path = os.path.expanduser(click.prompt("\nPath to your Obsidian vault"))
        if Path(vault_path).is_dir():
            # No file count here: walking a large iCloud vault is slow and the initial indexing reports it anyway
            click.echo("✓ Vault found")
            return vault_path
        click.echo(f"✗ Directory not found: {vault_path}")
        if not click.confirm("Try again?", default=True):
            click.echo("Setup cancelled.")
            return None


def _prompt_data_path() -> str | None:
    """Ask where to store the index; ``None`` when the default is kept so config.toml does not pin it."""
    default = str(get_data_dir())
    data_path = os.path.expanduser(click.prompt("\nWhere to store the search index?", default=default))
    return None if data_path == default else data_path


def _maybe_initial_index(config: Config, vault_path: str, settings: EmbedderSettings) -> None:
    if not click.confirm("\nRun initial indexing now?", default=True):
        return
    click.echo("\nIndexing vault...")
    try:
        embedder = settings.create()
        store = VectorStore(data_path=config.get_data_path())
        indexer = VaultIndexer(vault_path=vault_path, embedder=embedder, config=config.indexer)
        files = list(indexer.iter_markdown_files())
        chunk_count = _index_with_progress(indexer, store, files)
        embedder.close()
        click.echo(f"\n✓ Indexed {chunk_count} chunks from {len(files)} files")
    except Exception as e:
        click.echo(f"\n✗ Indexing failed: {e}", err=True)
        click.echo("You can run indexing later with: obsidian-notes-rag index")


def _maybe_install_service(config: Config, vault_path: str, settings: EmbedderSettings) -> None:
    click.echo("\nThe watcher service auto-indexes notes when they change.")
    if sys.platform != "darwin":
        # Linux/Windows: no background service support yet
        click.echo("  Background service not yet supported on this platform.")
        click.echo("  To auto-index on file changes, run: obsidian-notes-rag watch")
        return
    if not click.confirm("Install watcher as a background service?", default=True):
        return
    try:
        _install_watcher_service(vault_path, config.get_data_path(), settings.provider, settings.base_url)
    except ServiceInstallError as e:
        click.echo(f"✗ Error starting service: {e}", err=True)
    except Exception as e:
        click.echo(f"✗ Service installation failed: {e}", err=True)
        click.echo("  You can install later with: obsidian-notes-rag install-service")
    else:
        click.echo("✓ Watcher service installed and started")
        click.echo(f"  Logs: {LOG_DIR}/watcher.log")


def _print_next_steps() -> None:
    click.echo("\nSetup complete! You can now:")
    click.echo('  - Search: obsidian-notes-rag search "your query"')
    click.echo("  - Add to Claude Code:")
    click.echo("      claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve")


@main.command()
def setup():
    """Interactive setup wizard for obsidian-notes-rag."""
    click.echo("\nWelcome to Obsidian RAG setup!\n")
    if not _confirm_overwrite_existing_config():
        return

    provider = _prompt_provider()
    provider_fields = _PROVIDER_PROMPTS[provider]()
    vault_path = _prompt_vault_path()
    if vault_path is None:
        return
    config = Config(provider=provider, vault_path=vault_path, data_path=_prompt_data_path(), **provider_fields)

    click.echo(f"\n✓ Configuration saved to {save_config(config)}")
    # Resolved once: for openai without a saved key each resolution may read the macOS Keychain
    try:
        settings = resolve_embedder_settings(config)
    except EmbedderConfigError as e:  # unreachable via the menus, but never a traceback
        raise click.ClickException(str(e)) from e
    _maybe_initial_index(config, vault_path, settings)
    _maybe_install_service(config, vault_path, settings)
    _print_next_steps()


@main.command()
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.option("--path-filter", default=None, help="Only index files under this path prefix (e.g. 'Daily Notes/')")
@click.pass_context
def index(ctx, clear, path_filter):
    """Index all markdown files in the vault."""
    vault_path = ctx.obj["vault"]
    if not vault_path:
        click.echo("Error: No vault path configured. Run 'obsidian-rag setup' first.", err=True)
        sys.exit(1)
    data_path = ctx.obj["data"]
    config = ctx.obj["config"]
    settings = _embedder_settings(ctx)

    click.echo(f"Indexing vault: {vault_path}")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Provider: {settings.provider}")
    click.echo(f"Model: {settings.model}")

    # Initialize components
    embedder = settings.create()
    store = VectorStore(data_path=data_path)
    indexer = VaultIndexer(vault_path=vault_path, embedder=embedder, config=config.indexer)

    if clear:
        click.echo("Clearing existing index...")
        store.clear()

    # Count files first
    files = list(indexer.iter_markdown_files())
    click.echo(f"Found {len(files)} markdown files")

    if path_filter:
        files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]
        click.echo(f"Filtered to {len(files)} files matching '{path_filter}'")

    chunk_count = _index_with_progress(indexer, store, files)

    embedder.close()

    click.echo(f"\nIndexed {chunk_count} chunks from {len(files)} files")
    click.echo(f"Total documents in store: {store.get_stats()['count']}")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--type", "note_type", default=None, help="Filter by type (daily, note)")
@click.pass_context
def search(ctx, query, limit, note_type):
    """Search notes semantically."""
    data_path = ctx.obj["data"]
    embedder = _embedder_settings(ctx).create()
    store = VectorStore(data_path=data_path)

    # Generate query embedding
    click.echo(f"Searching for: {query}\n")
    query_embedding = embedder.embed(query, task_type="search_query")

    # Build filter
    where = None
    if note_type:
        where = {"type": note_type}

    # Search
    results = store.search(query_embedding, limit=limit, where=where)

    if not results:
        click.echo("No results found.")
        return

    # Display results
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        similarity = 1 - distance  # Cosine distance to similarity

        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Type: {meta.get('type', 'note')} | Similarity: {similarity:.2%}")
        click.echo()

        # Show truncated content
        content = result["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        click.echo(f"    {content}")
        click.echo()

    embedder.close()


@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes")
@click.pass_context
def similar(ctx, note_path, limit):
    """Find notes similar to a given note."""
    data_path = ctx.obj["data"]
    embedder = _embedder_settings(ctx).create()
    store = VectorStore(data_path=data_path)

    click.echo(f"Finding notes similar to: {note_path}\n")
    results = store.get_by_file(note_path)

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    note_content = "\n\n".join(r["content"] for r in results)
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)

    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if not similar_notes:
        click.echo("No similar notes found.")
        embedder.close()
        return

    for i, result in enumerate(similar_notes, 1):
        meta = result["metadata"]
        similarity = 1 - result["distance"]
        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Similarity: {similarity:.2%}")
        content = result["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        click.echo(f"\n    {content}\n")

    embedder.close()


@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes to include")
@click.pass_context
def context(ctx, note_path, limit):
    """Get a note and its related context."""
    data_path = ctx.obj["data"]
    embedder = _embedder_settings(ctx).create()
    store = VectorStore(data_path=data_path)

    click.echo(f"Getting context for: {note_path}\n")
    results = store.get_by_file(note_path)

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    # Display note content
    note_content = "\n\n".join(r["content"] for r in results)
    click.echo(f"{'═' * 60}")
    click.echo(f"Note: {note_path}")
    click.echo(f"{'═' * 60}")
    click.echo(note_content)
    click.echo()

    # Find similar notes
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)
    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if similar_notes:
        click.echo(f"{'─' * 60}")
        click.echo(f"Related Notes ({len(similar_notes)})")
        click.echo(f"{'─' * 60}")
        for i, result in enumerate(similar_notes, 1):
            meta = result["metadata"]
            similarity = 1 - result["distance"]
            click.echo(f"  [{i}] {meta['file_path']}")
            if meta.get("heading"):
                click.echo(f"      Section: {meta['heading']}")
            click.echo(f"      Similarity: {similarity:.2%}")
    else:
        click.echo("No related notes found.")

    embedder.close()


@main.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    data_path = ctx.obj["data"]
    store = VectorStore(data_path=data_path)

    stats = store.get_stats()
    click.echo(f"Collection: {stats['collection']}")
    click.echo(f"Documents: {stats['count']}")
    click.echo(f"Data path: {stats['data_path']}")


@main.command()
@click.option("--debounce", default=2.0, help="Seconds to wait before processing changes")
@click.pass_context
def watch(ctx, debounce):
    """Watch vault for changes and auto-reindex."""
    vault_path = ctx.obj["vault"]
    if not vault_path:
        click.echo("Error: No vault path configured. Run 'obsidian-rag setup' first.", err=True)
        sys.exit(1)
    data_path = ctx.obj["data"]
    settings = _embedder_settings(ctx)

    click.echo(f"Watching vault: {vault_path}")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Provider: {settings.provider}")
    click.echo(f"Model: {settings.model}")
    click.echo(f"Debounce: {debounce}s")
    click.echo("Press Ctrl+C to stop.\n")

    watcher = VaultWatcher(vault_path=vault_path, data_path=data_path, settings=settings, debounce_delay=debounce)
    watcher.run_forever()


@main.command()
def serve():
    """Start the MCP server (for Claude Code integration)."""
    run_server()


# Service management
# TODO: Add Linux systemd support (create .service file in ~/.config/systemd/user/)
# TODO: Add Windows Task Scheduler support (use schtasks or win32api)
SERVICE_LABEL = "com.obsidian-notes-rag.watcher"
PLIST_NAME = f"{SERVICE_LABEL}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
WRAPPER_SCRIPT_DIR = Path.home() / ".local" / "bin"
WRAPPER_SCRIPT_NAME = "obsidian-notes-rag-watcher"
LOG_DIR = Path.home() / "Library" / "Logs" / "obsidian-notes-rag"


def _get_wrapper_script_content() -> str:
    """Generate wrapper script that calls the watcher module."""
    import sys

    python_path = sys.executable
    return f"""#!/bin/bash
# Wrapper script for obsidian-notes-rag watcher service
# This script exists so macOS System Settings shows a descriptive name
# instead of "python3" in Login Items & Extensions.
exec "{python_path}" -m obsidian_rag.watcher "$@"
"""


def _install_wrapper_script() -> Path:
    """Install the wrapper script and return its path."""
    WRAPPER_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME
    wrapper_path.write_text(_get_wrapper_script_content())
    wrapper_path.chmod(0o755)
    return wrapper_path


def _uninstall_wrapper_script():
    """Remove the wrapper script if it exists."""
    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME
    if wrapper_path.exists():
        wrapper_path.unlink()


class ServiceInstallError(RuntimeError):
    """launchctl refused to load the watcher plist; the message is launchctl's stderr."""


def _launchctl(action: str, target: str | Path) -> subprocess.CompletedProcess[str]:
    """Run ``launchctl <action> <target>``; the single seam tests replace with a fake."""
    return subprocess.run(["launchctl", action, str(target)], capture_output=True, text=True)


def _plist_path() -> Path:
    return LAUNCH_AGENTS_DIR / PLIST_NAME


def _install_watcher_service(
    vault_path: str,
    data_path: str,
    provider: str,
    base_url: str | None,  # the chosen provider's server URL; None for providers without one (openai)
    model: str | None = None,  # None: the service follows config.toml; only an explicit --model is pinned
    *,
    echo: Callable[[str], None] = lambda _message: None,
) -> Path:
    """Write the wrapper script and plist, (re)load the launchd service and return the plist path.

    ``echo`` receives progress lines; ``install-service`` prints them, ``setup`` stays quiet.
    """
    plist_path = _plist_path()
    try:
        LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)  # launchd does not create StandardOutPath's directory
        Path(data_path).mkdir(parents=True, exist_ok=True)  # WorkingDirectory must exist before launchd chdirs
        # The wrapper script shows a descriptive name in System Settings > Login Items. It is
        # written before the running service is unloaded so a write failure leaves it running.
        echo(f"Created: {_install_wrapper_script()}")
        if plist_path.exists():
            echo("Unloading existing service...")
            _launchctl("unload", plist_path)
        plist_path.write_text(_get_plist_content(vault_path, data_path, provider, base_url, model))
    except OSError as e:
        raise ServiceInstallError(f"could not write the service files: {e}") from e
    echo(f"Created: {plist_path}")

    result = _launchctl("load", plist_path)
    if result.returncode != 0:
        raise ServiceInstallError(result.stderr)
    return plist_path


def _get_plist_content(vault_path: str, data_path: str, provider: str, base_url: str | None, model: str | None) -> str:
    """Generate launchd plist content using plistlib (safe XML escaping)."""
    import plistlib

    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME

    env_vars: dict[str, str] = {
        "OBSIDIAN_RAG_VAULT": vault_path,
        "OBSIDIAN_RAG_DATA": data_path,
        "OBSIDIAN_RAG_PROVIDER": provider,
    }
    url_env = PROVIDER_URL_ENV.get(provider)
    if url_env and base_url:
        env_vars[url_env] = base_url
    if model:
        env_vars["OBSIDIAN_RAG_MODEL"] = model

    plist_dict: dict = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [str(wrapper_path)],
        "EnvironmentVariables": env_vars,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "StandardOutPath": str(LOG_DIR / "watcher.log"),
        "StandardErrorPath": str(LOG_DIR / "watcher.err"),
        # Nothing in the watcher reads the cwd; use the directory the tool owns rather than the user's home
        "WorkingDirectory": data_path,
    }

    return plistlib.dumps(plist_dict, fmt=plistlib.FMT_XML, sort_keys=False).decode("utf-8")


@main.command("install-service")
@click.pass_context
def install_service(ctx):
    """Install launchd service for auto-start on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    vault_path = ctx.obj["vault"]
    data_path = ctx.obj["data"]
    settings = _embedder_settings(ctx)  # validates the provider and API key before installing the service
    # Only an explicit --model is pinned in the plist; otherwise the service follows config.toml
    model = ctx.obj["overrides"]["model"]

    try:
        _install_watcher_service(vault_path, data_path, settings.provider, settings.base_url, model, echo=click.echo)
    except ServiceInstallError as e:
        click.echo(f"Error loading service: {e}", err=True)
        sys.exit(1)

    click.echo("Service installed and started.")
    click.echo(f"Logs: {LOG_DIR}/watcher.log")
    click.echo(f"Errors: {LOG_DIR}/watcher.err")


@main.command("uninstall-service")
def uninstall_service():
    """Uninstall launchd service on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    plist_path = _plist_path()

    if not plist_path.exists():
        click.echo("Service not installed.")
        return

    result = _launchctl("unload", plist_path)
    if result.returncode != 0:
        click.echo(f"Warning: Error unloading service: {result.stderr}", err=True)

    # Remove plist
    plist_path.unlink()

    # Remove wrapper script
    _uninstall_wrapper_script()

    click.echo("Service uninstalled.")


@main.command("service-status")
def service_status():
    """Check launchd service status on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    if not _plist_path().exists():
        click.echo("Service not installed.")
        return

    result = _launchctl("list", SERVICE_LABEL)

    if result.returncode == 0:
        click.echo("Service is running.")
        click.echo(result.stdout)
    else:
        click.echo("Service is installed but not running.")


if __name__ == "__main__":
    main()
