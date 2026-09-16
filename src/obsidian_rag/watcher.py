"""File watcher daemon for auto-indexing Obsidian notes."""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import setproctitle

# Log rotation settings
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 3  # Keep 3 rotated files

import httpx
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from . import icloud
from .config import load_config
from .embedders import EmbedderConfigError, EmbedderSettings, resolve_embedder_settings
from .indexer import Embedder, IndexerConfig, VaultIndexer
from .retry_queue import MAX_RETRIES, RETRY_BASE_DELAY, RetryQueue, process_due
from .store import VectorStore

HEALTH_CHECK_INTERVAL = 60  # seconds

__all__ = ["MAX_RETRIES", "RETRY_BASE_DELAY", "RetryQueue", "VaultWatcher", "NoteEventHandler"]

# Lazy-loaded config — avoids module-level side effects (subprocess calls, file I/O)
_config = None


def _get_config():
    """Get or create the module-level config (lazy)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


logger = logging.getLogger(__name__)


def check_ollama_health(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def send_notification(title: str, message: str):
    """Send a macOS notification (uses argv passing to avoid AppleScript injection)."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run",
                "--",
                title,
                message,
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass  # Notifications are best-effort


def _notify_index_failure(path: Path):
    """Notify the user that a file was abandoned after repeated indexing failures."""
    send_notification("Obsidian RAG Error", f"Failed to index: {path.name}")


class DebouncedHandler:
    """Debounces file events to avoid processing rapid successive changes."""

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def debounce(self, key: str, callback, *args):
        """Schedule a callback after delay, canceling any pending call for the same key."""
        with self._lock:
            if key in self._timers:
                self._timers[key].cancel()

            timer = threading.Timer(self.delay, self._execute, args=(key, callback, args))
            self._timers[key] = timer
            timer.start()

    def _execute(self, key: str, callback, args):
        """Execute the callback and clean up."""
        with self._lock:
            self._timers.pop(key, None)
        try:
            callback(*args)
        except Exception as e:
            logger.error("Error in debounced callback for %s: %s", key, e)

    def cancel_all(self):
        """Cancel all pending timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class NoteEventHandler(FileSystemEventHandler):
    """Handles file system events for Obsidian notes."""

    def __init__(
        self,
        vault_path: Path,
        embedder: Embedder,
        store: VectorStore,
        debounce_delay: float = 2.0,
        exclude_patterns: list[str] | None = None,
        retry_queue: RetryQueue | None = None,
        indexer_config: IndexerConfig | None = None,
    ):
        super().__init__()
        self.vault_path = vault_path
        self.embedder = embedder
        self.store = store
        self.debouncer = DebouncedHandler(delay=debounce_delay)
        self.retry_queue = retry_queue
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**",
            ".venv/**",
            "node_modules/**",
            "__pycache__/**",
            "*.egg-info/**",
            "build/**",
            "dist/**",
            ".git/**",
        ]
        self.indexer = VaultIndexer(
            vault_path=vault_path,
            embedder=embedder,
            exclude_patterns=self.exclude_patterns,
            config=indexer_config,
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on exclude patterns."""
        if not path.suffix == ".md":
            return True

        # Ignore Obsidian temp/recovery files (pattern: .!NNNNN!filename.md)
        if path.name.startswith(".!") and "!" in path.name[2:]:
            return True

        try:
            rel_path = path.relative_to(self.vault_path)
        except ValueError:
            return True

        for pattern in self.exclude_patterns:
            if rel_path.match(pattern):
                return True

        return False

    def _get_relative_path(self, path: Path) -> str:
        """Get the path relative to the vault root."""
        return str(path.relative_to(self.vault_path))

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """Return True if the error should not be retried."""
        # Dimension mismatch (e.g. model changed but DB has old dimensions)
        if isinstance(exc, sqlite3.OperationalError) and "dimension mismatch" in str(exc).lower():
            return True
        try:
            from openai import BadRequestError

            if isinstance(exc, BadRequestError):
                return True
        except ImportError:
            pass
        # httpx 4xx (except 429) are permanent
        try:
            from httpx import HTTPStatusError

            if (
                isinstance(exc, HTTPStatusError)
                and 400 <= exc.response.status_code < 500
                and exc.response.status_code != 429
            ):
                return True
        except ImportError:
            pass
        return False

    def _index_file(self, path: Path):
        """Index a file from a filesystem event; failures are queued for retry, never raised."""
        try:
            self._try_index(path)
        except Exception as e:
            rel_path = self._get_relative_path(path)
            logger.error("Error indexing %s: %s", rel_path, e)
            if self._is_permanent_error(e):
                logger.warning("Permanent error for %s, skipping retry", rel_path)
            elif self.retry_queue is not None:
                self.retry_queue.add(path, now=time.monotonic())
        else:
            if self.retry_queue is not None:
                self.retry_queue.discard(path)

    def _try_index(self, path: Path):
        """Index or re-index a single file, raising on failure (used by the retry loop)."""
        if self._should_ignore(path) or not path.exists():
            return

        rel_path = self._get_relative_path(path)
        if icloud.is_dataless(path):
            logger.info("File not downloaded from iCloud yet, requesting: %s", rel_path)
            icloud.request_download(path)
            raise icloud.FileNotMaterializedError(path)

        logger.info("Indexing: %s", rel_path)
        try:
            self.store.delete_by_file(rel_path)
            results = self.indexer.index_file(path)
        except Exception as e:
            if icloud.is_dataless_error(e):
                icloud.request_download(path)
            raise

        if results:
            chunks, embeddings = zip(*results, strict=False)
            self.store.upsert_batch(list(chunks), list(embeddings))
            logger.info("Indexed %d chunks from %s", len(chunks), rel_path)

    def _delete_file(self, path: Path):
        """Remove a file from the index."""
        if not path.suffix == ".md":
            return

        try:
            rel_path = self._get_relative_path(path)
        except ValueError:
            return

        logger.info("Removing from index: %s", rel_path)
        try:
            self.store.delete_by_file(rel_path)
        except Exception as e:
            logger.error("Error removing %s: %s", rel_path, e)

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        self.debouncer.debounce(str(path), self._index_file, path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        self.debouncer.debounce(str(path), self._index_file, path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        # No need to debounce deletes
        self._delete_file(path)

    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if event.is_directory:
            return

        # Delete old location
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        old_path = Path(src_path)
        self._delete_file(old_path)

        # Index new location
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            if isinstance(dest_path, bytes):
                dest_path = dest_path.decode()
            new_path = Path(dest_path)
            self.debouncer.debounce(str(new_path), self._index_file, new_path)

    def shutdown(self):
        """Clean up resources."""
        self.debouncer.cancel_all()


class VaultWatcher:
    """Watches an Obsidian vault for changes and auto-indexes."""

    def __init__(
        self,
        vault_path: str | None = None,
        data_path: str | None = None,
        settings: EmbedderSettings | None = None,
        debounce_delay: float | None = None,
    ):
        config = _get_config()

        vault_path = vault_path or config.vault_path or os.environ.get("OBSIDIAN_RAG_VAULT", "")
        data_path = data_path or config.get_data_path()
        debounce_delay = (
            debounce_delay if debounce_delay is not None else float(os.environ.get("OBSIDIAN_RAG_DEBOUNCE", "2.0"))
        )
        # Resolved once, here or by the caller; the API key is passed directly, never via os.environ
        settings = settings or resolve_embedder_settings(config)

        self.vault_path = Path(vault_path)
        self.settings = settings
        self.provider = settings.provider
        self.ollama_url = settings.base_url or config.ollama_url

        # Health check for Ollama before starting
        if self.provider == "ollama" and not check_ollama_health(self.ollama_url):
            logger.warning("Ollama is not running! Waiting for it to start...")
            send_notification("Obsidian RAG", "Waiting for Ollama to start...")
            self._wait_for_ollama(self.ollama_url)

        self.embedder = settings.create()
        self.store = VectorStore(data_path=data_path)
        self.debounce_delay = debounce_delay
        self.retry_queue = RetryQueue(
            max_retries=MAX_RETRIES,
            base_delay=RETRY_BASE_DELAY,
            on_give_up=_notify_index_failure,
        )

        self._observer: BaseObserver | None = None
        self._handler: NoteEventHandler | None = None
        self._running = False
        self._health_thread: threading.Thread | None = None

    def _wait_for_ollama(self, ollama_url: str, timeout: int = 300):
        """Wait for Ollama to become available."""
        start = time.time()
        while time.time() - start < timeout:
            if check_ollama_health(ollama_url):
                logger.info("Ollama is now available!")
                return
            time.sleep(5)
        raise RuntimeError(f"Ollama did not start within {timeout} seconds")

    def _health_check_loop(self):
        """Periodically check embedding service health and process retry queue."""
        while self._running:
            time.sleep(HEALTH_CHECK_INTERVAL)

            if not self._running:
                break

            # Check health
            if self.provider == "ollama" and not check_ollama_health(self.ollama_url):
                logger.warning("Ollama health check failed!")
                send_notification("Obsidian RAG", "Ollama is not responding")
                continue

            # Process retry queue: each due file is tried at most once per cycle
            if self._handler:
                process_due(
                    self.retry_queue,
                    self._handler._try_index,
                    NoteEventHandler._is_permanent_error,
                    now=time.monotonic(),
                )

    def start(self):
        """Start watching the vault."""
        if self._running:
            return

        logger.info("Starting watcher for vault: %s", self.vault_path)
        logger.info("Debounce delay: %ss", self.debounce_delay)
        logger.info("Provider: %s", self.provider)

        self._handler = NoteEventHandler(
            vault_path=self.vault_path,
            embedder=self.embedder,
            store=self.store,
            debounce_delay=self.debounce_delay,
            retry_queue=self.retry_queue,
            indexer_config=_get_config().indexer,
        )

        observer = Observer()
        observer.schedule(self._handler, str(self.vault_path), recursive=True)
        observer.start()
        self._observer = observer
        self._running = True

        # Start health check thread
        self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_thread.start()

        logger.info("Watcher started. Press Ctrl+C to stop.")

    def stop(self):
        """Stop watching the vault."""
        if not self._running:
            return

        logger.info("Stopping watcher...")

        if self._handler:
            self._handler.shutdown()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

        self.embedder.close()
        self._running = False

        logger.info("Watcher stopped.")

    def run_forever(self):
        """Run the watcher until interrupted."""
        self.start()

        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("Received signal %s", signum)
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def _setup_logging():
    """Configure logging with rotation when running as a service."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured — avoid duplicate handlers

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Check if we're running as a service (stderr redirected to file)
    log_dir = Path.home() / "Library" / "Logs" / "obsidian-notes-rag"

    if not sys.stderr.isatty():
        log_dir.mkdir(parents=True, exist_ok=True)
        # Running as service - use rotating file handler
        log_file = log_dir / "watcher.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        handler.setFormatter(logging.Formatter(log_format, date_format))

        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        # Redirect stderr to rotating log to prevent unbounded watcher.err growth
        stderr_log = log_dir / "watcher.err.log"
        stderr_handler = logging.handlers.RotatingFileHandler(
            stderr_log,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        stderr_handler.setFormatter(logging.Formatter(log_format, date_format))
        stderr_handler.setLevel(logging.WARNING)
        root_logger.addHandler(stderr_handler)
    else:
        # Interactive mode - simple console logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
        )


def run_watcher(
    vault_path: str | None = None,
    data_path: str | None = None,
    provider: str | None = None,
    ollama_url: str | None = None,
    lmstudio_url: str | None = None,
    model: str | None = None,
    debounce: float | None = None,
):
    """Run the vault watcher (entry point for CLI)."""
    # Set process title for Activity Monitor visibility
    setproctitle.setproctitle("obsidian-notes-rag")

    # Configure logging with rotation support
    _setup_logging()

    try:
        settings = resolve_embedder_settings(
            _get_config(), provider=provider, model=model, ollama_url=ollama_url, lmstudio_url=lmstudio_url
        )
    except EmbedderConfigError as e:
        logger.error("Cannot start watcher: %s", e)
        raise SystemExit(1) from e

    watcher = VaultWatcher(
        vault_path=vault_path,
        data_path=data_path,
        settings=settings,
        debounce_delay=debounce or float(os.environ.get("OBSIDIAN_RAG_DEBOUNCE", "2.0")),
    )
    watcher.run_forever()


if __name__ == "__main__":
    run_watcher()
