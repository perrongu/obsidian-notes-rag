"""Retry queue with per-item attempt tracking and exponential backoff.

Design goals:
- Attempt counts live in the queue, keyed by path, so no caller can reset them by accident.
- Items become due only after a backoff delay; one call to `pop_due` returns a snapshot,
  so a persistently failing item is tried at most once per processing cycle.
- Once `max_retries` failures are reached the item is dropped and `on_give_up` fires once.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 60.0  # seconds; doubles after each failure (60, 120, 240, ...)

IndexFn = Callable[[Path], None]
IsPermanentFn = Callable[[Exception], bool]
GiveUpFn = Callable[[Path], None]


@dataclass(frozen=True)
class RetryEntry:
    """Immutable bookkeeping for one queued path."""

    attempts: int
    next_attempt_at: float
    queued: bool


class RetryQueue:
    """Queue for files that failed to index, with bounded exponential backoff."""

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY,
        on_give_up: GiveUpFn | None = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._on_give_up = on_give_up
        self._entries: dict[Path, RetryEntry] = {}
        self._lock = threading.Lock()

    def add(self, path: Path, now: float) -> None:
        """Queue a path for retry. Idempotent; never resets an existing attempt count."""
        with self._lock:
            existing = self._entries.get(path)
            if existing is None:
                self._entries[path] = RetryEntry(attempts=0, next_attempt_at=now, queued=True)
                logger.info("Added to retry queue: %s", path)
            elif not existing.queued:
                self._entries[path] = replace(existing, queued=True, next_attempt_at=now)

    def pop_due(self, now: float) -> list[Path]:
        """Return a snapshot of paths whose backoff has elapsed, marking them in-flight."""
        with self._lock:
            due = [path for path, entry in self._entries.items() if entry.queued and entry.next_attempt_at <= now]
            for path in due:
                self._entries[path] = replace(self._entries[path], queued=False)
            return due

    def mark_failed(self, path: Path, now: float) -> bool:
        """Record a failed attempt. Returns True if the path will be retried, False if given up."""
        with self._lock:
            entry = self._entries.get(path) or RetryEntry(attempts=0, next_attempt_at=now, queued=False)
            attempts = entry.attempts + 1
            if attempts >= self.max_retries:
                self._entries.pop(path, None)
                give_up = True
            else:
                delay = self.base_delay * (2 ** (attempts - 1))
                self._entries[path] = RetryEntry(attempts=attempts, next_attempt_at=now + delay, queued=True)
                logger.info("Re-queued %s (attempt %d/%d, next try in %.0fs)", path, attempts, self.max_retries, delay)
                give_up = False

        if give_up:
            logger.error("Max retries (%d) exceeded for %s, giving up", self.max_retries, path)
            self._notify_give_up(path)
        return not give_up

    def discard(self, path: Path) -> None:
        """Forget a path entirely (success or permanent error)."""
        with self._lock:
            self._entries.pop(path, None)

    def attempts(self, path: Path) -> int:
        """Number of failed attempts recorded for a path (0 if unknown)."""
        with self._lock:
            entry = self._entries.get(path)
            return entry.attempts if entry else 0

    def is_empty(self) -> bool:
        """True when nothing is queued or in flight."""
        with self._lock:
            return not self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _notify_give_up(self, path: Path) -> None:
        if self._on_give_up is None:
            return
        try:
            self._on_give_up(path)
        except Exception as e:
            logger.warning("on_give_up callback failed for %s: %s", path, e)


def process_due(queue: RetryQueue, index_fn: IndexFn, is_permanent: IsPermanentFn, now: float) -> None:
    """Run one retry cycle: try each due path exactly once, then return.

    A path that fails again is rescheduled with backoff (or dropped after max retries),
    so this function never spins on a persistently failing file.
    """
    for path in queue.pop_due(now):
        try:
            index_fn(path)
        except Exception as e:
            _handle_retry_failure(queue, path, e, is_permanent, now)
        else:
            queue.discard(path)


def _handle_retry_failure(
    queue: RetryQueue, path: Path, error: Exception, is_permanent: IsPermanentFn, now: float
) -> None:
    logger.error("Retry failed for %s: %s", path, error)
    if is_permanent(error):
        logger.warning("Permanent error for %s, dropping from retry queue", path)
        queue.discard(path)
        return
    queue.mark_failed(path, now)
