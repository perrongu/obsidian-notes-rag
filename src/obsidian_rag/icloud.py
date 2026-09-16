"""Helpers for vaults stored in iCloud Drive.

iCloud can evict a file's content locally, leaving a "dataless" placeholder. Reading such a
file from a background daemon fails with EDEADLK ("Resource deadlock avoided") instead of
triggering a download. These helpers detect that state and ask iCloud to materialize the file.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# BSD stat flag set on evicted iCloud files (SF_DATALESS in <sys/stat.h>).
SF_DATALESS = 0x40000000
BRCTL_TIMEOUT = 10.0  # seconds


class FileNotMaterializedError(OSError):
    """Raised when a file exists only as an iCloud placeholder; retry after download."""

    def __init__(self, path: Path):
        super().__init__(errno.EDEADLK, "iCloud file not downloaded locally", str(path))
        self.path = path


def is_dataless(path: Path) -> bool:
    """True if the file is an iCloud placeholder whose content is not on disk."""
    try:
        flags = getattr(os.stat(path), "st_flags", 0)
    except OSError:
        return False
    return bool(flags & SF_DATALESS)


def is_dataless_error(exc: Exception) -> bool:
    """True if an exception is the EDEADLK error macOS raises when reading a dataless file."""
    return isinstance(exc, OSError) and exc.errno == errno.EDEADLK


def request_download(path: Path) -> bool:
    """Ask iCloud to download a file's content (best effort, non-blocking for the caller)."""
    try:
        result = subprocess.run(
            ["brctl", "download", str(path)],
            check=False,
            capture_output=True,
            timeout=BRCTL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.warning("brctl not available; cannot request iCloud download of %s", path)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("brctl download timed out for %s", path)
        return False

    if result.returncode != 0:
        logger.warning("brctl download failed for %s (exit %d)", path, result.returncode)
        return False

    logger.info("Requested iCloud download of %s", path)
    return True
