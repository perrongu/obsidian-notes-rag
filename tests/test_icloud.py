"""Tests for iCloud Drive helpers: dataless detection, EDEADLK classification, download request."""

import errno
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from obsidian_rag import icloud

P = Path("/vault/note.md")


class TestIsDataless:
    def test_true_when_dataless_flag_set(self):
        with patch("obsidian_rag.icloud.os.stat", return_value=SimpleNamespace(st_flags=icloud.SF_DATALESS)):
            assert icloud.is_dataless(P) is True

    def test_false_when_flag_not_set(self):
        with patch("obsidian_rag.icloud.os.stat", return_value=SimpleNamespace(st_flags=0)):
            assert icloud.is_dataless(P) is False

    def test_false_when_platform_has_no_st_flags(self):
        with patch("obsidian_rag.icloud.os.stat", return_value=SimpleNamespace()):
            assert icloud.is_dataless(P) is False

    def test_false_when_stat_fails(self):
        with patch("obsidian_rag.icloud.os.stat", side_effect=FileNotFoundError):
            assert icloud.is_dataless(P) is False


class TestIsDatalessError:
    def test_edeadlk_oserror_is_dataless_error(self):
        exc = OSError(errno.EDEADLK, "Resource deadlock avoided")
        assert icloud.is_dataless_error(exc) is True

    def test_other_oserror_is_not(self):
        exc = OSError(errno.ENOENT, "No such file")
        assert icloud.is_dataless_error(exc) is False

    def test_non_oserror_is_not(self):
        assert icloud.is_dataless_error(ValueError("x")) is False


class TestRequestDownload:
    def test_invokes_brctl_download_with_path(self):
        with patch("obsidian_rag.icloud.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            assert icloud.request_download(P) is True
        run.assert_called_once()
        assert run.call_args.args[0] == ["brctl", "download", str(P)]

    def test_false_when_brctl_missing(self):
        with patch("obsidian_rag.icloud.subprocess.run", side_effect=FileNotFoundError):
            assert icloud.request_download(P) is False

    def test_false_when_brctl_fails(self):
        with patch("obsidian_rag.icloud.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            assert icloud.request_download(P) is False

    def test_false_when_brctl_times_out(self):
        with patch("obsidian_rag.icloud.subprocess.run", side_effect=subprocess.TimeoutExpired("brctl", 1)):
            assert icloud.request_download(P) is False
