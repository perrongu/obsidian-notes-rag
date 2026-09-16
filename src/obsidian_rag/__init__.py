"""Obsidian Memory - Vector store for AI-assisted note management."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("obsidian-notes-rag")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = ""
