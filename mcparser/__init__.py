"""LLM agents for dependency parsing over CoNLL-U files."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcparser")
except PackageNotFoundError:  # pragma: no cover - running from a source tree that is not installed
    __version__ = "0.0.0"
