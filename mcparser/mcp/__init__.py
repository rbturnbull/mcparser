"""MCP server over CoNLL-U corpora (see :func:`build_server`)."""

from .config import AlignmentSpec, CorpusSpec, SourceSpec, claude_desktop_config, load_store
from .server import build_server

__all__ = ["AlignmentSpec", "CorpusSpec", "SourceSpec", "build_server", "claude_desktop_config", "load_store"]
