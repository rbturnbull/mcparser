"""Source specifications and client configuration for the MCP server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..corpus import CorpusStore, SourceMode
from ..corpus.errors import CorpusError

MODES = {mode.value for mode in SourceMode}


class SourceSpec(BaseModel):
    """A source given on the command line as ``[NAME=]PATH[:MODE[:CONFIDENCE]]``.

    ``MODE`` is ``ro`` (default) or ``rw``. ``CONFIDENCE`` is the default
    confidence for cells without a ``Conf`` score. ``NAME`` defaults to the file
    stem. Examples: ``train.conllu``, ``pred=out/pred.conllu:rw``,
    ``silver=silver.conllu:ro:0.8``.
    """

    name: str
    path: Path
    mode: SourceMode = SourceMode.READ_ONLY
    default_confidence: float | None = None

    @classmethod
    def parse(cls, spec: str) -> SourceSpec:
        name, sep, rest = spec.partition("=")
        if not sep:
            name, rest = "", spec
        parts = rest.rsplit(":", 2)
        confidence = None
        if len(parts) == 3 and parts[1] in MODES:
            path, mode = parts[0], parts[1]
            try:
                confidence = float(parts[2])
            except ValueError:
                raise CorpusError(f"Invalid confidence {parts[2]!r} in source spec {spec!r}.") from None
        elif parts[-1] in MODES:
            path, mode = rest.rsplit(":", 1)
        else:
            path, mode = rest, SourceMode.READ_ONLY.value
        if not path:
            raise CorpusError(f"Source spec {spec!r} has no path.")
        resolved = Path(path).expanduser().resolve()
        return cls(name=name or resolved.stem, path=resolved, mode=SourceMode(mode), default_confidence=confidence)

    def to_arg(self) -> str:
        spec = f"{self.name}={self.path}:{self.mode.value}"
        if self.default_confidence is not None:
            spec += f":{self.default_confidence}"
        return spec


def load_store(specs: list[SourceSpec], autosave: bool = True, root_key: str = "Root") -> CorpusStore:
    store = CorpusStore(autosave=autosave)
    for spec in specs:
        store.add_source(spec.name, spec.path, mode=spec.mode, default_confidence=spec.default_confidence, root_key=root_key)
    return store


def claude_desktop_config(
    specs: list[SourceSpec], name: str = "conllu", read_only: bool = False, root_key: str = "Root"
) -> dict[str, Any]:
    """The ``mcpServers`` entry for Claude Desktop's ``claude_desktop_config.json``."""
    args = ["-m", "llmparser", "mcp", "serve", "--name", name]
    for spec in specs:
        args += ["--source", spec.to_arg()]
    if read_only:
        args.append("--read-only")
    if root_key != "Root":
        args += ["--root-key", root_key]
    return {"mcpServers": {name: {"command": sys.executable, "args": args}}}
