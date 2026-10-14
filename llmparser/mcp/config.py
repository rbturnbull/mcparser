"""Source specifications and client configuration for the MCP server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..corpus import CorpusStore, SourceMode
from ..corpus.errors import CorpusError

MODES = {mode.value for mode in SourceMode}


def split_file_spec(text: str, spec: str) -> tuple[Path, SourceMode | None, float | None]:
    """``PATH[:MODE[:CONFIDENCE]]`` as (resolved path, mode or None, confidence or None)."""
    parts = text.rsplit(":", 2)
    mode = confidence = None
    if len(parts) == 3 and parts[1] in MODES:
        path, mode = parts[0], parts[1]
        try:
            confidence = float(parts[2])
        except ValueError:
            raise CorpusError(f"Invalid confidence {parts[2]!r} in {spec!r}.") from None
    elif parts[-1] in MODES:
        path, mode = text.rsplit(":", 1)
    else:
        path = text
    if not path:
        raise CorpusError(f"{spec!r} has no path.")
    return Path(path).expanduser().resolve(), SourceMode(mode) if mode else None, confidence


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
        path, mode, confidence = split_file_spec(rest, spec)
        return cls(name=name or path.stem, path=path, mode=mode or SourceMode.READ_ONLY, default_confidence=confidence)

    def to_arg(self) -> str:
        spec = f"{self.name}={self.path}:{self.mode.value}"
        if self.default_confidence is not None:
            spec += f":{self.default_confidence}"
        return spec


class AlignmentSpec(BaseModel):
    """An alignment set given as ``NAME=SOURCE_A,SOURCE_B,PATH[:ro|rw[:CONFIDENCE]]``.

    ``SOURCE_A`` and ``SOURCE_B`` are names of ``--source`` sources. The mode
    defaults to ``rw``; a writable file that does not exist yet is created.
    Example: ``verses=arabic,greek,alignments.jsonl``.
    """

    name: str
    source_a: str
    source_b: str
    path: Path
    mode: SourceMode = SourceMode.READ_WRITE
    default_confidence: float | None = None

    @classmethod
    def parse(cls, spec: str) -> AlignmentSpec:
        name, sep, rest = spec.partition("=")
        parts = rest.split(",", 2)
        if not sep or not name or len(parts) != 3 or not all(parts):
            raise CorpusError(f"Alignment spec {spec!r} should be NAME=SOURCE_A,SOURCE_B,PATH[:ro|rw[:CONFIDENCE]].")
        source_a, source_b, file_spec = parts
        path, mode, confidence = split_file_spec(file_spec, spec)
        return cls(
            name=name, source_a=source_a, source_b=source_b, path=path,
            mode=mode or SourceMode.READ_WRITE, default_confidence=confidence,
        )  # fmt: skip

    def to_arg(self) -> str:
        spec = f"{self.name}={self.source_a},{self.source_b},{self.path}:{self.mode.value}"
        if self.default_confidence is not None:
            spec += f":{self.default_confidence}"
        return spec


def load_store(
    specs: list[SourceSpec],
    autosave: bool = True,
    root_key: str = "Root",
    alignments: list[AlignmentSpec] | None = None,
) -> CorpusStore:
    store = CorpusStore(autosave=autosave)
    for spec in specs:
        store.add_source(spec.name, spec.path, mode=spec.mode, default_confidence=spec.default_confidence, root_key=root_key)
    for alignment in alignments or []:
        store.add_alignment(
            alignment.name, alignment.source_a, alignment.source_b, alignment.path,
            mode=alignment.mode, default_confidence=alignment.default_confidence,
        )  # fmt: skip
    return store


def claude_desktop_config(
    specs: list[SourceSpec],
    name: str = "conllu",
    read_only: bool = False,
    root_key: str = "Root",
    alignments: list[AlignmentSpec] | None = None,
) -> dict[str, Any]:
    """The ``mcpServers`` entry for Claude Desktop's ``claude_desktop_config.json``."""
    args = ["-m", "llmparser", "mcp", "serve", "--name", name]
    for spec in specs:
        args += ["--source", spec.to_arg()]
    for alignment in alignments or []:
        args += ["--alignment", alignment.to_arg()]
    if read_only:
        args.append("--read-only")
    if root_key != "Root":
        args += ["--root-key", root_key]
    return {"mcpServers": {name: {"command": sys.executable, "args": args}}}
