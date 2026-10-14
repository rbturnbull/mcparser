"""Word alignments between two sources (e.g. a translation and its original).

An alignment set pairs sentences with the same ``sent_id`` in two sources and
holds word links between them. Links are written in the Ugarit style
``1-3S 2-5P`` (word id in source A, word id in source B, S = sure, P =
probable); many-to-many alignments are several links. Each link may carry a
confidence, as cells do.

Sets are stored as JSON Lines, one line per sentence pair::

    {"sent_id": "B10K1V1", "links": [{"source": "1", "target": "1", "sure": true, "confidence": 0.95}]}

Like sources, a set is read-only (trusted, e.g. gold alignments) or writable,
edits are saved immediately, and every change is recorded in a
``<file>.history.jsonl`` sidecar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .confidence import check_confidence
from .errors import ConflictError, EditError, NotFoundError, ReadOnlyError
from .models import Edit
from .source import CorpusSource, SourceMode, atomic_write

LINK_PATTERN = re.compile(r"^(\d+)-(\d+)([SP]?)$")


class Link(BaseModel):
    source: str = Field(description="Word id in source A.")
    target: str = Field(description="Word id in source B.")
    sure: bool = Field(True, description="S (sure) or P (probable), as in Ugarit.")
    confidence: float | None = Field(None, ge=0, le=1)

    @property
    def key(self) -> str:
        return f"{self.source}-{self.target}"

    def ugarit(self) -> str:
        return f"{self.key}{'S' if self.sure else 'P'}"


def parse_ugarit(text: str, confidence: float | None = None) -> list[Link]:
    """Parse ``"1-3S 2-5P 4-4"`` (no suffix means sure) into links."""
    links = []
    for item in text.split():
        match = LINK_PATTERN.match(item)
        if match is None:
            raise EditError(f"Invalid link {item!r}; expected SOURCE-TARGET with an optional S or P, e.g. 3-5S.")
        source, target, kind = match.groups()
        links.append(Link(source=source, target=target, sure=kind != "P", confidence=confidence))
    return links


def format_ugarit(links: list[Link]) -> str:
    return " ".join(link.ugarit() for link in links)


class AlignmentInfo(BaseModel):
    name: str
    source_a: str
    source_b: str
    path: str
    mode: str
    default_confidence: float
    sentences: int = Field(description="Sentence pairs with at least one link.")
    links: int


class AlignmentSet:
    """Links between the sentences of two sources, persisted as JSON Lines."""

    def __init__(
        self,
        name: str,
        source_a: CorpusSource,
        source_b: CorpusSource,
        path: Path | str,
        mode: SourceMode | str = SourceMode.READ_WRITE,
        default_confidence: float | None = None,
    ):
        self.name = name
        self.source_a = source_a
        self.source_b = source_b
        self.path = Path(path).expanduser().resolve()
        self.mode = SourceMode(mode)
        if default_confidence is None:
            default_confidence = 1.0 if self.mode is SourceMode.READ_ONLY else 0.0
        self.default_confidence = check_confidence(default_confidence)
        self.load()

    @property
    def writable(self) -> bool:
        return self.mode is SourceMode.READ_WRITE

    @property
    def history_path(self) -> Path:
        return self.path.with_name(self.path.name + ".history.jsonl")

    def load(self) -> None:
        """(Re)load the file; a missing writable file starts empty and is created on first save."""
        self.pairs: dict[str, list[Link]] = {}
        self.history: list[Edit] = []
        self._unsaved_history: list[Edit] = []
        if not self.path.is_file():
            if not self.writable:
                raise NotFoundError(f"Alignment file not found: {self.path}")
            self._mtime: int | None = None
            return
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                self.pairs[str(data["sent_id"])] = [Link.model_validate(link) for link in data.get("links", [])]
            except (ValueError, KeyError, TypeError) as error:
                raise EditError(f"{self.path}, line {number}: {error}") from None
        self._mtime = self.path.stat().st_mtime_ns
        if self.writable and self.history_path.is_file():
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            self.history = [Edit.model_validate_json(line) for line in lines if line.strip()]

    def require_writable(self) -> None:
        if not self.writable:
            raise ReadOnlyError(f"Alignment {self.name!r} is read-only.")

    def confidence(self, link: Link) -> float:
        return self.default_confidence if link.confidence is None else link.confidence

    def links(self, sent_id: str) -> list[Link]:
        return self.pairs.get(sent_id, [])

    def record(self, edit: Edit) -> None:
        edit.seq = len(self.history) + 1
        self.history.append(edit)
        self._unsaved_history.append(edit)

    def save(self) -> None:
        self.require_writable()
        current = self.path.stat().st_mtime_ns if self.path.exists() else None
        if current != self._mtime:
            raise ConflictError(f"{self.path} was modified by another process; reload before editing.")
        order = {sentence.sent_id: i for i, sentence in enumerate(self.source_a.sentences)}
        lines = [
            json.dumps(
                {"sent_id": sent_id, "links": [link.model_dump(exclude_none=True) for link in links]},
                ensure_ascii=False,
            )
            for sent_id, links in sorted(self.pairs.items(), key=lambda item: order.get(item[0], len(order)))
            if links
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self.path, "".join(f"{line}\n" for line in lines))
        self._mtime = self.path.stat().st_mtime_ns
        if self._unsaved_history:
            with self.history_path.open("a", encoding="utf-8") as stream:
                for edit in self._unsaved_history:
                    stream.write(edit.model_dump_json() + "\n")
            self._unsaved_history = []

    def info(self) -> AlignmentInfo:
        return AlignmentInfo(
            name=self.name,
            source_a=self.source_a.name,
            source_b=self.source_b.name,
            path=str(self.path),
            mode=self.mode.value,
            default_confidence=self.default_confidence,
            sentences=sum(1 for links in self.pairs.values() if links),
            links=sum(len(links) for links in self.pairs.values()),
        )
