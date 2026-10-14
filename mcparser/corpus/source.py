"""One CoNLL-U file loaded into memory, read-only or writable."""

from __future__ import annotations

import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path

from .confidence import check_confidence, read_confidence
from .conllu import Sentence, Token, format_conllu, parse_conllu
from .errors import ConflictError, CorpusError, NotFoundError, ReadOnlyError
from .index import SourceIndex
from .models import Edit, Flag, SourceInfo

SCORED_COLUMNS = ("form", "lemma", "upos", "xpos", "feats", "head", "deprel")
"""Columns that count towards a sentence's confidence (DEPS is optional in UD)."""

REQUIRED_COLUMNS = ("lemma", "upos", "head", "deprel")
"""Columns where ``_`` means "not annotated yet" (in XPOS, FEATS and DEPS it can be a real value)."""


class SourceMode(StrEnum):
    READ_ONLY = "ro"
    READ_WRITE = "rw"


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file and an atomic rename."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


class CorpusSource:
    """A CoNLL-U file plus, for writable sources, its review flags and history.

    Writable sources keep two sidecar files next to the CoNLL-U file:
    ``<file>.review.json`` (flags) and ``<file>.history.jsonl`` (edit log).
    """

    def __init__(
        self,
        name: str,
        path: Path | str,
        mode: SourceMode | str = SourceMode.READ_ONLY,
        default_confidence: float | None = None,
        root_key: str = "Root",
    ):
        self.name = name
        self.path = Path(path).expanduser().resolve()
        self.mode = SourceMode(mode)
        if default_confidence is None:
            default_confidence = 1.0 if self.mode is SourceMode.READ_ONLY else 0.0
        self.default_confidence = check_confidence(default_confidence)
        self.root_key = root_key
        self.load()

    @property
    def writable(self) -> bool:
        return self.mode is SourceMode.READ_WRITE

    @property
    def review_path(self) -> Path:
        return self.path.with_name(self.path.name + ".review.json")

    @property
    def history_path(self) -> Path:
        return self.path.with_name(self.path.name + ".history.jsonl")

    def load(self) -> None:
        """(Re)load the file and sidecars, discarding unsaved changes."""
        if not self.path.is_file():
            raise NotFoundError(f"CoNLL-U file not found: {self.path}")
        sentences = parse_conllu(self.path.read_text(encoding="utf-8"))
        positions: dict[str, int] = {}
        for position, sentence in enumerate(sentences):
            if sentence.sent_id is None:
                sentence.comments.insert(0, f"sent_id = {position + 1}")
            sent_id = sentence.sent_id
            if sent_id in positions:
                raise CorpusError(f"Duplicate sent_id {sent_id!r} in {self.path}.")
            positions[sent_id] = position
        self.sentences = sentences
        self._positions = positions
        self._mtime = self.path.stat().st_mtime_ns
        self._index: SourceIndex | None = None
        self.flags: list[Flag] = []
        self.history: list[Edit] = []
        self._unsaved_history: list[Edit] = []
        if self.writable and self.review_path.is_file():
            data = json.loads(self.review_path.read_text(encoding="utf-8"))
            self.flags = [Flag.model_validate(flag) for flag in data.get("flags", [])]
        if self.writable and self.history_path.is_file():
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            self.history = [Edit.model_validate_json(line) for line in lines if line.strip()]

    @property
    def index(self) -> SourceIndex:
        if self._index is None:
            self._index = SourceIndex(self.sentences, root_key=self.root_key)
        return self._index

    def changed(self) -> None:
        """Mark the indexes stale after an edit."""
        self._index = None

    def position(self, sent_id: str) -> int:
        try:
            return self._positions[sent_id]
        except KeyError:
            raise NotFoundError(f"No sentence {sent_id!r} in source {self.name!r}.") from None

    def sentence(self, sent_id: str) -> Sentence:
        return self.sentences[self.position(sent_id)]

    def replace(self, sent_id: str, sentence: Sentence) -> None:
        self.sentences[self.position(sent_id)] = sentence
        self.changed()

    def confidence(self, token: Token, column: str) -> float:
        return read_confidence(token).get(column, self.default_confidence)

    def sentence_confidence(self, sentence: Sentence) -> float:
        scores = [self.confidence(word, column) for word in sentence.words for column in SCORED_COLUMNS]
        return sum(scores) / len(scores) if scores else self.default_confidence

    def require_writable(self) -> None:
        if not self.writable:
            raise ReadOnlyError(f"Source {self.name!r} is read-only.")

    def next_flag_id(self) -> int:
        return max((flag.id for flag in self.flags), default=0) + 1

    def record(self, edit: Edit) -> None:
        edit.seq = len(self.history) + 1
        self.history.append(edit)
        self._unsaved_history.append(edit)

    def save(self) -> None:
        """Write the CoNLL-U file and sidecars.

        Raises :class:`ConflictError` if the file was modified by another
        process since it was loaded or last saved.
        """
        self.require_writable()
        if self.path.stat().st_mtime_ns != self._mtime:
            raise ConflictError(
                f"{self.path} was modified by another process; reload the source before editing."
            )
        atomic_write(self.path, format_conllu(self.sentences))
        self._mtime = self.path.stat().st_mtime_ns
        review = {"flags": [flag.model_dump() for flag in self.flags]}
        atomic_write(self.review_path, json.dumps(review, ensure_ascii=False, indent=1) + "\n")
        if self._unsaved_history:
            with self.history_path.open("a", encoding="utf-8") as stream:
                for edit in self._unsaved_history:
                    stream.write(edit.model_dump_json() + "\n")
            self._unsaved_history = []

    def info(self) -> SourceInfo:
        return SourceInfo(
            name=self.name,
            path=str(self.path),
            mode=self.mode.value,
            default_confidence=self.default_confidence,
            sentences=len(self.sentences),
            words=sum(len(sentence.words) for sentence in self.sentences),
            open_flags=sum(flag.status == "open" for flag in self.flags),
        )
