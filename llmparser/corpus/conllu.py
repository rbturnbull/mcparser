"""A small, lossless CoNLL-U model.

Each cell is kept as the raw string from the file so that a read/write round
trip never reorders features or rewrites values. Helpers parse FEATS and MISC
into dictionaries when structure is needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from .errors import ConlluFormatError, EditError, NotFoundError

COLUMNS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")
"""The ten CoNLL-U columns in file order."""

EDITABLE_COLUMNS = COLUMNS[1:]
"""Columns that may be changed with a cell edit (everything except ``id``)."""

ANNOTATION_COLUMNS = ("form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps")
"""Columns that carry a confidence score. ``form`` stands for segmentation."""

UPOS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)  # fmt: skip
"""The universal part-of-speech tags."""

EMPTY = "_"

ID_PATTERN = re.compile(r"^(\d+|\d+-\d+|\d+\.\d+)$")


def parse_pairs(value: str, separator: str = "|") -> dict[str, str | None]:
    """Parse ``Key=Value|Key2=Value2`` into an ordered dict.

    Items without ``=`` map to ``None``. ``_`` and the empty string give ``{}``.
    """
    if value in ("", EMPTY):
        return {}
    pairs: dict[str, str | None] = {}
    for item in value.split(separator):
        key, sep, val = item.partition("=")
        pairs[key] = val if sep else None
    return pairs


def format_pairs(pairs: dict[str, str | None], separator: str = "|") -> str:
    """Inverse of :func:`parse_pairs`. An empty dict gives ``_``."""
    if not pairs:
        return EMPTY
    return separator.join(key if val is None else f"{key}={val}" for key, val in pairs.items())


def format_feats(pairs: dict[str, str | None]) -> str:
    """Format FEATS with keys sorted case-insensitively, as UD requires."""
    return format_pairs(dict(sorted(pairs.items(), key=lambda item: item[0].lower())))


def check_cell(column: str, value: str) -> str:
    """Validate a cell value for ``column`` and return it (``""`` becomes ``_``)."""
    if column not in COLUMNS:
        raise EditError(f"Unknown column {column!r}; expected one of {', '.join(COLUMNS)}.")
    if any(char in value for char in "\t\n\r"):
        raise EditError(f"Value for {column} must not contain tabs or newlines.")
    value = value.strip()
    if not value:
        return EMPTY
    if " " in value and column not in ("form", "lemma"):
        raise EditError(f"Value for {column} must not contain spaces: {value!r}.")
    return value


class Token(BaseModel):
    """One CoNLL-U line: a word, a multiword range (``3-4``) or an empty node (``5.1``)."""

    id: str
    form: str = EMPTY
    lemma: str = EMPTY
    upos: str = EMPTY
    xpos: str = EMPTY
    feats: str = EMPTY
    head: str = EMPTY
    deprel: str = EMPTY
    deps: str = EMPTY
    misc: str = EMPTY

    @property
    def is_word(self) -> bool:
        return self.id.isdigit()

    @property
    def is_multiword(self) -> bool:
        return "-" in self.id

    @property
    def is_empty_node(self) -> bool:
        return "." in self.id

    def span(self) -> tuple[int, int]:
        """The ``(start, end)`` word ids covered by a multiword token."""
        start, _, end = self.id.partition("-")
        return int(start), int(end)

    def get(self, column: str) -> str:
        if column not in COLUMNS:
            raise EditError(f"Unknown column {column!r}; expected one of {', '.join(COLUMNS)}.")
        return getattr(self, column)

    def set(self, column: str, value: str) -> None:
        setattr(self, column, check_cell(column, value))

    @property
    def misc_dict(self) -> dict[str, str | None]:
        return parse_pairs(self.misc)

    @property
    def feats_dict(self) -> dict[str, str | None]:
        return parse_pairs(self.feats)

    def misc_value(self, key: str) -> str | None:
        return self.misc_dict.get(key)

    def set_misc_value(self, key: str, value: str | None) -> None:
        """Set (or with ``None`` remove) one MISC attribute, keeping the others."""
        pairs = self.misc_dict
        if value is None:
            pairs.pop(key, None)
        else:
            pairs[key] = value
        self.misc = format_pairs(pairs)

    def to_line(self) -> str:
        return "\t".join(getattr(self, column) for column in COLUMNS)

    @classmethod
    def from_line(cls, line: str, line_number: int | None = None) -> Token:
        where = f"line {line_number}: " if line_number is not None else ""
        cells = line.split("\t")
        if len(cells) != len(COLUMNS):
            raise ConlluFormatError(f"{where}expected 10 tab-separated columns, found {len(cells)}.")
        if not ID_PATTERN.match(cells[0]):
            raise ConlluFormatError(f"{where}invalid token id {cells[0]!r}.")
        return cls(**dict(zip(COLUMNS, cells, strict=True)))


class Sentence(BaseModel):
    """A sentence: its comment lines (without ``# ``) and its token lines."""

    comments: list[str] = Field(default_factory=list)
    tokens: list[Token] = Field(default_factory=list)

    def meta(self, key: str) -> str | None:
        """The value of a ``# key = value`` comment, if present."""
        for comment in self.comments:
            name, sep, value = comment.partition("=")
            if sep and name.strip() == key:
                return value.strip()
        return None

    def set_meta(self, key: str, value: str) -> None:
        line = f"{key} = {value}"
        for index, comment in enumerate(self.comments):
            name, sep, _ = comment.partition("=")
            if sep and name.strip() == key:
                self.comments[index] = line
                return
        self.comments.append(line)

    @property
    def sent_id(self) -> str | None:
        return self.meta("sent_id")

    @property
    def text(self) -> str:
        """The ``# text`` comment, or the surface forms joined if it is missing."""
        text = self.meta("text")
        if text is not None:
            return text
        return surface_text(self.tokens)

    @property
    def words(self) -> list[Token]:
        return [token for token in self.tokens if token.is_word]

    def token(self, token_id: str | int) -> Token:
        token_id = str(token_id)
        for token in self.tokens:
            if token.id == token_id:
                return token
        raise NotFoundError(f"No token with id {token_id!r} in sentence {self.sent_id!r}.")

    def word_forms(self) -> list[str]:
        return [token.form for token in self.words]

    def to_conllu(self) -> str:
        lines = [f"# {comment}" for comment in self.comments]
        lines.extend(token.to_line() for token in self.tokens)
        return "\n".join(lines)


def surface_text(tokens: Iterable[Token]) -> str:
    """Rebuild running text from surface tokens, honouring ``SpaceAfter=No``."""
    parts: list[str] = []
    skip_until = 0
    for token in tokens:
        if token.is_empty_node:
            continue
        if token.is_multiword:
            skip_until = token.span()[1]
        elif int(token.id) <= skip_until:
            continue
        parts.append(token.form)
        if token.misc_value("SpaceAfter") != "No":
            parts.append(" ")
    return "".join(parts).strip()


def parse_conllu(text: str) -> list[Sentence]:
    """Parse CoNLL-U text into sentences."""
    sentences: list[Sentence] = []
    current = Sentence()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            if current.tokens or current.comments:
                sentences.append(_finish(current, line_number))
                current = Sentence()
            continue
        if line.startswith("#"):
            if current.tokens:
                raise ConlluFormatError(f"line {line_number}: comment after token lines.")
            current.comments.append(line[1:].strip())
            continue
        current.tokens.append(Token.from_line(line, line_number))
    if current.tokens or current.comments:
        sentences.append(_finish(current, None))
    return sentences


def _finish(sentence: Sentence, line_number: int | None) -> Sentence:
    if not sentence.tokens:
        where = f"line {line_number}: " if line_number is not None else "end of file: "
        raise ConlluFormatError(f"{where}sentence has comments but no tokens.")
    return sentence


def format_conllu(sentences: Iterable[Sentence]) -> str:
    """Serialise sentences, each followed by a blank line."""
    return "".join(f"{sentence.to_conllu()}\n\n" for sentence in sentences)
