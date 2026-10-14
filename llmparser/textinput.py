"""Turn raw text into an unannotated CoNLL-U file.

Two input formats are accepted:

* plain text, one sentence per line (sentence ids are the line numbers of the
  non-empty lines);
* YAML mapping ids to entries with a ``text`` field, as in
  ``ArabGr1-Ephesians-Parsed.yaml``. Other scalar fields (e.g. ``verse``) become
  comments; a ``conllu`` field is ignored. A YAML list of such entries (with
  ``sent_id`` or ``id``) or of plain strings also works.
"""

from __future__ import annotations

import itertools
import unicodedata
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .corpus.conllu import Sentence, Token
from .corpus.errors import CorpusError

YAML_SUFFIXES = (".yaml", ".yml")
SKIPPED_FIELDS = frozenset({"text", "conllu", "sent_id", "id"})


class InputSentence(BaseModel):
    sent_id: str
    text: str
    meta: dict[str, str] = Field(default_factory=dict)


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char)[0] in "PS"


def _runs(chars: list[str]) -> list[str]:
    """Group runs of the same character (so ``...`` stays one token)."""
    return ["".join(group) for _, group in itertools.groupby(chars)]


def tokenise(text: str) -> list[tuple[str, bool]]:
    """Split on whitespace and detach leading/trailing punctuation.

    Returns ``(form, space_after)`` pairs. Punctuation inside a word (``don't``,
    ``U.S``) is kept, and combining marks are never split off.
    """
    tokens: list[tuple[str, bool]] = []
    for chunk in text.split():
        start, end = 0, len(chunk)
        while start < end and _is_punctuation(chunk[start]):
            start += 1
        while end > start and _is_punctuation(chunk[end - 1]):
            end -= 1
        pieces = _runs(list(chunk[:start])) + ([chunk[start:end]] if end > start else []) + _runs(list(chunk[end:]))
        tokens.extend((piece, index == len(pieces) - 1) for index, piece in enumerate(pieces))
    return tokens


def _entry(sent_id: str, value: object) -> InputSentence:
    if isinstance(value, str):
        return InputSentence(sent_id=sent_id, text=value)
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise CorpusError(f"Entry {sent_id!r} has no 'text' field.")
    meta = {
        str(key): str(field)
        for key, field in value.items()
        if key not in SKIPPED_FIELDS and isinstance(field, str | int | float) and "\n" not in str(field)
    }
    return InputSentence(sent_id=sent_id, text=value["text"], meta=meta)


def read_input(path: Path | str) -> list[InputSentence]:
    """Read plain text (one sentence per line) or YAML (by file extension)."""
    path = Path(path)
    if path.suffix.lower() not in YAML_SUFFIXES:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [InputSentence(sent_id=str(n), text=line) for n, line in enumerate((l for l in lines if l), start=1)]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [_entry(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        entries = []
        for n, value in enumerate(data, start=1):
            key = value.get("sent_id", value.get("id", n)) if isinstance(value, dict) else n
            entries.append(_entry(str(key), value))
        return entries
    raise CorpusError(f"{path} must contain a YAML mapping or list of sentences.")


def build_sentences(entries: list[InputSentence]) -> list[Sentence]:
    """Unannotated sentences: ids, forms and ``SpaceAfter=No`` only."""
    sentences = []
    for entry in entries:
        text = " ".join(entry.text.split())
        if not text:
            raise CorpusError(f"Sentence {entry.sent_id!r} is empty.")
        comments = [f"sent_id = {entry.sent_id}", f"text = {text}"]
        comments += [f"{key} = {value}" for key, value in entry.meta.items()]
        tokens = tokenise(text)
        words = []
        for index, (form, space_after) in enumerate(tokens, start=1):
            word = Token(id=str(index), form=form)
            if not space_after and index < len(tokens):
                word.set_misc_value("SpaceAfter", "No")
            words.append(word)
        sentences.append(Sentence(comments=comments, tokens=words))
    return sentences
