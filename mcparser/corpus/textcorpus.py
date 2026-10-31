"""Unannotated text as evidence: how often a word occurs, next to which words, in which sentences.

Raw text has no analyses, so it cannot say what a word *is*, but it shows how
the word is used: its frequency, its neighbours, whether longer words contain it
(a stem with a clitic) and what it looks like in running text. That is useful
for forms the treebank has never seen. A corpus can be tens of millions of
words, so only counts and a few line numbers per word are kept in memory; the
sentences are read from the files when an example is wanted.
"""

from __future__ import annotations

import re
import threading
from array import array
from collections import Counter
from pathlib import Path

from .errors import CorpusError
from .index import loose as loosen
from .models import CorpusExample, CorpusInfo, CorpusSearch, ValueCount

WORD = re.compile(r"[\ẁ-֑ͯ-ׇؐ-ًؚ-ٰٟۖ-ۭ]+")
"""Letters and digits plus the combining marks of Latin/Greek, Hebrew and Arabic (vowels, accents)."""
POSTINGS_MAX = 50
"""Line numbers kept per word: enough for examples and neighbour counts, small enough for 25 million words."""
AFFIX_MINIMUM = 3
"""Words shorter than this are not searched for inside longer words."""
AFFIX_EXTRA = 4
"""Longer words are at most this many letters longer (clitics are short; longer ones are mostly run-together text)."""
FILE_BITS = 40
"""A line is addressed as ``file_number << FILE_BITS | byte_offset``."""


class TextCorpus:
    """One or more plain-text files, one sentence or paragraph per line.

    The index is built the first time it is needed (about 10 seconds for 25
    million words) and is safe to share between threads. Words are runs of
    letters, digits and combining marks, so punctuation is ignored.
    """

    def __init__(self, name: str, paths: list[Path | str]):
        if not paths:
            raise CorpusError(f"Corpus {name!r} has no files.")
        self.name = name
        self.paths = [Path(path).expanduser().resolve() for path in paths]
        for path in self.paths:
            if not path.is_file():
                raise CorpusError(f"Corpus file {path} does not exist.")
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._postings: dict[str, array] = {}
        self._lines = array("Q")
        self._numbers = array("I")
        self._loose: dict[str, list[str]] | None = None
        self._built = False

    # ------------------------------------------------------------------ index

    def build(self) -> None:
        """Read the files and count the words (done once, on first use)."""
        with self._lock:
            if self._built:
                return
            for file_number, path in enumerate(self.paths):
                with path.open("rb") as stream:
                    offset = 0
                    for number, raw in enumerate(stream, start=1):
                        words = WORD.findall(raw.decode("utf-8", errors="replace"))
                        if words:
                            line = len(self._lines)
                            self._lines.append(file_number << FILE_BITS | offset)
                            self._numbers.append(number)
                            self._counts.update(words)
                            for word in set(words):
                                posting = self._postings.get(word)
                                if posting is None:
                                    self._postings[word] = array("I", [line])
                                elif len(posting) < POSTINGS_MAX:
                                    posting.append(line)
                        offset += len(raw)
            self._built = True

    def info(self) -> CorpusInfo:
        self.build()
        return CorpusInfo(
            name=self.name,
            files=[str(path) for path in self.paths],
            lines=len(self._lines),
            words=sum(self._counts.values()),
            types=len(self._counts),
        )

    def _read(self, line: int) -> tuple[Path, str]:
        address = self._lines[line]
        path = self.paths[address >> FILE_BITS]
        with path.open("rb") as stream:
            stream.seek(address & ((1 << FILE_BITS) - 1))
            return path, stream.readline().decode("utf-8", errors="replace").rstrip("\r\n")

    # ------------------------------------------------------------------ lookups

    def _matching(self, form: str, loose: bool) -> list[str]:
        """The spellings in the corpus that match ``form`` (itself, or with loose matching all that fold to the same string)."""
        if not loose:
            return [form] if form in self._counts else []
        if self._loose is None:
            folded: dict[str, list[str]] = {}
            for word in self._counts:
                folded.setdefault(loosen(word), []).append(word)
            self._loose = folded
        return self._loose.get(loosen(form), [])

    def count(self, form: str, loose: bool = False) -> int:
        """Occurrences of ``form`` (with ``loose``, ignoring case and diacritics)."""
        self.build()
        return sum(self._counts[word] for word in self._matching(form, loose))

    def _containing(self, form: str, loose: bool, limit: int) -> tuple[list[ValueCount], list[ValueCount]]:
        """Commonest longer words that end (``prefixed``) or begin (``suffixed``) with ``form``."""
        if len(form) < AFFIX_MINIMUM:
            return [], []
        key = loosen(form) if loose else form
        ends: Counter[str] = Counter()
        starts: Counter[str] = Counter()
        for word, count in self._counts.items():
            if not len(key) < len(word) <= len(key) + AFFIX_EXTRA:
                continue
            if word.endswith(key):
                ends[word] = count
            elif word.startswith(key):
                starts[word] = count
        return [ValueCount(value=w, count=c) for w, c in ends.most_common(limit)], [
            ValueCount(value=w, count=c) for w, c in starts.most_common(limit)
        ]

    def find(
        self, form: str, loose: bool = False, examples: int = 3, width: int = 6, neighbours: int = 5, affixed: int = 5
    ) -> CorpusSearch:
        """Counts, neighbours, longer words containing ``form`` and example lines (``width`` words each side)."""
        self.build()
        spellings = self._matching(form, loose)
        lines = sorted({line for word in spellings for line in self._postings[word]})
        previous: Counter[str] = Counter()
        following: Counter[str] = Counter()
        hits: list[CorpusExample] = []
        sample = 0
        wanted = set(spellings)
        step = max(1, len(lines) // examples) if examples else 1
        for position, line in enumerate(lines):
            path, text = self._read(line)
            spans = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text)]
            shown = False
            for index, (start, end, word) in enumerate(spans):
                if word not in wanted:
                    continue
                sample += 1
                if index:
                    previous[spans[index - 1][2]] += 1
                if index + 1 < len(spans):
                    following[spans[index + 1][2]] += 1
                if position % step == 0 and len(hits) < examples and not shown:
                    shown = True
                    first = spans[max(0, index - width)][0]
                    last = spans[min(len(spans) - 1, index + width)][1]
                    hits.append(
                        CorpusExample(
                            file=path.name, line=self._numbers[line],
                            before=text[first:start].strip(), form=word, after=text[end:last].strip(),
                        )  # fmt: skip
                    )
        prefixed, suffixed = self._containing(form, loose, affixed)
        return CorpusSearch(
            corpus=self.name,
            form=form,
            total=sum(self._counts[word] for word in spellings),
            forms=[ValueCount(value=w, count=self._counts[w]) for w in sorted(spellings, key=lambda w: -self._counts[w])[:neighbours]],
            previous=[ValueCount(value=w, count=c) for w, c in previous.most_common(neighbours)],
            next=[ValueCount(value=w, count=c) for w, c in following.most_common(neighbours)],
            sample=sample,
            prefixed=prefixed,
            suffixed=suffixed,
            examples=hits,
        )  # fmt: skip
