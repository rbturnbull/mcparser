"""In-memory indexes over one source's sentences.

Lookups are by exact string or by a *loose* key (Unicode NFD with combining
marks removed, case-folded), which lets unvocalised or unaccented forms match
annotated ones. Sentence search uses BM25 over loose word tokens.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from .conllu import EMPTY, Sentence

WORD_PATTERN = re.compile(r"\w+")


def loose(text: str) -> str:
    """Case-fold and strip combining marks (accents, vowel points, breathings)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()


def terms(text: str) -> list[str]:
    return WORD_PATTERN.findall(loose(text))


Location = tuple[int, int]
"""(sentence position, token position) within a source."""


class SourceIndex:
    """Postings and tag inventories for one list of sentences."""

    FIELDS = ("form", "lemma", "upos", "xpos", "deprel", "root")

    def __init__(self, sentences: list[Sentence], root_key: str = "Root", bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.root_key = root_key
        self.exact: dict[str, defaultdict[str, list[Location]]] = {field: defaultdict(list) for field in self.FIELDS}
        self.loose: dict[str, defaultdict[str, list[Location]]] = {field: defaultdict(list) for field in ("form", "lemma", "root")}
        self.multiword_exact: defaultdict[str, list[Location]] = defaultdict(list)
        self.multiword_loose: defaultdict[str, list[Location]] = defaultdict(list)
        self.values: dict[str, Counter[str]] = {column: Counter() for column in ("upos", "xpos", "deprel", "feats")}
        self.by_upos: dict[str, defaultdict[str, Counter[str]]] = {column: defaultdict(Counter) for column in ("xpos", "deprel", "feats")}
        self.features: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

        self.k1, self.b = bm25_k1, bm25_b
        self.doc_terms: list[Counter[str]] = []
        self.doc_freq: Counter[str] = Counter()

        for s_pos, sentence in enumerate(sentences):
            self._add(s_pos, sentence)
        lengths = [sum(doc.values()) for doc in self.doc_terms]
        self.avg_length = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _add(self, s_pos: int, sentence: Sentence) -> None:
        for t_pos, token in enumerate(sentence.tokens):
            location = (s_pos, t_pos)
            if token.is_multiword:
                self.multiword_exact[token.form].append(location)
                self.multiword_loose[loose(token.form)].append(location)
                continue
            if not token.is_word:
                continue
            root = token.misc_value(self.root_key)
            values = {"form": token.form, "lemma": token.lemma, "upos": token.upos, "xpos": token.xpos, "deprel": token.deprel, "root": root}
            for field, value in values.items():
                if value and value != EMPTY:
                    self.exact[field][value].append(location)
                    if field in self.loose:
                        self.loose[field][loose(value)].append(location)
            for column in ("upos", "xpos", "deprel", "feats"):
                value = token.get(column)
                if value != EMPTY:
                    self.values[column][value] += 1
                    if column != "upos":
                        self.by_upos[column][token.upos][value] += 1
            for feature, value in token.feats_dict.items():
                if value is not None:
                    self.features[token.upos][feature][value] += 1
        doc = Counter(terms(sentence.text))
        self.doc_terms.append(doc)
        self.doc_freq.update(doc.keys())

    def lookup(self, field: str, value: str, loose_match: bool = False) -> list[Location]:
        if loose_match and field in self.loose:
            return self.loose[field].get(loose(value), [])
        return self.exact[field].get(value, [])

    def multiword(self, surface: str, loose_match: bool = False) -> list[Location]:
        if loose_match:
            return self.multiword_loose.get(loose(surface), [])
        return self.multiword_exact.get(surface, [])

    def feature_pairs(self) -> set[str]:
        return {f"{feature}={value}" for features in self.features.values() for feature, values in features.items() for value in values}

    def search(self, query: str) -> list[tuple[int, float]]:
        """BM25-ranked ``(sentence position, score)`` pairs with a positive score."""
        query_terms = set(terms(query))
        n_docs = len(self.doc_terms)
        scores: list[tuple[int, float]] = []
        for s_pos, doc in enumerate(self.doc_terms):
            length = sum(doc.values())
            score = 0.0
            for term in query_terms & doc.keys():
                idf = math.log(1 + (n_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                tf = doc[term]
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            if score > 0:
                scores.append((s_pos, score))
        return scores


def merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return total
