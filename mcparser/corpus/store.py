"""The corpus store: several CoNLL-U sources behind one search and edit API.

This is the API the MCP server exposes, and it can be used directly from Python
(e.g. by llmaligner) without MCP. All public methods are thread-safe.
"""

from __future__ import annotations

import functools
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from . import segmentation
from .confidence import check_confidence, check_confidence_column, set_confidence, strip_confidence
from .conllu import ANNOTATION_COLUMNS, EDITABLE_COLUMNS, EMPTY, UPOS_TAGS, Sentence, Token, check_cell, parse_pairs
from .errors import CorpusError, EditError, NotFoundError
from .index import loose as loose_key
from .index import relation_keys
from .models import (
    ArcHit,
    ArcSearch,
    CellPage,
    CellRef,
    CellUpdate,
    CorpusInfo,
    CorpusSearch,
    Edit,
    EditResult,
    FeatureConvention,
    FixedExpression,
    Flag,
    FlagPage,
    GroupCount,
    LinkPage,
    LinkRef,
    MultiwordAnalysis,
    MultiwordSearch,
    ParallelView,
    RelationStats,
    SentenceHit,
    SentenceIssue,
    SentencePage,
    SentenceSummary,
    SentenceView,
    SourceInfo,
    TokenHit,
    TokenRef,
    TokenSearch,
    TokenView,
    TranslationSearch,
    ValidationReport,
    ValueCount,
)
from .alignment import AlignmentInfo, AlignmentSet, Link, format_ugarit, parse_ugarit
from .source import REQUIRED_COLUMNS, SCORED_COLUMNS, CorpusSource, SourceMode
from .textcorpus import TextCorpus
from .validation import validate_sentence

COLUMN_ALIASES = {
    "segmentation": "form",
    "tokenization": "form",
    "tokenisation": "form",
    "pos": "upos",
    "morphology": "feats",
    "features": "feats",
    "arc": "head",
    "relation": "deprel",
}
"""Friendly names accepted wherever a column is expected."""

MORPHOLOGY_COLUMNS = ("lemma", "upos", "xpos", "feats")
SYNTAX_COLUMNS = ("head", "deprel")
CANDIDATE_FIELDS = ("form", "lemma", "root", "upos", "xpos")
WORD_CHANGING_ACTIONS = {"split_word", "merge_words", "delete_word", "insert_word"}

SentenceOrder = Literal["file", "confidence_asc", "confidence_desc"]
FlagFilter = Literal["open", "resolved", "dismissed", "all"]


def normalise_column(column: str) -> str:
    column = column.strip().lower()
    return COLUMN_ALIASES.get(column, column)


def _locked(method):
    @functools.wraps(method)
    def wrapper(self: CorpusStore, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _ref(token: Token) -> TokenRef:
    return TokenRef(id=token.id, form=token.form, lemma=token.lemma, upos=token.upos, deprel=token.deprel)


def _context(sentence: Sentence, token: Token, window: int) -> str:
    words = sentence.words
    index = next(i for i, word in enumerate(words) if word is token)
    left = [word.form for word in words[max(0, index - window) : index]]
    right = [word.form for word in words[index + 1 : index + 1 + window]]
    return " ".join([*left, f"[[{token.form}]]", *right])


def _deprel_matches(actual: str, query: str) -> bool:
    """``nmod`` matches ``nmod:gen``; ``nmod:gen`` only matches itself."""
    return actual == query or (":" not in query and actual.split(":")[0] == query)


def _neighbour(sentence: Sentence, token: Token, offset: int) -> Token | None:
    """The word ``offset`` words before (negative) or after (positive) ``token``."""
    words = sentence.words
    index = next((i for i, word in enumerate(words) if word is token), None)
    if index is None or not 0 <= index + offset < len(words):
        return None
    return words[index + offset]


def _direction(token: Token, head: Token | None) -> str:
    """The side of ``token`` on which its head lies: left, right or root."""
    if head is None:
        return "root"
    return "left" if int(head.id) < int(token.id) else "right"


def _head_token(sentence: Sentence, token: Token) -> Token | None:
    if not token.head.isdigit() or token.head == "0":
        return None
    try:
        return sentence.token(token.head)
    except NotFoundError:
        return None


class CorpusStore:
    """A set of named CoNLL-U sources.

    Read-only sources (``ro``) hold trusted examples such as treebank training
    data. Writable sources (``rw``) hold predictions that agents edit; every
    change is saved immediately when ``autosave`` is on.
    """

    def __init__(self, autosave: bool = True):
        self.sources: dict[str, CorpusSource] = {}
        self.alignments: dict[str, AlignmentSet] = {}
        self.text_corpora: dict[str, TextCorpus] = {}
        self.autosave = autosave
        self.round: int | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ sources

    @_locked
    def add_source(
        self,
        name: str,
        path: Path | str,
        mode: SourceMode | str = SourceMode.READ_ONLY,
        default_confidence: float | None = None,
        root_key: str = "Root",
    ) -> SourceInfo:
        if name in self.sources:
            raise CorpusError(f"A source named {name!r} already exists.")
        source = CorpusSource(name, path, mode=mode, default_confidence=default_confidence, root_key=root_key)
        self.sources[name] = source
        return source.info()

    def source(self, name: str) -> CorpusSource:
        try:
            return self.sources[name]
        except KeyError:
            available = ", ".join(self.sources) or "none"
            raise NotFoundError(f"No source named {name!r}; available: {available}.") from None

    def _select(self, names: list[str] | None) -> list[CorpusSource]:
        if not names:
            return list(self.sources.values())
        return [self.source(name) for name in names]

    def _trusted(self, names: list[str] | None) -> list[CorpusSource]:
        """The given sources, or by default the read-only ones (all if there are none)."""
        if names:
            return self._select(names)
        trusted = [source for source in self.sources.values() if not source.writable]
        return trusted or list(self.sources.values())

    @_locked
    def list_sources(self) -> list[SourceInfo]:
        return [source.info() for source in self.sources.values()]

    # ------------------------------------------------------------ text corpora
    # Unannotated text is independent of the CoNLL-U sources, so these methods do
    # not take the store lock (indexing a large corpus takes seconds).

    def add_text_corpus(self, name: str, paths: list[Path | str]) -> TextCorpus:
        """Register plain-text files as a corpus of unannotated evidence (indexed on first use)."""
        with self._lock:
            if name in self.text_corpora:
                raise CorpusError(f"A corpus named {name!r} already exists.")
            self.text_corpora[name] = TextCorpus(name, paths)
            return self.text_corpora[name]

    def text_corpus(self, name: str | None = None) -> TextCorpus:
        """The named corpus, or the only one when no name is given."""
        if name is None and len(self.text_corpora) == 1:
            return next(iter(self.text_corpora.values()))
        if name in self.text_corpora:
            return self.text_corpora[name]
        available = ", ".join(self.text_corpora) or "none"
        raise NotFoundError(f"Name a corpus; available: {available}." if name is None else f"No corpus named {name!r}; available: {available}.")

    def list_corpora(self) -> list[CorpusInfo]:
        return [corpus.info() for corpus in self.text_corpora.values()]

    def corpus_find(
        self,
        form: str,
        corpus: str | None = None,
        loose: bool = False,
        examples: int = 3,
        width: int = 6,
        neighbours: int = 5,
        affixed: int = 5,
    ) -> CorpusSearch:
        """How ``form`` is used in an unannotated corpus (see :meth:`TextCorpus.find`)."""
        return self.text_corpus(corpus).find(
            form, loose=loose, examples=examples, width=width, neighbours=neighbours, affixed=affixed
        )

    @_locked
    def save(self, source: str | None = None) -> list[str]:
        """Save one writable source or alignment set, or all of them. Returns the names saved."""
        if source in self.alignments:
            self.alignments[source].save()
            return [source]
        targets: list[CorpusSource | AlignmentSet] = (
            [self.source(source)]
            if source
            else [s for s in self.sources.values() if s.writable] + [a for a in self.alignments.values() if a.writable]
        )
        for target in targets:
            target.save()
        return [target.name for target in targets]

    @_locked
    def reload(self, source: str) -> SourceInfo:
        """Reload a source from disk, discarding unsaved changes."""
        target = self.source(source)
        target.load()
        return target.info()

    def _autosave(self, source: CorpusSource) -> None:
        if self.autosave:
            source.save()

    # ----------------------------------------------------------------- browsing

    def _summary(self, source: CorpusSource, sentence: Sentence) -> SentenceSummary:
        sent_id = sentence.sent_id
        return SentenceSummary(
            source=source.name,
            sent_id=sent_id,
            position=source.position(sent_id),
            text=sentence.text,
            words=len(sentence.words),
            confidence=round(source.sentence_confidence(sentence), 3),
            open_flags=sum(flag.sent_id == sent_id and flag.status == "open" for flag in source.flags),
        )

    @_locked
    def list_sentences(
        self,
        source: str,
        offset: int = 0,
        limit: int = 20,
        order: SentenceOrder = "file",
        max_confidence: float | None = None,
    ) -> SentencePage:
        target = self.source(source)
        summaries = [self._summary(target, sentence) for sentence in target.sentences]
        if max_confidence is not None:
            summaries = [summary for summary in summaries if summary.confidence <= max_confidence]
        if order == "confidence_asc":
            summaries.sort(key=lambda summary: summary.confidence)
        elif order == "confidence_desc":
            summaries.sort(key=lambda summary: -summary.confidence)
        return SentencePage(total=len(summaries), sentences=summaries[offset : offset + limit])

    @_locked
    def get_sentence(self, source: str, sent_id: str, include_tokens: bool = False) -> SentenceView:
        target = self.source(source)
        sentence = target.sentence(sent_id)
        return SentenceView(
            source=target.name,
            sent_id=sent_id,
            position=target.position(sent_id),
            text=sentence.text,
            conllu=sentence.to_conllu(),
            tokens=sentence.tokens if include_tokens else None,
            confidence=round(target.sentence_confidence(sentence), 3),
            open_flags=[flag for flag in target.flags if flag.sent_id == sent_id and flag.status == "open"],
            issues=validate_sentence(sentence, self.known_values()),
        )

    @_locked
    def get_context(self, source: str, sent_id: str, before: int = 2, after: int = 2) -> list[SentenceSummary]:
        """The sentence and its neighbours in file order."""
        target = self.source(source)
        position = target.position(sent_id)
        window = target.sentences[max(0, position - before) : position + after + 1]
        return [self._summary(target, sentence) for sentence in window]

    @_locked
    def get_token(self, source: str, sent_id: str, token_id: str) -> TokenView:
        target = self.source(source)
        sentence = target.sentence(sent_id)
        token = sentence.token(token_id)
        head = _head_token(sentence, token)
        multiword = None
        if token.is_word:
            for candidate in sentence.tokens:
                if candidate.is_multiword:
                    start, end = candidate.span()
                    if start <= int(token.id) <= end:
                        multiword = _ref(candidate)
        return TokenView(
            source=target.name,
            sent_id=sent_id,
            token=token,
            confidence={column: target.confidence(token, column) for column in ANNOTATION_COLUMNS},
            head=_ref(head) if head is not None else None,
            dependents=[_ref(word) for word in sentence.words if word.head == token.id],
            multiword=multiword,
        )

    # ------------------------------------------------------------------ search

    @staticmethod
    def _parse_exclude(exclude_sentence: str | None) -> tuple[str | None, str] | None:
        if not exclude_sentence:
            return None
        name, sep, sent_id = exclude_sentence.partition(":")
        return (name, sent_id) if sep else (None, exclude_sentence)

    @staticmethod
    def _excluded(exclude: tuple[str | None, str] | None, source: CorpusSource, sentence: Sentence) -> bool:
        if exclude is None:
            return False
        name, sent_id = exclude
        return sentence.sent_id == sent_id and name in (None, source.name)

    @staticmethod
    def _field_value(source: CorpusSource, token: Token, field: str) -> str:
        if field == "root":
            return token.misc_value(source.root_key) or ""
        return token.get(field)

    def _token_matches(
        self, source: CorpusSource, token: Token, criteria: dict[str, str], feats: dict[str, str | None], loose: bool
    ) -> bool:
        for field, wanted in criteria.items():
            if field == "suffix":
                if not loose_key(token.form).endswith(loose_key(wanted)):
                    return False
                continue
            actual = self._field_value(source, token, field)
            if field == "deprel":
                if not _deprel_matches(actual, wanted):
                    return False
            elif loose and field in ("form", "lemma", "root"):
                if loose_key(actual) != loose_key(wanted):
                    return False
            elif actual != wanted:
                return False
        token_feats = token.feats_dict
        return all(token_feats.get(key) == value for key, value in feats.items())

    @staticmethod
    def _context_matches(sentence: Sentence, token: Token, context: dict[tuple[int, str], str]) -> bool:
        """Whether the neighbouring words have the given forms (ignoring case and diacritics) or UPOS."""
        for (offset, column), wanted in context.items():
            neighbour = _neighbour(sentence, token, offset)
            if neighbour is None:
                return False
            if column == "form" and loose_key(neighbour.form) != loose_key(wanted):
                return False
            if column == "upos" and neighbour.upos != wanted:
                return False
        return True

    @staticmethod
    def _candidates(source: CorpusSource, criteria: dict[str, str], loose: bool) -> list[tuple[int, int]]:
        for field in CANDIDATE_FIELDS:
            if field in criteria:
                return source.index.lookup(field, criteria[field], loose)
        if "suffix" in criteria:
            found = source.index.suffix(criteria["suffix"])
            if found is not None:
                return found
        return [
            (s_pos, t_pos)
            for s_pos, sentence in enumerate(source.sentences)
            for t_pos, token in enumerate(sentence.tokens)
            if token.is_word
        ]

    def _hit(self, source: CorpusSource, sentence: Sentence, token: Token, confidence: float, window: int) -> TokenHit:
        head = _head_token(sentence, token)
        return TokenHit(
            source=source.name,
            sent_id=sentence.sent_id,
            token_id=token.id,
            form=token.form,
            lemma=token.lemma,
            upos=token.upos,
            xpos=token.xpos,
            feats=token.feats,
            head=token.head,
            deprel=token.deprel,
            misc=token.misc,
            head_form=head.form if head is not None else None,
            confidence=confidence,
            context=_context(sentence, token, window),
        )

    @_locked
    def find_tokens(
        self,
        form: str | None = None,
        lemma: str | None = None,
        upos: str | None = None,
        xpos: str | None = None,
        feats: str | None = None,
        deprel: str | None = None,
        root: str | None = None,
        suffix: str | None = None,
        previous_form: str | None = None,
        next_form: str | None = None,
        previous_upos: str | None = None,
        next_upos: str | None = None,
        loose: bool = False,
        sources: list[str] | None = None,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        group_by: list[str] | None = None,
        limit: int = 10,
        window: int = 3,
    ) -> TokenSearch:
        """Find words matching every given criterion.

        ``feats`` matches words having at least those features. ``root`` matches
        the MISC root attribute (``Root=`` by default). ``suffix`` matches word
        endings, for analogy with unseen forms; ``previous_form`` and
        ``next_form`` require the neighbouring words (these three ignore case and
        diacritics), ``previous_upos`` and ``next_upos`` their UPOS. Hits are ordered by confidence (in LEMMA/UPOS/XPOS/FEATS),
        and ``groups`` counts all matches.
        """
        criteria = {
            field: value
            for field, value in {
                "form": form, "lemma": lemma, "upos": upos, "xpos": xpos, "deprel": deprel, "root": root, "suffix": suffix,
            }.items()
            if value
        }  # fmt: skip
        context = {
            key: value
            for key, value in (
                ((-1, "form"), previous_form), ((1, "form"), next_form), ((-1, "upos"), previous_upos), ((1, "upos"), next_upos),
            )
            if value
        }  # fmt: skip
        if not criteria and not feats:
            raise EditError("Give at least one search criterion.")
        wanted_feats = parse_pairs(feats) if feats else {}
        columns = [normalise_column(column) for column in (group_by or ["lemma", "upos"])]
        for column in columns:
            Token(id="1").get(column)
        exclude = self._parse_exclude(exclude_sentence)

        matches = []
        for order, source in enumerate(self._select(sources)):
            for s_pos, t_pos in self._candidates(source, criteria, loose):
                sentence = source.sentences[s_pos]
                token = sentence.tokens[t_pos]
                if self._excluded(exclude, source, sentence):
                    continue
                if not self._token_matches(source, token, criteria, wanted_feats, loose):
                    continue
                if context and not self._context_matches(sentence, token, context):
                    continue
                confidence = min(source.confidence(token, column) for column in MORPHOLOGY_COLUMNS)
                if confidence >= min_confidence:
                    matches.append((confidence, order, s_pos, t_pos, source, sentence, token))
        matches.sort(key=lambda match: (-match[0], match[1], match[2], match[3]))
        groups = Counter(tuple(match[-1].get(column) for column in columns) for match in matches)
        return TokenSearch(
            total=len(matches),
            groups=[GroupCount(values=dict(zip(columns, key, strict=True)), count=count) for key, count in groups.most_common(25)],
            hits=[self._hit(source, sentence, token, conf, window) for conf, _, _, _, source, sentence, token in matches[:limit]],
        )

    @_locked
    def analyse_form(
        self,
        form: str,
        loose: bool = False,
        previous_form: str | None = None,
        next_form: str | None = None,
        previous_upos: str | None = None,
        next_upos: str | None = None,
        sources: list[str] | None = None,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        limit: int = 5,
    ) -> TokenSearch:
        """How a word form is analysed (optionally next to given words or tags): counts of (lemma, upos, xpos, feats) plus examples."""
        return self.find_tokens(
            form=form,
            previous_form=previous_form,
            next_form=next_form,
            previous_upos=previous_upos,
            next_upos=next_upos,
            loose=loose,
            sources=sources,
            min_confidence=min_confidence,
            exclude_sentence=exclude_sentence,
            group_by=["lemma", "upos", "xpos", "feats"],
            limit=limit,
        )

    @_locked
    def find_multiword(
        self, surface: str, loose: bool = False, sources: list[str] | None = None, limit: int = 10
    ) -> MultiwordSearch:
        """How a surface token is split into words, and how often it is left unsplit."""
        counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        examples: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
        single = 0
        for source in self._select(sources):
            single += len(source.index.lookup("form", surface, loose))
            for s_pos, t_pos in source.index.multiword(surface, loose):
                sentence = source.sentences[s_pos]
                token = sentence.tokens[t_pos]
                start, end = token.span()
                pieces = [word for word in sentence.words if start <= int(word.id) <= end]
                key = (tuple(word.form for word in pieces), tuple(word.upos for word in pieces))
                counts[key] += 1
                examples.setdefault(key, [])
                if len(examples[key]) < 3:
                    examples[key].append(f"{source.name}:{sentence.sent_id}:{token.id}")
        return MultiwordSearch(
            surface=surface,
            total=sum(counts.values()),
            analyses=[
                MultiwordAnalysis(pieces=list(forms), upos=list(tags), count=count, examples=examples[(forms, tags)])
                for (forms, tags), count in counts.most_common(limit)
            ],
            as_single_word=single,
        )

    @_locked
    def find_arcs(
        self,
        dependent_form: str | None = None,
        dependent_lemma: str | None = None,
        dependent_upos: str | None = None,
        head_form: str | None = None,
        head_lemma: str | None = None,
        head_upos: str | None = None,
        deprel: str | None = None,
        direction: Literal["left", "right", "root"] | None = None,
        loose: bool = False,
        sources: list[str] | None = None,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        limit: int = 10,
        window: int = 3,
    ) -> ArcSearch:
        """Find dependency arcs by properties of the dependent, the head and the relation.

        ``deprel`` without a subtype (``nmod``) also matches subtypes (``nmod:gen``).
        Use ``head_upos="ROOT"`` to find root words. ``direction`` is the side of
        the dependent on which the head lies.
        """
        dependent = {k: v for k, v in {"form": dependent_form, "lemma": dependent_lemma, "upos": dependent_upos}.items() if v}
        head_criteria = {k: v for k, v in {"form": head_form, "lemma": head_lemma, "upos": head_upos}.items() if v}
        if not dependent and not head_criteria and not deprel:
            raise EditError("Give at least one search criterion.")
        want_root = head_criteria.get("upos") == "ROOT"
        if want_root:
            del head_criteria["upos"]
        exclude = self._parse_exclude(exclude_sentence)

        matches = []
        for order, source in enumerate(self._select(sources)):
            for s_pos, t_pos in self._candidates(source, dependent, loose):
                sentence = source.sentences[s_pos]
                token = sentence.tokens[t_pos]
                if token.head == EMPTY or self._excluded(exclude, source, sentence):
                    continue
                if not self._token_matches(source, token, dependent, {}, loose):
                    continue
                if deprel and not _deprel_matches(token.deprel, deprel):
                    continue
                head = _head_token(sentence, token)
                if head is None and token.head != "0":
                    continue
                if want_root and head is not None:
                    continue
                if head_criteria and (head is None or not self._token_matches(source, head, head_criteria, {}, loose)):
                    continue
                if direction and _direction(token, head) != direction:
                    continue
                confidence = min(source.confidence(token, column) for column in SYNTAX_COLUMNS)
                if confidence >= min_confidence:
                    matches.append((confidence, order, s_pos, t_pos, source, sentence, token, head))
        matches.sort(key=lambda match: (-match[0], match[1], match[2], match[3]))

        hits = [
            ArcHit(
                source=source.name,
                sent_id=sentence.sent_id,
                dependent=_ref(token),
                head=_ref(head) if head is not None else None,
                deprel=token.deprel,
                direction=_direction(token, head),
                distance=abs(int(head.id) - int(token.id)) if head is not None else 0,
                confidence=confidence,
                context=_context(sentence, token, window),
            )
            for confidence, _, _, _, source, sentence, token, head in matches[:limit]
        ]
        deprels = Counter(match[6].deprel for match in matches)
        directions = Counter(_direction(match[6], match[7]) for match in matches)
        return ArcSearch(
            total=len(matches),
            deprels=[ValueCount(value=value, count=count) for value, count in deprels.most_common()],
            directions=[ValueCount(value=value, count=count) for value, count in directions.most_common()],
            hits=hits,
        )

    @_locked
    def search_sentences(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 5,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
    ) -> list[SentenceHit]:
        """BM25 search over sentence text (accent- and case-insensitive)."""
        exclude = self._parse_exclude(exclude_sentence)
        hits: list[SentenceHit] = []
        for source in self._select(sources):
            for s_pos, score in source.index.search(query):
                sentence = source.sentences[s_pos]
                confidence = source.sentence_confidence(sentence)
                if self._excluded(exclude, source, sentence) or confidence < min_confidence:
                    continue
                hits.append(
                    SentenceHit(
                        source=source.name,
                        sent_id=sentence.sent_id,
                        text=sentence.text,
                        score=round(score, 4),
                        confidence=round(confidence, 3),
                    )
                )
        hits.sort(key=lambda hit: -hit.score)
        return hits[:limit]

    # ------------------------------------------------------------------ tagsets

    @_locked
    def allowed_values(
        self, column: str, upos: str | None = None, sources: list[str] | None = None, limit: int = 100
    ) -> list[ValueCount]:
        """Values of UPOS, XPOS, DEPREL or FEATS with their frequencies.

        By default only read-only (trusted) sources are counted. With ``upos``,
        XPOS/DEPREL/FEATS are restricted to words with that UPOS.
        """
        column = normalise_column(column)
        if column not in ("upos", "xpos", "deprel", "feats"):
            raise EditError("allowed_values supports the columns upos, xpos, deprel and feats.")
        total: Counter[str] = Counter()
        for source in self._trusted(sources):
            index = source.index
            if upos and column != "upos":
                total.update(index.by_upos[column].get(upos, Counter()))
            else:
                total.update(index.values[column])
        return [ValueCount(value=value, count=count) for value, count in total.most_common(limit)]

    @_locked
    def feature_inventory(self, upos: str | None = None, sources: list[str] | None = None) -> dict[str, list[ValueCount]]:
        """Each morphological feature and its values (optionally for one UPOS)."""
        totals: dict[str, Counter[str]] = {}
        for source in self._trusted(sources):
            for tag, features in source.index.features.items():
                if upos and tag != upos:
                    continue
                for feature, values in features.items():
                    totals.setdefault(feature, Counter()).update(values)
        return {
            feature: [ValueCount(value=value, count=count) for value, count in totals[feature].most_common()]
            for feature in sorted(totals, key=str.lower)
        }

    @_locked
    def feature_conventions(
        self, upos: str, min_count: int = 20, min_share: float = 0.1, sources: list[str] | None = None
    ) -> list[FeatureConvention]:
        """Where the trusted data writes a combined feature value (``Gender=Masc,Neut``) for ``upos``.

        One entry per (Case, Number, feature) with at least ``min_count`` examples
        of which at least ``min_share`` carry a combined value, most frequent first.
        """
        total: dict[tuple[str, str, str], Counter[str]] = {}
        for source in self._trusted(sources):
            for (tag, case, number, feature), values in source.index.by_case.items():
                if tag == upos:
                    total.setdefault((case, number, feature), Counter()).update(values)
        found = [
            FeatureConvention(
                upos=upos, case=case, number=number, feature=feature,
                values=[ValueCount(value=value, count=count) for value, count in values.most_common()],
            )  # fmt: skip
            for (case, number, feature), values in total.items()
            if sum(values.values()) >= min_count and sum(c for v, c in values.items() if "," in v) >= min_share * sum(values.values())
        ]
        return sorted(found, key=lambda convention: -convention.total)

    @_locked
    def relation_stats(
        self,
        dependent_lemma: str,
        dependent_upos: str,
        head_lemma: str | None,
        head_upos: str | None,
        direction: Literal["left", "right", "root"],
        min_total: int = 3,
        sources: list[str] | None = None,
    ) -> RelationStats:
        """DEPREL counts for arcs like this one, from the most specific level with ``min_total`` arcs.

        Levels: dependent lemma + head lemma; dependent lemma + head UPOS +
        direction; dependent UPOS + head UPOS + direction. Use ``None`` heads for
        the root. Counts come from the read-only sources by default.
        """
        trusted = self._trusted(sources)
        for key in relation_keys(dependent_lemma, dependent_upos, head_lemma, head_upos, direction):
            counts: Counter[str] = Counter()
            for source in trusted:
                counts.update(source.index.relations.get(key, Counter()))
            total = sum(counts.values())
            if total >= min_total:
                return RelationStats(
                    basis=key[0], total=total, deprels=[ValueCount(value=v, count=c) for v, c in counts.most_common()]
                )
        return RelationStats(basis="none", total=0, deprels=[])

    @_locked
    def upos_for_function(self, lemma: str, deprel: str, sources: list[str] | None = None) -> list[ValueCount]:
        """UPOS values used for ``lemma`` in relation ``deprel`` (subtype ignored); default: read-only sources."""
        total: Counter[str] = Counter()
        for source in self._trusted(sources):
            total.update(source.index.upos_by_function.get((lemma, deprel.split(":")[0]), Counter()))
        return [ValueCount(value=value, count=count) for value, count in total.most_common()]

    @_locked
    def deprels_for_voice(self, deprel: str, passive: bool, sources: list[str] | None = None) -> list[ValueCount]:
        """DEPREL values (with subtype) used for ``deprel`` (subtype ignored) on dependents of a passive or active head."""
        total: Counter[str] = Counter()
        for source in self._trusted(sources):
            total.update(source.index.by_voice.get((deprel.split(":")[0], passive), Counter()))
        return [ValueCount(value=value, count=count) for value, count in total.most_common()]

    @_locked
    def fixed_expressions(
        self, source: str, sent_id: str, min_count: int = 3, min_share: float = 0.5, sources: list[str] | None = None
    ) -> list[FixedExpression]:
        """Word sequences in a sentence that the trusted data annotates as fixed expressions.

        A sequence is reported if it is fixed at least ``min_count`` times and in
        at least ``min_share`` of its occurrences.
        """
        sentence = self.source(source).sentence(sent_id)
        forms = [loose_key(word.form) for word in sentence.words]
        trusted = self._trusted(sources)
        found = []
        lengths = sorted({len(seq) for s in trusted for seq in s.index.fixed}, reverse=True)
        for length in lengths:
            for start in range(len(forms) - length + 1):
                sequence = tuple(forms[start : start + length])
                fixed = sum(s.index.fixed.get(sequence, 0) for s in trusted)
                occurrences = sum(s.index.fixed_occurrences.get(sequence, 0) for s in trusted)
                if fixed >= min_count and fixed >= min_share * occurrences:
                    words = sentence.words[start : start + length]
                    found.append(
                        FixedExpression(
                            start=words[0].id, end=words[-1].id, forms=[w.form for w in words], fixed=fixed,
                            occurrences=occurrences,
                        )
                    )  # fmt: skip
        return found

    @_locked
    def single_relations(
        self, min_heads: int = 20, max_share: float = 0.01, sources: list[str] | None = None
    ) -> dict[str, tuple[int, int]]:
        """Relations a word almost never has twice (e.g. ``nsubj``), from the trusted data.

        Returns ``{deprel: (words with such a dependent, words with two or more)}``
        for relations seen on at least ``min_heads`` words and repeated on at most
        ``max_share`` of them.
        """
        dependents: Counter[str] = Counter()
        repeated: Counter[str] = Counter()
        for source in self._trusted(sources):
            dependents.update(source.index.dependents)
            repeated.update(source.index.repeated)
        return {
            deprel: (count, repeated[deprel])
            for deprel, count in dependents.items()
            if count >= min_heads and repeated[deprel] <= max_share * count
        }

    @_locked
    def unchained_relations(
        self, min_count: int = 100, max_share: float = 0.05, sources: list[str] | None = None
    ) -> dict[str, tuple[int, int]]:
        """Relations that (almost) never attach to a word with the same relation, from the trusted data.

        Returns ``{deprel: (words with the relation, of which chained)}``, e.g. ``conj``
        attaches to the first conjunct rather than to the previous ``conj``.
        """
        total: Counter[str] = Counter()
        chained: Counter[str] = Counter()
        for source in self._trusted(sources):
            total.update(source.index.values["deprel"])
            chained.update(source.index.chained)
        return {
            deprel: (count, chained[deprel])
            for deprel, count in total.items()
            if count >= min_count and chained[deprel] <= max_share * count
        }

    @_locked
    def xpos_for_tags(self, upos: str, feats: str, sources: list[str] | None = None) -> list[ValueCount]:
        """XPOS values used with exactly this UPOS and FEATS (default: read-only sources)."""
        total: Counter[str] = Counter()
        for source in self._trusted(sources):
            total.update(source.index.xpos_by_tags.get((upos, feats), Counter()))
        return [ValueCount(value=value, count=count) for value, count in total.most_common()]

    @_locked
    def known_values(self) -> dict[str, set[str]]:
        """XPOS, DEPREL and feature values seen in read-only sources (empty if none)."""
        trusted = [source for source in self.sources.values() if not source.writable]
        if not trusted:
            return {}
        known: dict[str, set[str]] = {"xpos": set(), "deprel": set(), "feats": set()}
        for source in trusted:
            known["xpos"].update(source.index.values["xpos"])
            known["deprel"].update(source.index.values["deprel"])
            known["feats"].update(source.index.feature_pairs())
        return known

    # ------------------------------------------------------------------ quality

    @_locked
    def validate(self, source: str, sent_id: str | None = None, limit: int = 50) -> ValidationReport:
        """Validate one sentence or the whole source; errors are listed before warnings."""
        target = self.source(source)
        sentences = [target.sentence(sent_id)] if sent_id else target.sentences
        known = self.known_values()
        issues = [
            SentenceIssue(sent_id=sentence.sent_id, issue=issue)
            for sentence in sentences
            for issue in validate_sentence(sentence, known)
        ]
        issues.sort(key=lambda item: item.issue.severity != "error")
        errors = sum(item.issue.severity == "error" for item in issues)
        return ValidationReport(total=len(issues), errors=errors, warnings=len(issues) - errors, issues=issues[:limit])

    @_locked
    def low_confidence(
        self,
        source: str,
        threshold: float = 0.5,
        columns: list[str] | None = None,
        sent_id: str | None = None,
        limit: int = 50,
    ) -> CellPage:
        """Word cells whose confidence is below ``threshold``, least confident first."""
        target = self.source(source)
        wanted = [check_confidence_column(normalise_column(column)) for column in (columns or SCORED_COLUMNS)]
        sentences = [target.sentence(sent_id)] if sent_id else target.sentences
        cells = [
            CellRef(
                sent_id=sentence.sent_id,
                token_id=word.id,
                form=word.form,
                column=column,
                value=word.get(column),
                confidence=target.confidence(word, column),
            )
            for sentence in sentences
            for word in sentence.words
            for column in wanted
            if target.confidence(word, column) < threshold
        ]
        cells.sort(key=lambda cell: cell.confidence)
        return CellPage(total=len(cells), cells=cells[:limit])

    @_locked
    def get_history(
        self, source: str, sent_id: str | None = None, token_id: str | None = None, limit: int = 20
    ) -> list[Edit]:
        """Edits to a source, newest first."""
        edits = [
            edit
            for edit in self.source(source).history
            if (sent_id is None or edit.sent_id == sent_id) and (token_id is None or edit.token_id == token_id)
        ]
        return edits[::-1][:limit]

    # ------------------------------------------------------------------ editing

    def _edit(self, agent: str, **fields: Any) -> Edit:
        return Edit(seq=0, time=_now(), agent=agent, round=self.round, **fields)

    @staticmethod
    def _check_value(sentence: Sentence, column: str, value: str) -> str:
        if value == EMPTY:
            return value
        if column == "upos" and value not in UPOS_TAGS:
            raise EditError(f"{value!r} is not a universal POS tag; expected one of {', '.join(UPOS_TAGS)}.")
        if column == "head" and (not value.isdigit() or int(value) > len(sentence.words)):
            raise EditError(f"HEAD must be 0 or a word id between 1 and {len(sentence.words)}, got {value!r}.")
        return value

    def _result(self, source: CorpusSource, sentence: Sentence, edits: list[Edit], **extra: Any) -> EditResult:
        return EditResult(
            source=source.name,
            sent_id=sentence.sent_id,
            edits=edits,
            conllu=sentence.to_conllu(),
            issues=validate_sentence(sentence, self.known_values()),
            **extra,
        )

    @_locked
    def update_cells(
        self,
        source: str,
        sent_id: str,
        updates: list[CellUpdate | dict[str, Any]],
        agent: str = "agent",
        dry_run: bool = False,
    ) -> EditResult:
        """Apply several cell updates to one sentence, all or nothing.

        An update with a ``value`` replaces the cell and sets its confidence to
        ``confidence`` (or clears it, falling back to the source default). An
        update with only ``confidence`` re-scores the cell. MISC edits never
        change ``Conf``. With ``dry_run`` the updates are checked and the result
        (including validation issues) is returned, but nothing is changed.
        """
        target = self.source(source)
        target.require_writable()
        work = target.sentence(sent_id).model_copy(deep=True)
        edits: list[Edit] = []
        for update in updates:
            update = CellUpdate.model_validate(update)
            column = normalise_column(update.column)
            token = work.token(update.token_id)
            if column not in EDITABLE_COLUMNS:
                raise EditError(f"Column {column!r} cannot be edited; expected one of {', '.join(EDITABLE_COLUMNS)}.")
            if update.value is None and update.confidence is None:
                raise EditError("Each update needs a value, a confidence, or both.")
            if token.is_multiword and column not in ("form", "misc"):
                raise EditError(f"Only FORM and MISC can be set on multiword token {token.id}.")
            if update.value is None:
                set_confidence(token, check_confidence_column(column), update.confidence)
                action, old, new = "confidence", None, None
            else:
                if update.confidence is not None:
                    check_confidence_column(column)
                value = self._check_value(work, column, check_cell(column, update.value))
                if column == "misc":
                    value = strip_confidence(value, token.misc)
                old = token.get(column)
                token.set(column, value)
                if column in ANNOTATION_COLUMNS:
                    set_confidence(token, column, update.confidence)
                action, new = "set", token.get(column)
            edits.append(
                self._edit(
                    agent,
                    sent_id=sent_id,
                    token_id=token.id,
                    action=action,
                    column=column,
                    old=old,
                    new=new,
                    confidence=update.confidence,
                    reason=update.reason,
                )
            )
        if not dry_run:
            target.replace(sent_id, work)
            for edit in edits:
                target.record(edit)
            self._autosave(target)
        return self._result(target, work, edits)

    @_locked
    def score_cells(
        self,
        source: str,
        sent_id: str,
        scores: dict[str, float],
        reason: str = "",
        agent: str = "agent",
        dry_run: bool = False,
    ) -> EditResult:
        """Set confidences compactly: ``{"3:upos": 0.9, "4:*": 0.95}``.

        ``"ID:*"`` (or just ``"ID"``) scores every scored column of the word except
        required columns that are still empty (``_``), which stay unscored.
        Specific keys override a wildcard for the same word.
        """
        target = self.source(source)
        sentence = target.sentence(sent_id)
        wildcard: dict[tuple[str, str], float] = {}
        specific: dict[tuple[str, str], float] = {}
        for key, confidence in scores.items():
            token_id, _, column = key.partition(":")
            token = sentence.token(token_id.strip())
            column = normalise_column(column)
            if column in ("", "*"):
                for name in SCORED_COLUMNS:
                    if not (name in REQUIRED_COLUMNS and token.get(name) == EMPTY):
                        wildcard[(token.id, name)] = confidence
            else:
                specific[(token.id, column)] = confidence
        merged = {**wildcard, **specific}
        updates = [
            CellUpdate(token_id=token_id, column=column, confidence=confidence, reason=reason)
            for (token_id, column), confidence in merged.items()
        ]
        if not updates:
            raise EditError("No cells to score.")
        return self.update_cells(source, sent_id, updates, agent=agent, dry_run=dry_run)

    def set_cell(
        self,
        source: str,
        sent_id: str,
        token_id: str,
        column: str,
        value: str,
        confidence: float | None = None,
        reason: str = "",
        agent: str = "agent",
    ) -> EditResult:
        """Set one cell. See :meth:`update_cells`."""
        update = CellUpdate(token_id=token_id, column=column, value=value, confidence=confidence, reason=reason)
        return self.update_cells(source, sent_id, [update], agent=agent)

    @_locked
    def _segment(
        self,
        action: str,
        operation: Callable[..., segmentation.SegmentationChange],
        source: str,
        sent_id: str,
        agent: str,
        reason: str,
        confidence: float | None,
        flag_head: bool = True,
        **arguments: Any,
    ) -> EditResult:
        target = self.source(source)
        target.require_writable()
        work = target.sentence(sent_id).model_copy(deep=True)
        before = " ".join(work.word_forms())
        change = operation(work, **arguments)
        if confidence is not None:
            for new_id in change.new_ids:
                set_confidence(work.token(new_id), "form", confidence)
        for flag in target.flags:
            if flag.sent_id == sent_id and flag.status == "open" and flag.token_id is not None:
                flag.token_id = change.id_map.get(flag.token_id)
        target.replace(sent_id, work)
        edit = self._edit(
            agent,
            sent_id=sent_id,
            action=action,
            column="form",
            old=before,
            new=" ".join(work.word_forms()),
            confidence=confidence,
            reason=reason,
        )
        target.record(edit)
        if flag_head and action in WORD_CHANGING_ACTIONS:
            self._add_flag(
                target,
                sent_id=sent_id,
                token_id=None,
                column="head",
                reason=f"Segmentation changed ({action}: {before!r} -> {edit.new!r}); check HEAD and DEPREL.",
                suggestion=None,
                agent=agent,
            )
        self._autosave(target)
        return self._result(target, work, [edit], id_map=change.id_map, new_ids=change.new_ids)

    def split_word(
        self,
        source: str,
        sent_id: str,
        token_id: str,
        forms: list[str],
        multiword: bool = False,
        head_index: int = 0,
        confidence: float | None = None,
        reason: str = "",
        agent: str = "agent",
        flag_head: bool = True,
    ) -> EditResult:
        """Split a word. See :func:`mcparser.corpus.segmentation.split_word`.

        Like the other word-changing edits, this raises a sentence-level flag for
        the head agent unless ``flag_head`` is false.
        """
        return self._segment(
            "split_word", segmentation.split_word, source, sent_id, agent, reason, confidence, flag_head,
            token_id=token_id, forms=forms, multiword=multiword, head_index=head_index,
        )  # fmt: skip

    def merge_words(
        self,
        source: str,
        sent_id: str,
        token_ids: list[str],
        form: str | None = None,
        confidence: float | None = None,
        reason: str = "",
        agent: str = "agent",
    ) -> EditResult:
        """Merge adjacent words. See :func:`mcparser.corpus.segmentation.merge_words`."""
        return self._segment(
            "merge_words", segmentation.merge_words, source, sent_id, agent, reason, confidence,
            token_ids=token_ids, form=form,
        )  # fmt: skip

    def delete_word(
        self, source: str, sent_id: str, token_id: str, reason: str = "", agent: str = "agent"
    ) -> EditResult:
        """Delete a word; its dependents move to its head."""
        return self._segment(
            "delete_word", segmentation.delete_word, source, sent_id, agent, reason, None, token_id=token_id
        )

    def insert_word(
        self,
        source: str,
        sent_id: str,
        after: str,
        form: str,
        confidence: float | None = None,
        reason: str = "",
        agent: str = "agent",
    ) -> EditResult:
        """Insert an unannotated word after word ``after`` (``"0"`` = at the start)."""
        return self._segment(
            "insert_word", segmentation.insert_word, source, sent_id, agent, reason, confidence, after=after, form=form
        )

    def add_multiword(
        self,
        source: str,
        sent_id: str,
        start: int,
        end: int,
        form: str,
        reason: str = "",
        agent: str = "agent",
    ) -> EditResult:
        """Group words ``start``..``end`` under a multiword surface token."""
        return self._segment(
            "add_multiword", segmentation.add_multiword, source, sent_id, agent, reason, None,
            start=start, end=end, form=form,
        )  # fmt: skip

    def remove_multiword(
        self, source: str, sent_id: str, range_id: str, reason: str = "", agent: str = "agent"
    ) -> EditResult:
        """Remove a multiword token line, keeping its words."""
        return self._segment(
            "remove_multiword", segmentation.remove_multiword, source, sent_id, agent, reason, None, range_id=range_id
        )

    # ------------------------------------------------------------------ review flags

    def _add_flag(
        self,
        target: CorpusSource,
        sent_id: str,
        token_id: str | None,
        column: str,
        reason: str,
        suggestion: str | None,
        agent: str,
    ) -> Flag:
        flag = Flag(
            id=target.next_flag_id(),
            sent_id=sent_id,
            token_id=token_id,
            column=column,
            reason=reason,
            suggestion=suggestion,
            raised_by=agent,
            created=_now(),
            round=self.round,
        )
        target.flags.append(flag)
        target.record(
            self._edit(agent, sent_id=sent_id, token_id=token_id, action="flag", column=column, new=reason)
        )
        return flag

    @_locked
    def flag_issue(
        self,
        source: str,
        sent_id: str,
        column: str,
        reason: str,
        token_id: str | None = None,
        suggestion: str | None = None,
        agent: str = "agent",
    ) -> Flag:
        """Ask the agent responsible for ``column`` to review a word or sentence."""
        target = self.source(source)
        target.require_writable()
        sentence = target.sentence(sent_id)
        if token_id is not None:
            token_id = sentence.token(token_id).id
        column = check_confidence_column(normalise_column(column))
        if not reason.strip():
            raise EditError("A flag needs a reason.")
        flag = self._add_flag(target, sent_id, token_id, column, reason.strip(), suggestion, agent)
        self._autosave(target)
        return flag

    @_locked
    def list_flags(
        self,
        source: str,
        status: FlagFilter = "open",
        column: str | None = None,
        sent_id: str | None = None,
        limit: int = 50,
    ) -> FlagPage:
        target = self.source(source)
        column = normalise_column(column) if column else None
        flags = [
            flag
            for flag in target.flags
            if (status == "all" or flag.status == status)
            and (column is None or flag.column == column)
            and (sent_id is None or flag.sent_id == sent_id)
        ]
        return FlagPage(total=len(flags), flags=flags[:limit])

    @_locked
    def resolve_flag(
        self,
        source: str,
        flag_id: int,
        status: Literal["resolved", "dismissed"] = "resolved",
        note: str = "",
        agent: str = "agent",
    ) -> Flag:
        """Close a flag as ``resolved`` (acted on) or ``dismissed`` (no change needed)."""
        target = self.source(source)
        target.require_writable()
        if status not in ("resolved", "dismissed"):
            raise EditError("status must be 'resolved' or 'dismissed'.")
        flag = next((flag for flag in target.flags if flag.id == flag_id), None)
        if flag is None:
            raise NotFoundError(f"No flag {flag_id} in source {source!r}.")
        if flag.status != "open":
            raise EditError(f"Flag {flag_id} is already {flag.status}.")
        flag.status, flag.resolved_by, flag.note = status, agent, note
        target.record(
            self._edit(agent, sent_id=flag.sent_id, token_id=flag.token_id, action=status, column=flag.column, new=note)
        )
        self._autosave(target)
        return flag

    # ------------------------------------------------------------------ alignments

    @_locked
    def add_alignment(
        self,
        name: str,
        source_a: str,
        source_b: str,
        path: Path | str,
        mode: SourceMode | str = SourceMode.READ_WRITE,
        default_confidence: float | None = None,
    ) -> AlignmentInfo:
        """Add word alignments between two loaded sources (sentences paired by ``sent_id``)."""
        if name in self.alignments or name in self.sources:
            raise CorpusError(f"A source or alignment named {name!r} already exists.")
        alignment = AlignmentSet(
            name, self.source(source_a), self.source(source_b), path, mode=mode, default_confidence=default_confidence
        )
        self.alignments[name] = alignment
        return alignment.info()

    def alignment(self, name: str) -> AlignmentSet:
        try:
            return self.alignments[name]
        except KeyError:
            available = ", ".join(self.alignments) or "none"
            raise NotFoundError(f"No alignment named {name!r}; available: {available}.") from None

    @_locked
    def list_alignments(self) -> list[AlignmentInfo]:
        return [alignment.info() for alignment in self.alignments.values()]

    def _pair(self, alignment: AlignmentSet, sent_id: str) -> tuple[Sentence, Sentence]:
        return alignment.source_a.sentence(sent_id), alignment.source_b.sentence(sent_id)

    def _parallel(self, alignment: AlignmentSet, sent_id: str, links: list[Link]) -> ParallelView:
        a, b = self._pair(alignment, sent_id)
        linked_a = {link.source for link in links}
        linked_b = {link.target for link in links}
        return ParallelView(
            alignment=alignment.name,
            sent_id=sent_id,
            source_a=alignment.source_a.name,
            source_b=alignment.source_b.name,
            conllu_a=a.to_conllu(),
            conllu_b=b.to_conllu(),
            links=format_ugarit(links),
            confidence={link.key: alignment.confidence(link) for link in links},
            agents={link.key: link.agent for link in links if link.agent},
            unaligned_a=[word.id for word in a.words if word.id not in linked_a],
            unaligned_b=[word.id for word in b.words if word.id not in linked_b],
        )

    @_locked
    def get_parallel(self, alignment: str, sent_id: str) -> ParallelView:
        """A sentence pair: both sentences as CoNLL-U, the links, and the words without links."""
        target = self.alignment(alignment)
        return self._parallel(target, sent_id, target.links(sent_id))

    @_locked
    def set_links(
        self,
        alignment: str,
        sent_id: str,
        links: str | list[Link | dict[str, Any]],
        confidence: float | None = None,
        replace: bool = True,
        reason: str = "",
        agent: str = "agent",
        dry_run: bool = False,
    ) -> ParallelView:
        """Set the links of a sentence pair.

        ``links`` is Ugarit text (``"1-1S 2-3P"``) or a list of links;
        ``confidence`` applies to links that have none. With ``replace`` the
        pair's links are replaced, otherwise they are added (a link with the
        same word ids is updated). Word ids must exist in both sentences.
        Each new or changed link records ``agent``; a link sent again unchanged
        (same S/P and confidence) keeps the agent that last changed it.
        """
        target = self.alignment(alignment)
        target.require_writable()
        a, b = self._pair(target, sent_id)
        parsed = (
            parse_ugarit(links)
            if isinstance(links, str)
            else [Link.model_validate(link.model_dump() if isinstance(link, Link) else link) for link in links]
        )  # copies: the caller's (or the stored) links are never modified
        ids_a, ids_b = {word.id for word in a.words}, {word.id for word in b.words}
        previous = {link.key: link for link in target.links(sent_id)}
        for link in parsed:
            if link.source not in ids_a:
                raise EditError(f"Link {link.key}: no word {link.source} in {target.source_a.name}:{sent_id}.")
            if link.target not in ids_b:
                raise EditError(f"Link {link.key}: no word {link.target} in {target.source_b.name}:{sent_id}.")
            if link.confidence is None and confidence is not None:
                link.confidence = check_confidence(confidence)
            old = previous.get(link.key)
            unchanged = old is not None and (old.sure, old.confidence) == (link.sure, link.confidence)
            link.agent = old.agent if unchanged else agent
        merged = {} if replace else {link.key: link for link in target.links(sent_id)}
        merged.update({link.key: link for link in parsed})
        new = sorted(merged.values(), key=lambda link: (int(link.source), int(link.target)))
        if not dry_run:
            old = format_ugarit(target.links(sent_id))
            target.pairs[sent_id] = new
            target.record(self._edit(agent, sent_id=sent_id, action="align", column="alignment", old=old, new=format_ugarit(new), confidence=confidence, reason=reason))
            if self.autosave:
                target.save()
        return self._parallel(target, sent_id, new)

    @_locked
    def remove_links(self, alignment: str, sent_id: str, links: str, reason: str = "", agent: str = "agent") -> ParallelView:
        """Remove links (Ugarit text; S/P suffixes are ignored) from a sentence pair."""
        target = self.alignment(alignment)
        target.require_writable()
        self._pair(target, sent_id)
        keys = {link.key for link in parse_ugarit(links)}
        old = target.links(sent_id)
        new = [link for link in old if link.key not in keys]
        target.pairs[sent_id] = new
        target.record(self._edit(agent, sent_id=sent_id, action="unalign", column="alignment", old=format_ugarit(old), new=format_ugarit(new), reason=reason))
        if self.autosave:
            target.save()
        return self._parallel(target, sent_id, new)

    @staticmethod
    def _lemma(word: Token) -> str:
        return word.lemma if word.lemma != EMPTY else word.form

    @_locked
    def translation_candidates(
        self,
        lemma: str,
        side: Literal["a", "b"] = "a",
        alignments: list[str] | None = None,
        loose: bool = False,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        limit: int = 10,
    ) -> TranslationSearch:
        """What ``lemma`` (of source A, or B with ``side="b"``) is linked to, over the aligned sentence pairs.

        A word's lemma falls back to its form when the lemma is empty. Only
        links with at least ``min_confidence`` count.
        """
        sets = [self.alignment(name) for name in alignments] if alignments else list(self.alignments.values())
        wanted = loose_key(lemma) if loose else lemma
        exclude = self._parse_exclude(exclude_sentence)
        occurrences = unaligned = 0
        counts: Counter[str] = Counter()
        examples: list[str] = []
        for aset in sets:
            for sent_id, links in aset.pairs.items():
                if not links or (exclude is not None and exclude[1] == sent_id and exclude[0] in (None, aset.name)):
                    continue
                try:
                    a, b = self._pair(aset, sent_id)
                except NotFoundError:
                    continue
                here, there = (a, b) if side == "a" else (b, a)
                other = {word.id: word for word in there.words}
                linked: dict[str, list[str]] = {}
                for link in links:
                    if aset.confidence(link) >= min_confidence:
                        mine, theirs = (link.source, link.target) if side == "a" else (link.target, link.source)
                        linked.setdefault(mine, []).append(theirs)
                for word in here.words:
                    key = loose_key(self._lemma(word)) if loose else self._lemma(word)
                    if key != wanted:
                        continue
                    occurrences += 1
                    partners = [other[i] for i in linked.get(word.id, []) if i in other]
                    if not partners:
                        unaligned += 1
                    for partner in partners:
                        counts[self._lemma(partner)] += 1
                        if len(examples) < 5:
                            examples.append(f"{aset.name}:{sent_id} {word.form}→{partner.form}")
        return TranslationSearch(
            lemma=lemma,
            side=side,
            occurrences=occurrences,
            unaligned=unaligned,
            candidates=[ValueCount(value=value, count=count) for value, count in counts.most_common(limit)],
            examples=examples,
        )

    @_locked
    def low_confidence_links(
        self, alignment: str, threshold: float = 0.5, sent_id: str | None = None, limit: int = 50
    ) -> LinkPage:
        """Links below ``threshold``, least confident first."""
        target = self.alignment(alignment)
        found = []
        for pair_id, links in target.pairs.items():
            if sent_id is not None and pair_id != sent_id:
                continue
            a, b = self._pair(target, pair_id)
            for link in links:
                confidence = target.confidence(link)
                if confidence < threshold:
                    found.append(
                        LinkRef(
                            alignment=target.name, sent_id=pair_id, link=link.ugarit(),
                            form_a=a.token(link.source).form, form_b=b.token(link.target).form, confidence=confidence,
                        )
                    )  # fmt: skip
        found.sort(key=lambda ref: ref.confidence)
        return LinkPage(total=len(found), links=found[:limit])
