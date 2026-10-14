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
from .confidence import check_confidence_column, set_confidence, strip_confidence
from .conllu import ANNOTATION_COLUMNS, EDITABLE_COLUMNS, EMPTY, UPOS_TAGS, Sentence, Token, check_cell, parse_pairs
from .errors import CorpusError, EditError, NotFoundError
from .index import loose as loose_key
from .models import (
    ArcHit,
    ArcSearch,
    CellPage,
    CellRef,
    CellUpdate,
    Edit,
    EditResult,
    Flag,
    FlagPage,
    GroupCount,
    MultiwordAnalysis,
    MultiwordSearch,
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
    ValidationReport,
    ValueCount,
)
from .source import SCORED_COLUMNS, CorpusSource, SourceMode
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

    @_locked
    def save(self, source: str | None = None) -> list[str]:
        """Save one writable source, or all of them. Returns the names saved."""
        targets = [self.source(source)] if source else [s for s in self.sources.values() if s.writable]
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
    def _candidates(source: CorpusSource, criteria: dict[str, str], loose: bool) -> list[tuple[int, int]]:
        for field in CANDIDATE_FIELDS:
            if field in criteria:
                return source.index.lookup(field, criteria[field], loose)
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
        the MISC root attribute (``Root=`` by default). Hits are ordered by
        confidence (in LEMMA/UPOS/XPOS/FEATS), and ``groups`` counts all matches.
        """
        criteria = {
            field: value
            for field, value in {"form": form, "lemma": lemma, "upos": upos, "xpos": xpos, "deprel": deprel, "root": root}.items()
            if value
        }
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
        sources: list[str] | None = None,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        limit: int = 5,
    ) -> TokenSearch:
        """How a word form is analysed: counts of (lemma, upos, xpos, feats) plus examples."""
        return self.find_tokens(
            form=form,
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
        loose: bool = False,
        sources: list[str] | None = None,
        min_confidence: float = 0.0,
        exclude_sentence: str | None = None,
        limit: int = 10,
        window: int = 3,
    ) -> ArcSearch:
        """Find dependency arcs by properties of the dependent, the head and the relation.

        ``deprel`` without a subtype (``nmod``) also matches subtypes (``nmod:gen``).
        Use ``head_upos="ROOT"`` to find root words.
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
                confidence = min(source.confidence(token, column) for column in SYNTAX_COLUMNS)
                if confidence >= min_confidence:
                    matches.append((confidence, order, s_pos, t_pos, source, sentence, token, head))
        matches.sort(key=lambda match: (-match[0], match[1], match[2], match[3]))

        def direction(token: Token, head: Token | None) -> str:
            if head is None:
                return "root"
            return "left" if int(head.id) < int(token.id) else "right"

        hits = [
            ArcHit(
                source=source.name,
                sent_id=sentence.sent_id,
                dependent=_ref(token),
                head=_ref(head) if head is not None else None,
                deprel=token.deprel,
                direction=direction(token, head),
                distance=abs(int(head.id) - int(token.id)) if head is not None else 0,
                confidence=confidence,
                context=_context(sentence, token, window),
            )
            for confidence, _, _, _, source, sentence, token, head in matches[:limit]
        ]
        deprels = Counter(match[6].deprel for match in matches)
        directions = Counter(direction(match[6], match[7]) for match in matches)
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
        self, source: str, sent_id: str, updates: list[CellUpdate | dict[str, Any]], agent: str = "agent"
    ) -> EditResult:
        """Apply several cell updates to one sentence, all or nothing.

        An update with a ``value`` replaces the cell and sets its confidence to
        ``confidence`` (or clears it, falling back to the source default). An
        update with only ``confidence`` re-scores the cell. MISC edits never
        change ``Conf``.
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
        target.replace(sent_id, work)
        for edit in edits:
            target.record(edit)
        self._autosave(target)
        return self._result(target, work, edits)

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
        if action in WORD_CHANGING_ACTIONS:
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
    ) -> EditResult:
        """Split a word. See :func:`llmparser.corpus.segmentation.split_word`."""
        return self._segment(
            "split_word", segmentation.split_word, source, sent_id, agent, reason, confidence,
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
        """Merge adjacent words. See :func:`llmparser.corpus.segmentation.merge_words`."""
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
