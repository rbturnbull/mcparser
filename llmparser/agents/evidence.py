"""Evidence gathered in advance and put into an agent's task prompt.

Without it, an agent spends its first requests calling the search tools, and
every request resends the growing conversation. With the main evidence in the
prompt, most tasks need a single request. Agents can still call the tools.

Evidence for a batch of sentences is given once per distinct form or arc.
"""

from __future__ import annotations

from collections import Counter

from ..corpus import CorpusStore
from ..corpus.conllu import EMPTY, Sentence, Token
from ..corpus.index import SUFFIX_LENGTHS, direction_of, loose
from ..corpus.models import CellRef, TokenSearch
from .roles import Role

MORPHOLOGY_ROLES = frozenset({"upos", "xpos", "lemma", "feats"})
ANALYSES_SHOWN = 3
ARC_PATTERNS_SHOWN = 3
ARC_SAMPLE = 300
LEMMA_ARC_MINIMUM = 3
AMBIGUOUS_SHARE = 0.9
"""Below this share for the commonest analysis, context evidence is added."""
RARE_COUNT = 3
"""With fewer trusted examples than this, word-ending analogies are added."""
SUFFIX_MINIMUM = 5
EXAMPLE_MAX_WORDS = 60
"""Similar trusted sentences longer than this are not shown (too many tokens)."""


class EvidenceBuilder:
    """Builds compact evidence for one role over one or more sentences of the target source."""

    def __init__(self, store: CorpusStore, target: str):
        self.store = store
        self.target = target
        self.trusted = [info.name for info in store.list_sources() if info.mode == "ro"] or None
        self._cache: dict[tuple, str] = {}

    def _cached(self, key: tuple, compute) -> str:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    # ------------------------------------------------------------ morphology

    @staticmethod
    def _groups(result: TokenSearch, shown: int = ANALYSES_SHOWN) -> str:
        text = "; ".join(f"{' '.join(g.values.values())} ×{g.count}" for g in result.groups[:shown])
        more = len(result.groups) - shown
        return text + (f"; +{more} rarer" if more > 0 else "")

    def analyses(self, form: str) -> str:
        """``lemma upos xpos feats ×count`` for the commonest analyses of ``form``."""
        return self._cached(("analyses", form), lambda: self._analyses(form))

    def _analyses(self, form: str) -> str:
        result = self.store.analyse_form(form, sources=self.trusted, limit=1)
        if result.total:
            return self._groups(result)
        result = self.store.analyse_form(form, loose=True, sources=self.trusted, limit=1)
        return self._groups(result) + " (ignoring case/diacritics)" if result.total else "no trusted example"

    def context(self, form: str, previous: str) -> str:
        """Analyses of ``form`` right after ``previous``, when the form is ambiguous."""
        return self._cached(("context", form, previous), lambda: self._context(form, previous))

    def _context(self, form: str, previous: str) -> str:
        overall = self.store.analyse_form(form, sources=self.trusted, limit=1)
        if overall.total and overall.groups[0].count / overall.total >= AMBIGUOUS_SHARE:
            return ""
        after = self.store.analyse_form(form, previous_form=previous, sources=self.trusted, limit=1)
        if after.total:
            return f"after '{previous}': {self._groups(after)}"
        return ""

    def analogy(self, form: str) -> str:
        """For rare or unseen forms: how words with the same ending are analysed, with examples."""
        return self._cached(("analogy", form), lambda: self._analogy(form))

    def _analogy(self, form: str) -> str:
        if self.store.analyse_form(form, sources=self.trusted, limit=1).total >= RARE_COUNT:
            return ""
        key = loose(form)
        for length in sorted(SUFFIX_LENGTHS, reverse=True):
            if len(key) <= length:
                continue
            result = self.store.find_tokens(
                suffix=key[-length:], sources=self.trusted, group_by=["upos", "feats"], limit=40
            )
            if result.total >= SUFFIX_MINIMUM:
                pairs = list(dict.fromkeys(f"{hit.form}→{hit.lemma}" for hit in result.hits if hit.form != form))[:4]
                return f"words ending '-{key[-length:]}': {self._groups(result)}; e.g. {', '.join(pairs)}"
        return ""

    def morphology_line(self, sentence: Sentence, word: Token, with_arc: bool) -> str:
        words = sentence.words
        index = next(i for i, w in enumerate(words) if w is word)
        parts = [self.analyses(word.form)]
        if index:
            parts.append(self.context(word.form, words[index - 1].form))
        parts.append(self.analogy(word.form))
        if with_arc:
            parts.append(self.arc_labels(sentence, word))
        return f"- {word.form}: " + " | ".join(part for part in parts if part)

    def tagset(self, role: Role, sentences: list[Sentence]) -> list[str]:
        tags = sorted({word.upos for sentence in sentences for word in sentence.words if word.upos != EMPTY})
        if role.name == "xpos":
            return [
                f"- XPOS used with {tag}: "
                + (", ".join(f"{v.value}×{v.count}" for v in self.store.allowed_values("xpos", upos=tag, sources=self.trusted, limit=8)) or "none")
                for tag in tags
            ]
        if role.name == "feats":
            lines = []
            for tag in tags:
                inventory = self.store.feature_inventory(upos=tag, sources=self.trusted)
                features = "; ".join(f"{name}={'|'.join(v.value for v in values)}" for name, values in inventory.items())
                lines.append(f"- Features used with {tag}: {features or 'none'}")
            return lines
        return []

    # ------------------------------------------------------------ syntax

    def arc_patterns(self, word: Token) -> str:
        """How words like this one attach: ``relation→HEAD_UPOS side ×count``."""
        by_lemma = word.lemma != EMPTY and self.store.find_arcs(
            dependent_lemma=word.lemma, sources=self.trusted, limit=1
        ).total >= LEMMA_ARC_MINIMUM
        if by_lemma:
            search, basis = self.store.find_arcs(dependent_lemma=word.lemma, sources=self.trusted, limit=ARC_SAMPLE), "lemma"
        elif word.upos != EMPTY:
            search, basis = self.store.find_arcs(dependent_upos=word.upos, sources=self.trusted, limit=ARC_SAMPLE), "UPOS"
        else:
            return "no UPOS or lemma yet"
        patterns = Counter(
            f"{hit.deprel}→{hit.head.upos if hit.head else 'ROOT'} {hit.direction}" for hit in search.hits
        )
        if not patterns:
            return f"no trusted arcs for this {basis}"
        shown = "; ".join(f"{pattern} ×{count}" for pattern, count in patterns.most_common(ARC_PATTERNS_SHOWN))
        return f"{shown} (by {basis}, {search.total} arcs)"

    def arc_labels(self, sentence: Sentence, word: Token) -> str:
        """Relations used in the trusted data for arcs like the word's current one."""
        head = next((other for other in sentence.words if other.id == word.head), None)
        if word.upos == EMPTY or (head is None and word.head != "0"):
            return ""
        direction = direction_of(word.id, word.head)
        stats = self.store.relation_stats(
            word.lemma, word.upos, head.lemma if head else None, head.upos if head else None, direction, sources=self.trusted
        )
        target = f"{head.form} ({head.upos}, {direction})" if head else "ROOT"
        if not stats.total:
            return f"arc to {target}: no similar trusted arcs"
        labels = ", ".join(f"{v.value}×{v.count}" for v in stats.deprels[:4])
        return f"arc to {target}: {labels} (by {stats.basis})"

    def fixed(self, sent_id: str) -> list[str]:
        lines = []
        for expression in self.store.fixed_expressions(self.target, sent_id, sources=self.trusted):
            lines.append(
                f"- {sent_id} words {expression.start}–{expression.end} '{' '.join(expression.forms)}': fixed expression "
                f"in {expression.fixed} of {expression.occurrences} occurrences (the first word is the head; "
                f"the others attach to it as fixed)"
            )
        return lines

    def examples(self, sentences: list[Sentence], limit: int = 1) -> list[str]:
        """The most similar trusted sentence for each sentence, as compact CoNLL-U (ID FORM LEMMA UPOS HEAD DEPREL)."""
        shown: set[tuple[str, str]] = set()
        lines: list[str] = []
        for sentence in sentences:
            hits = self.store.search_sentences(sentence.text, sources=self.trusted, limit=limit + 2)
            for hit in hits:
                key = (hit.source, hit.sent_id)
                example = self.store.source(hit.source).sentence(hit.sent_id)
                if key in shown or len(example.words) > EXAMPLE_MAX_WORDS:
                    continue
                shown.add(key)
                lines.append(f"## {hit.source}:{hit.sent_id} (like {sentence.sent_id})")
                lines += [f"{w.id}\t{w.form}\t{w.lemma}\t{w.upos}\t{w.head}\t{w.deprel}" for w in example.words]
                break
        return lines

    # ------------------------------------------------------------ segmentation

    @staticmethod
    def surface_tokens(sentence: Sentence) -> list[Token]:
        covered: set[str] = set()
        surface = []
        for token in sentence.tokens:
            if token.is_multiword:
                start, end = token.span()
                covered |= {str(i) for i in range(start, end + 1)}
                surface.append(token)
            elif token.is_word and token.id not in covered:
                surface.append(token)
        return surface

    def segmentation(self, sentences: list[Sentence]) -> list[str]:
        lines, seen = [], set()
        for sentence in sentences:
            for token in self.surface_tokens(sentence):
                if token.form in seen:
                    continue
                seen.add(token.form)
                found = self.store.find_multiword(token.form, sources=self.trusted, limit=ANALYSES_SHOWN)
                if found.total:
                    splits = "; ".join(f"{'+'.join(a.pieces)} ({' '.join(a.upos)}) ×{a.count}" for a in found.analyses)
                    lines.append(f"- {token.form!r}: split as {splits}; unsplit ×{found.as_single_word}")
        return lines or ["- No token of these sentences is ever split in the trusted data."]

    # ------------------------------------------------------------ driver

    def build(self, role: Role, batch: list[tuple[str, list[CellRef]]]) -> str:
        """Evidence for ``role`` over a batch of ``(sent_id, cells)``; each form or arc appears once."""
        source = self.store.source(self.target)
        sentences = [source.sentence(sent_id) for sent_id, _ in batch]
        if role.name == "segmentation":
            return "\n".join(["Multiword evidence (surface token: splits in the trusted data):", *self.segmentation(sentences)])
        lines: list[str] = []
        seen: set[str] = set()
        if role.name in MORPHOLOGY_ROLES or role.is_scorer:
            lines.append(
                "Analyses in the trusted data (lemma upos xpos feats ×count; then after the previous word, "
                "word-ending analogies" + (", relations for the current arc" if role.is_scorer else "") + "):"
            )
            for sentence, (_, cells) in zip(sentences, batch, strict=True):
                wanted = {cell.token_id for cell in cells}
                for word in sentence.words:
                    if wanted and word.id not in wanted:
                        continue
                    line = self.morphology_line(sentence, word, with_arc=role.is_scorer)
                    if role.is_scorer or line not in seen:
                        seen.add(line)
                        lines.append(line)
            lines += self.tagset(role, sentences)
        if role.name in ("head", "deprel"):
            fixed = [line for sentence in sentences for line in self.fixed(sentence.sent_id)]
            if fixed:
                lines += ["Fixed expressions of the treebank in these sentences:", *fixed]
            examples = self.examples(sentences)
            if examples:
                lines += ["Similar sentences from the treebank, to show its conventions (ID FORM LEMMA UPOS HEAD DEPREL):", *examples]
        if role.name == "deprel":
            lines.append("Relations in the trusted data for arcs like each word's current one:")
            for sentence, (sent_id, cells) in zip(sentences, batch, strict=True):
                wanted = {cell.token_id for cell in cells}
                for word in sentence.words:
                    if not wanted or word.id in wanted:
                        lines.append(f"- {sent_id} {word.id} {word.form} ({word.upos}): {self.arc_labels(sentence, word) or 'no head yet'}")
        elif not (role.name in MORPHOLOGY_ROLES or role.is_scorer):  # head and deps
            lines.append("How similar words attach in the trusted data (relation→head UPOS, side of head ×count):")
            for sentence, (_, cells) in zip(sentences, batch, strict=True):
                wanted = {cell.token_id for cell in cells}
                for word in sentence.words:
                    key = f"{word.lemma}|{word.upos}"
                    if (wanted and word.id not in wanted) or key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"- {word.form} ({word.upos}): {self.arc_patterns(word)}")
        return "\n".join(lines)
