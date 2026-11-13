"""Segmentation edits: splitting, merging, inserting and deleting words, and
adding or removing multiword tokens.

Every edit works on a :class:`_Graph` in which HEAD and DEPS point at token
objects rather than ids. After the edit the graph is renumbered, so heads,
enhanced dependencies, multiword ranges and empty nodes stay consistent.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from .confidence import CONF_KEY
from .conllu import EMPTY, Sentence, Token, check_cell
from .errors import EditError


class _Root:
    """Sentinel for HEAD 0."""


ROOT = _Root()

Ref = Token | _Root | None


class SegmentationChange(BaseModel):
    """What a segmentation edit did to the sentence's token ids."""

    id_map: dict[str, str | None] = Field(
        description="Old word/empty-node id -> new id (None if the word was deleted)."
    )
    new_ids: list[str] = Field(description="Ids, after renumbering, of the words created or rebuilt.")


class _Graph:
    def __init__(self, sentence: Sentence):
        self.sentence = sentence
        self.words = [token for token in sentence.tokens if token.is_word]
        nodes = {token.id: token for token in sentence.tokens if not token.is_multiword}
        self.old_ids = {id(token): (token.id, token) for token in nodes.values()}
        self.empties: list[tuple[Token | None, Token]] = []
        anchor: Token | None = None
        for token in sentence.tokens:
            if token.is_word:
                anchor = token
            elif token.is_empty_node:
                self.empties.append((anchor, token))
        self.ranges: list[list[Token]] = []
        for token in sentence.tokens:
            if token.is_multiword:
                start, end = token.span()
                first, last = nodes.get(str(start)), nodes.get(str(end))
                if first is not None and last is not None:
                    self.ranges.append([first, last, token])
        self.head: dict[int, Ref] = {id(word): self._ref(word.head, nodes) for word in self.words}
        self.deps: dict[int, list[tuple[Ref, str]] | None] = {
            id(token): self._deps(token.deps, nodes)
            for token in [*self.words, *(empty for _, empty in self.empties)]
        }
        self.redirect: dict[int, Ref] = {}
        self.deleted: set[int] = set()

    @staticmethod
    def _ref(value: str, nodes: dict[str, Token]) -> Ref:
        if value == "0":
            return ROOT
        return nodes.get(value)

    def _deps(self, value: str, nodes: dict[str, Token]) -> list[tuple[Ref, str]] | None:
        """Parse DEPS into references; ``None`` means keep the raw string."""
        if value == EMPTY:
            return None
        items: list[tuple[Ref, str]] = []
        for item in value.split("|"):
            head, sep, relation = item.partition(":")
            ref = self._ref(head, nodes)
            if not sep or ref is None:
                return None
            items.append((ref, relation))
        return items

    def index(self, word: Token) -> int:
        return next(i for i, candidate in enumerate(self.words) if candidate is word)

    def range_of(self, word: Token) -> Token | None:
        position = self.index(word)
        for first, last, token in self.ranges:
            if self.index(first) <= position <= self.index(last):
                return token
        return None

    def move_empties(self, old: Token, new: Token | None) -> None:
        self.empties = [(new if anchor is old else anchor, empty) for anchor, empty in self.empties]

    def resolve(self, ref: Ref, alive: set[int]) -> Ref:
        """Follow redirects from removed words to the word that replaced them.

        Each edit redirects a removed word to a surviving word (or ROOT), so
        the chain is at most one step long and cannot loop.
        """
        while ref is not None and ref is not ROOT and id(ref) not in alive:
            ref = self.redirect.get(id(ref))
        return ref

    def build(self, new_words: list[Token]) -> SegmentationChange:
        alive = {id(word) for word in self.words} | {id(empty) for _, empty in self.empties}
        number = {id(word): str(i) for i, word in enumerate(self.words, start=1)}
        counts: Counter[str] = Counter()
        for anchor, empty in self.empties:
            base = number[id(anchor)] if anchor is not None else "0"
            counts[base] += 1
            number[id(empty)] = f"{base}.{counts[base]}"

        position = {id(word): i for i, word in enumerate(self.words)}
        starts: dict[int, list[Token]] = {}
        for first, last, token in self.ranges:
            if id(first) in position and id(last) in position and position[id(first)] < position[id(last)]:
                token.id = f"{number[id(first)]}-{number[id(last)]}"
                starts.setdefault(id(first), []).append(token)

        def head_value(ref: Ref) -> str:
            ref = self.resolve(ref, alive)
            if ref is ROOT:
                return "0"
            return number[id(ref)] if ref is not None else EMPTY

        tokens: list[Token] = [empty for anchor, empty in self.empties if anchor is None]
        for word in self.words:
            tokens.extend(starts.get(id(word), []))
            tokens.append(word)
            tokens.extend(empty for anchor, empty in self.empties if anchor is word)

        for token in tokens:
            if token.is_multiword:
                continue
            if id(token) in self.head:
                token.head = head_value(self.head[id(token)])
            deps = self.deps.get(id(token))
            if deps is not None:
                values = [(head_value(ref), relation) for ref, relation in deps]
                token.deps = "|".join(f"{h}:{r}" for h, r in values if h != EMPTY) or EMPTY
            token.id = number[id(token)]
        self.sentence.tokens = tokens

        id_map: dict[str, str | None] = {}
        for key, (old_id, token) in self.old_ids.items():
            if key in self.deleted:
                id_map[old_id] = None
            else:
                ref = self.resolve(token, alive)
                id_map[old_id] = number[id(ref)] if isinstance(ref, Token) else None
        return SegmentationChange(id_map=id_map, new_ids=[number[id(word)] for word in new_words])


def _word(sentence: Sentence, token_id: str | int) -> Token:
    token = sentence.token(token_id)
    if not token.is_word:
        raise EditError(f"Token {token.id!r} is not a word (it is a multiword range or empty node).")
    return token


def _form(form: str) -> str:
    form = check_cell("form", form)
    if form == EMPTY:
        raise EditError("A word form must not be empty.")
    return form


def _misc_without(token: Token, *keys: str) -> str:
    probe = Token(id="1", misc=token.misc)
    for key in keys:
        probe.set_misc_value(key, None)
    return probe.misc


def split_word(
    sentence: Sentence,
    token_id: str | int,
    forms: list[str],
    multiword: bool = False,
    head_index: int = 0,
) -> SegmentationChange:
    """Replace one word with several.

    The piece at ``head_index`` keeps the original word's head, relation, lemma,
    tags and features, and receives its dependents; the other pieces attach to
    it with an empty relation. With ``multiword=True`` the original surface form
    is kept as a multiword range (e.g. Arabic clitics); otherwise the word is
    re-tokenised into separate surface tokens.
    """
    word = _word(sentence, token_id)
    forms = [_form(form) for form in forms]
    if len(forms) < 2:
        raise EditError("Splitting a word needs at least two forms.")
    if not 0 <= head_index < len(forms):
        raise EditError(f"head_index must be between 0 and {len(forms) - 1}.")
    graph = _Graph(sentence)
    if multiword and graph.range_of(word) is not None:
        raise EditError(f"Word {word.id} is already inside a multiword token.")

    pieces = [Token(id="0", form=form) for form in forms]
    head_piece = pieces[head_index]
    for column in ("lemma", "upos", "xpos", "feats", "deprel"):
        head_piece.set(column, word.get(column))
    head_piece.misc = _misc_without(word, CONF_KEY, "SpaceAfter")
    no_space_after = word.misc_value("SpaceAfter") == "No"
    if not multiword:
        for piece in pieces[:-1]:
            piece.set_misc_value("SpaceAfter", "No")
        if no_space_after:
            pieces[-1].set_misc_value("SpaceAfter", "No")

    index = graph.index(word)
    graph.words[index : index + 1] = pieces
    graph.head[id(head_piece)] = graph.head[id(word)]
    graph.deps[id(head_piece)] = graph.deps[id(word)]
    for piece in pieces:
        if piece is not head_piece:
            graph.head[id(piece)] = head_piece
    graph.redirect[id(word)] = head_piece
    for bounds in graph.ranges:
        if bounds[0] is word:
            bounds[0] = pieces[0]
        if bounds[1] is word:
            bounds[1] = pieces[-1]
    graph.move_empties(word, pieces[-1])
    if multiword:
        surface = Token(id="0", form=word.form)
        if no_space_after:
            surface.set_misc_value("SpaceAfter", "No")
        graph.ranges.append([pieces[0], pieces[-1], surface])
    return graph.build(pieces)


def merge_words(
    sentence: Sentence,
    token_ids: list[str | int],
    form: str | None = None,
) -> SegmentationChange:
    """Merge adjacent words into one.

    The merged word takes its annotation from the group's syntactic head (the
    first word whose head lies outside the group). If ``form`` is omitted it is
    the surface form of a multiword token that exactly covers the group, or the
    forms concatenated.
    """
    if len(token_ids) < 2:
        raise EditError("Merging needs at least two word ids.")
    words = [_word(sentence, token_id) for token_id in token_ids]
    graph = _Graph(sentence)
    positions = sorted(graph.index(word) for word in words)
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise EditError("Only adjacent words can be merged.")
    group = graph.words[positions[0] : positions[-1] + 1]
    members = {id(word) for word in group}

    def outside(ref: Ref) -> bool:
        return not isinstance(ref, Token) or id(ref) not in members

    group_head = next((word for word in group if outside(graph.head[id(word)])), group[0])
    if form is None:
        exact = [t for first, last, t in graph.ranges if first is group[0] and last is group[-1]]
        form = exact[0].form if exact else "".join(word.form for word in group)
    merged = group_head.model_copy()
    merged.form = _form(form)
    merged.misc = _misc_without(group_head, CONF_KEY, "SpaceAfter")
    if group[-1].misc_value("SpaceAfter") == "No":
        merged.set_misc_value("SpaceAfter", "No")

    head = graph.head[id(group_head)]
    graph.head[id(merged)] = head if outside(head) else None
    graph.deps[id(merged)] = graph.deps[id(group_head)]
    graph.words[positions[0] : positions[-1] + 1] = [merged]
    for word in group:
        graph.redirect[id(word)] = merged
        graph.move_empties(word, merged)
    for bounds in graph.ranges:
        if id(bounds[0]) in members:
            bounds[0] = merged
        if id(bounds[1]) in members:
            bounds[1] = merged
    return graph.build([merged])


def delete_word(sentence: Sentence, token_id: str | int) -> SegmentationChange:
    """Delete a word. Its dependents are reattached to its head."""
    word = _word(sentence, token_id)
    graph = _Graph(sentence)
    if len(graph.words) == 1:
        raise EditError("Cannot delete the only word in a sentence.")
    index = graph.index(word)
    previous = graph.words[index - 1] if index > 0 else None
    following = graph.words[index + 1] if index + 1 < len(graph.words) else None
    for bounds in graph.ranges:
        if bounds[0] is word and following is not None:
            bounds[0] = following
        if bounds[1] is word and previous is not None:
            bounds[1] = previous
    head = graph.head[id(word)]
    graph.redirect[id(word)] = None if head is word else head
    graph.deleted.add(id(word))
    graph.move_empties(word, previous)
    del graph.words[index]
    return graph.build([])


def insert_word(sentence: Sentence, after: str | int, form: str) -> SegmentationChange:
    """Insert a new, unannotated word after word ``after`` (``0`` = at the start)."""
    graph = _Graph(sentence)
    word = Token(id="0", form=_form(form))
    if str(after) == "0":
        index = 0
    else:
        index = graph.index(_word(sentence, after)) + 1
    graph.words.insert(index, word)
    graph.head[id(word)] = None
    return graph.build([word])


def add_multiword(sentence: Sentence, start: int, end: int, form: str) -> SegmentationChange:
    """Group words ``start``..``end`` under a multiword token with surface ``form``."""
    if end <= start:
        raise EditError("A multiword token must span at least two words (end > start).")
    first, last = _word(sentence, start), _word(sentence, end)
    graph = _Graph(sentence)
    for word in graph.words[graph.index(first) : graph.index(last) + 1]:
        existing = graph.range_of(word)
        if existing is not None:
            raise EditError(f"Word {word.id} is already inside multiword token {existing.id}.")
    surface = Token(id="0", form=_form(form))
    if last.misc_value("SpaceAfter") == "No":
        last.set_misc_value("SpaceAfter", None)
        surface.set_misc_value("SpaceAfter", "No")
    graph.ranges.append([first, last, surface])
    return graph.build([])


def remove_multiword(sentence: Sentence, range_id: str) -> SegmentationChange:
    """Remove a multiword token line, keeping its words."""
    token = sentence.token(range_id)
    if not token.is_multiword:
        raise EditError(f"Token {range_id!r} is not a multiword token.")
    graph = _Graph(sentence)
    # A range whose words are missing is not in the graph and is dropped by build().
    bounds = next((bounds for bounds in graph.ranges if bounds[2] is token), None)
    if bounds is not None:
        graph.ranges.remove(bounds)
        if token.misc_value("SpaceAfter") == "No":
            bounds[1].set_misc_value("SpaceAfter", "No")
    return graph.build([])
