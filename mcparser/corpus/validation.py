"""Structural and tagset checks for a sentence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .conllu import EMPTY, UPOS_TAGS, Sentence

Severity = Literal["error", "warning"]


class Issue(BaseModel):
    code: str
    severity: Severity
    message: str
    token_id: str | None = None
    column: str | None = None


def validate_sentence(sentence: Sentence, known: dict[str, set[str]] | None = None) -> list[Issue]:
    """Check ids, ranges, tags, features and the dependency tree.

    ``known`` maps a column (``xpos``, ``deprel``, ``feats``) to the values seen in
    trusted data; values outside it give an ``unseen-value`` warning. For
    ``feats`` the set holds individual ``Feature=Value`` pairs.
    """
    issues: list[Issue] = []
    words = sentence.words
    n_words = len(words)

    def add(code: str, severity: Severity, message: str, token_id=None, column=None) -> None:
        issues.append(Issue(code=code, severity=severity, message=message, token_id=token_id, column=column))

    for expected, word in enumerate(words, start=1):
        if word.id != str(expected):
            add("word-id-sequence", "error", f"Word ids must run 1..n; found {word.id} at position {expected}.", word.id, "id")
            break

    covered: set[int] = set()
    for token in sentence.tokens:
        if not token.is_multiword:
            continue
        start, end = token.span()
        span = set(range(start, end + 1))
        if not 1 <= start < end <= n_words:
            add("bad-range", "error", f"Multiword token {token.id} does not cover at least two existing words.", token.id, "id")
        elif span & covered:
            add("overlapping-range", "error", f"Multiword token {token.id} overlaps another multiword token.", token.id, "id")
        covered |= span

    known = known or {}
    heads: dict[str, int] = {}
    for word in words:
        for column in ("lemma", "upos", "head", "deprel"):
            if word.get(column) == EMPTY:
                add(f"missing-{column}", "warning", f"Word {word.id} ({word.form}) has no {column.upper()}.", word.id, column)
        if word.upos != EMPTY and word.upos not in UPOS_TAGS:
            add("invalid-upos", "error", f"Word {word.id} has UPOS {word.upos!r}, which is not a universal tag.", word.id, "upos")
        issues.extend(_check_feats(word.id, word.feats, known.get("feats")))
        for column in ("xpos", "deprel"):
            value = word.get(column)
            if column in known and value != EMPTY and value not in known[column]:
                add("unseen-value", "warning", f"Word {word.id} has {column.upper()} {value!r}, which never occurs in the training data.", word.id, column)

        if word.head == EMPTY:
            continue
        if not word.head.isdigit() or int(word.head) > n_words:
            add("invalid-head", "error", f"Word {word.id} has HEAD {word.head!r}, which is not 0 or a word id.", word.id, "head")
            continue
        if word.head == word.id:
            add("self-loop", "error", f"Word {word.id} is its own head.", word.id, "head")
            continue
        heads[word.id] = int(word.head)
        if word.head == "0" and word.deprel not in ("root", EMPTY):
            add("root-deprel", "error", f"Word {word.id} has HEAD 0 but DEPREL {word.deprel!r}; expected root.", word.id, "deprel")
        if word.head != "0" and word.deprel.split(":")[0] == "root":
            add("root-deprel", "error", f"Word {word.id} has DEPREL root but HEAD {word.head}; expected 0.", word.id, "deprel")

    if words and len(heads) == n_words:
        issues.extend(_check_tree(heads))
    return issues


def _check_feats(token_id: str, feats: str, known: set[str] | None) -> list[Issue]:
    if feats == EMPTY:
        return []
    issues: list[Issue] = []
    keys: list[str] = []
    for pair in feats.split("|"):
        key, sep, value = pair.partition("=")
        if not (sep and key and value):
            return [Issue(code="bad-feats", severity="error", message=f"Word {token_id} has malformed FEATS item {pair!r}; expected Feature=Value.", token_id=token_id, column="feats")]
        keys.append(key)
        if known is not None and pair not in known:
            issues.append(Issue(code="unseen-value", severity="warning", message=f"Word {token_id} has feature {pair!r}, which never occurs in the training data.", token_id=token_id, column="feats"))
    if len(set(keys)) != len(keys):
        issues.append(Issue(code="duplicate-feature", severity="error", message=f"Word {token_id} repeats a feature in {feats!r}.", token_id=token_id, column="feats"))
    elif keys != sorted(keys, key=str.lower):
        issues.append(Issue(code="unsorted-feats", severity="warning", message=f"Word {token_id} FEATS are not sorted alphabetically.", token_id=token_id, column="feats"))
    return issues


def _check_tree(heads: dict[str, int]) -> list[Issue]:
    issues: list[Issue] = []
    roots = [token_id for token_id, head in heads.items() if head == 0]
    if len(roots) != 1:
        issues.append(Issue(code="root-count", severity="error", message=f"The sentence has {len(roots)} roots ({', '.join(roots) or 'none'}); expected exactly 1.", column="head"))
    in_cycle: set[str] = set()
    for start in heads:
        seen: list[str] = []
        node = start
        while node != "0" and node not in seen and node not in in_cycle:
            seen.append(node)
            node = str(heads[node])
        if node in seen:
            cycle = seen[seen.index(node) :]
            in_cycle.update(cycle)
            issues.append(Issue(code="cycle", severity="error", message=f"Words {' -> '.join(cycle)} form a cycle.", token_id=cycle[0], column="head"))
    return issues
