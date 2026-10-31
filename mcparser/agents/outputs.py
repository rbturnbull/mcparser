"""Compact structured answers from the agents, checked and applied in code.

Instead of writing through tool calls and then sending a summary (two or more
requests), an agent returns one answer whose cell values are plain lines::

    12841 3 NOUN 0.95
    12841 4 VERB 0.9

Lines cost a fraction of the output tokens of JSON objects. The answer is
validated before anything is written: an unknown sentence or word, an invalid
value or (for HEAD) a broken tree is sent back to the model to fix, and only a
fully valid answer is applied, all at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from ..corpus import CellUpdate, CorpusError, CorpusStore
from ..corpus.store import normalise_column
from .roles import Role

TREE_ERRORS = frozenset({"cycle", "root-count", "self-loop", "invalid-head"})

DEPENDENT_COLUMNS = {"upos": ("xpos", "feats"), "feats": ("xpos",), "head": ("deprel",)}
"""When a model changes the key, the dependent cells of that word are marked for review."""

CLOSED_TAGSETS = ("xpos", "deprel", "feats")
"""Columns whose values must occur in the trusted data (FEATS: each Feature=Value pair)."""


class FlagRequest(BaseModel):
    sent_id: str
    token_id: str | None = Field(None, description="Omit for the whole sentence.")
    column: str = Field(description="The column that looks wrong (form, lemma, upos, xpos, feats, head, deprel).")
    reason: str
    suggestion: str | None = None


class FlagResolution(BaseModel):
    flag_id: int
    status: Literal["resolved", "dismissed"]
    note: str = ""


class Annotation(BaseModel):
    """Your answer: the values for your column."""

    values: str = Field(
        description=(
            'One line per cell you set or confirm: "SENTENCE TOKEN_ID VALUE CONFIDENCE", '
            'e.g. "s1 3 NOUN 0.95" (s1 = the first sentence of the task). Use _ for an empty value.'
        )
    )
    flags: list[FlagRequest] = Field(default_factory=list, description="Problems in other columns.")
    resolved: list[FlagResolution] = Field(default_factory=list, description="Flags addressed to you that you closed.")
    summary: str = Field("", description="At most 15 words.")


class Scores(BaseModel):
    """Your answer: confidences for the cells."""

    scores: str = Field(
        description=(
            'Lines "SENTENCE TOKEN_ID COLUMN CONFIDENCE", e.g. "s1 3 upos 0.4". "s1 3 * 0.95" scores every '
            'listed cell of word 3, "s1 * * 0.95" every listed cell of sentence s1; later lines override earlier ones.'
        )
    )
    flags: list[FlagRequest] = Field(default_factory=list, description="Cells you are fairly sure are wrong, with a fix.")
    summary: str = Field("", description="At most 15 words.")


@dataclass
class Batch:
    """The sentences of one task; passed to the agent run as its dependencies.

    In prompts and answers the sentences are called ``s1``, ``s2``, ... rather
    than by their ``sent_id``, which can be long (e.g. 20 tokens in Coptic
    Scriptorium) and would otherwise be repeated on every answer line.
    """

    target: str
    sent_ids: list[str]
    cells: dict[str, list[tuple[str, str]]] | None = None
    """For the scorer: the (token_id, column) cells listed per sentence, which wildcards expand to."""
    warned: set[tuple[str, str, str]] = field(default_factory=set)
    """Doubled relations already pointed out to the model, which may confirm them by answering again."""

    def label(self, sent_id: str) -> str:
        return f"s{self.sent_ids.index(sent_id) + 1}"

    def resolve(self, name: str) -> str | None:
        """The sent_id for a label (``s2``) or a sent_id of this batch, else ``None``."""
        if name in self.sent_ids:
            return name
        if name[:1] == "s" and name[1:].isdigit() and 0 < int(name[1:]) <= len(self.sent_ids):
            return self.sent_ids[int(name[1:]) - 1]
        return None


def _lines(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _confidence(text: str) -> float:
    return min(max(float(text), 0.0), 1.0)


def parse_syntax(text: str, batch: Batch) -> tuple[dict[str, list[CellUpdate]], list[str]]:
    """``SENTENCE TOKEN_ID HEAD DEPREL CONFIDENCE [DEPREL_CONFIDENCE]`` lines as HEAD and DEPREL updates."""
    updates: dict[str, list[CellUpdate]] = {}
    errors = []
    for number, parts in enumerate(_lines(text), start=1):
        if len(parts) not in (5, 6):
            errors.append(
                f"line {number}: expected SENTENCE TOKEN_ID HEAD DEPREL CONFIDENCE [DEPREL_CONFIDENCE], got {' '.join(parts)!r}"
            )
            continue
        name, token_id, head, deprel, *confidences = parts
        sent_id = batch.resolve(name)
        if sent_id is None:
            errors.append(f"line {number}: sentence {name!r} is not in this task")
            continue
        try:
            head_score, deprel_score = _confidence(confidences[0]), _confidence(confidences[-1])
        except ValueError:
            errors.append(f"line {number}: confidences {' '.join(confidences)!r} are not numbers")
            continue
        updates.setdefault(sent_id, []).extend([
            CellUpdate(token_id=token_id, column="head", value=head, confidence=head_score),
            CellUpdate(token_id=token_id, column="deprel", value=deprel, confidence=deprel_score),
        ])  # fmt: skip
    return updates, errors


def parse_values(text: str, role: Role, batch: Batch) -> tuple[dict[str, list[CellUpdate]], list[str]]:
    """``SENTENCE TOKEN_ID VALUE CONFIDENCE`` lines as updates per sentence, plus error messages."""
    if role.columns == ("head", "deprel"):
        return parse_syntax(text, batch)
    updates: dict[str, list[CellUpdate]] = {}
    errors = []
    for number, parts in enumerate(_lines(text), start=1):
        if len(parts) < 4:
            errors.append(f"line {number}: expected SENTENCE TOKEN_ID VALUE CONFIDENCE, got {' '.join(parts)!r}")
            continue
        name, token_id, value, confidence = parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]
        sent_id = batch.resolve(name)
        if sent_id is None:
            errors.append(f"line {number}: sentence {name!r} is not in this task")
            continue
        try:
            score = _confidence(confidence)
        except ValueError:
            errors.append(f"line {number}: confidence {confidence!r} is not a number")
            continue
        updates.setdefault(sent_id, []).append(CellUpdate(token_id=token_id, column=role.column, value=value, confidence=score))
    return updates, errors


def parse_scores(text: str, batch: Batch) -> tuple[dict[str, dict[str, float]], list[str]]:
    """``SENT_ID TOKEN_ID COLUMN CONFIDENCE`` lines as ``score_cells`` dicts per sentence."""
    scores: dict[str, dict[str, float]] = {}
    errors = []
    for number, parts in enumerate(_lines(text), start=1):
        if len(parts) != 4:
            errors.append(f"line {number}: expected SENTENCE TOKEN_ID COLUMN CONFIDENCE, got {' '.join(parts)!r}")
            continue
        name, token_id, column, confidence = parts
        sent_id = batch.resolve(name)
        if sent_id is None:
            errors.append(f"line {number}: sentence {name!r} is not in this task")
            continue
        try:
            score = float(confidence)
        except ValueError:
            errors.append(f"line {number}: confidence {confidence!r} is not a number")
            continue
        score = min(max(score, 0.0), 1.0)
        column = "*" if column == "*" else normalise_column(column)
        cells = scores.setdefault(sent_id, {})
        listed = batch.cells.get(sent_id) if batch.cells else None
        if listed is not None and "*" in (token_id, column):
            # Wildcards cover only the listed cells, so they never re-score cells settled by rules.
            matches = [
                (token, name) for token, name in listed
                if token_id in ("*", token) and column in ("*", name)
            ]  # fmt: skip
            if not matches:
                errors.append(f"line {number}: no listed cell matches {token_id} {column}")
            for token, name in matches:
                cells[f"{token}:{name}"] = score
        elif token_id == "*":
            errors.append(f"line {number}: TOKEN_ID * is only allowed for the cells listed in the task")
        else:
            cells[f"{token_id}:{column}"] = score
    return scores, errors


class Applier:
    """Checks an agent's answer against the store, then applies it."""

    def __init__(self, store: CorpusStore, role: Role):
        self.store = store
        self.role = role

    def _retry(self, errors: list[str]) -> None:
        if errors:
            raise ModelRetry("Fix these problems and answer again:\n" + "\n".join(f"- {e}" for e in errors[:20]))

    def _check_flags(self, flags: list[FlagRequest], batch: Batch, errors: list[str]) -> None:
        for flag in flags:
            resolved = batch.resolve(flag.sent_id)
            if resolved is None:
                errors.append(f"flag on sentence {flag.sent_id!r}, which is not in this task")
            else:
                flag.sent_id = resolved

    def _apply_flags(self, flags: list[FlagRequest], batch: Batch) -> None:
        for flag in flags:
            self.store.flag_issue(
                batch.target, flag.sent_id, flag.column, flag.reason, token_id=flag.token_id,
                suggestion=flag.suggestion, agent=self.role.name,
            )  # fmt: skip

    def _unseen(self, updates: dict[str, list[CellUpdate]]) -> list[str]:
        """Values the trusted data never uses (only for closed tagsets, and only if there is trusted data)."""
        known_values = self.store.known_values()
        errors = []
        for sent_id, cells in updates.items():
            for cell in cells:
                column = cell.column
                known = known_values.get(column) if column in CLOSED_TAGSETS else None
                if not known:
                    continue
                value = cell.value or "_"
                parts = value.split("|") if column == "feats" else [value]
                unseen = [part for part in parts if part != "_" and part not in known]
                if unseen:
                    errors.append(
                        f"sentence {sent_id} word {cell.token_id}: {', '.join(unseen)} never occurs in the treebank's "
                        f"{column.upper()}; use the treebank's values (see allowed_values / feature_inventory)."
                    )
        return errors

    def _doubled(self, updates: dict[str, list[CellUpdate]], batch: Batch) -> list[str]:
        """Warn (once) when the answer leaves a word with two dependents of a relation the treebank almost never repeats.

        In Coptic, a noun subject before a verb that also has a pronoun subject
        is ``dislocated``; in most treebanks a word has one ``nsubj`` and one ``obj``.
        """
        if not {"head", "deprel"} & set(self.role.columns):
            return []
        single = self.store.single_relations()
        warnings = []
        for sent_id, cells in updates.items():
            words = {w.id: [w.form, w.head, w.deprel] for w in self.store.source(batch.target).sentence(sent_id).words}
            for cell in cells:  # ids and values were checked by the dry run
                words[cell.token_id][1 if cell.column == "head" else 2] = cell.value or "_"
            groups: dict[tuple[str, str], list[str]] = {}
            for token_id, (_, head, deprel) in words.items():
                if deprel in single:
                    groups.setdefault((head, deprel), []).append(token_id)
            for (head, deprel), ids in groups.items():
                key = (sent_id, head, deprel)
                if len(ids) < 2 or key in batch.warned:
                    continue
                batch.warned.add(key)
                heads, repeated = single[deprel]
                dependents = ", ".join(f"{token_id} {words[token_id][0]}" for token_id in ids)
                warnings.append(
                    f"sentence {batch.label(sent_id)}: word {head} would have {len(ids)} '{deprel}' dependents ({dependents}), "
                    f"but in the treebank only {repeated} of {heads} words have more than one. Check whether one of them "
                    "has another relation (e.g. dislocated, conj, appos, det) or another head. If the answer is right, "
                    "send it again unchanged."
                )
        return warnings

    def _follow_up(self, sent_id: str, changed: list[tuple[str, str, str]], batch: Batch) -> None:
        """Mark dependent cells for review and close flags that this change answers.

        ``changed`` holds ``(token_id, column, new value)``. Dependent columns that
        this role set itself (DEPREL for the joint syntax role) are not marked.
        """
        sentence = self.store.source(batch.target).sentence(sent_id)
        review = [
            CellUpdate(token_id=token_id, column=dependent, confidence=0.0, reason=f"{column.upper()} changed")
            for token_id, column, _ in changed
            for dependent in DEPENDENT_COLUMNS.get(column, ())
            if dependent not in self.role.columns and sentence.token(token_id).get(dependent) != "_"
        ]
        if review:
            self.store.update_cells(batch.target, sent_id, review, agent=self.role.name)
        values = {(token_id, column): value for token_id, column, value in changed}
        for column in self.role.columns:
            for flag in self.store.list_flags(batch.target, column=column, sent_id=sent_id, limit=10_000).flags:
                if (flag.token_id, column) in values:
                    self.store.resolve_flag(
                        batch.target, flag.id, "resolved", f"{self.role.name} changed it to {values[(flag.token_id, column)]}",
                        agent=self.role.name,
                    )  # fmt: skip

    def annotation(self, answer: Annotation, batch: Batch) -> Annotation:
        updates, errors = parse_values(answer.values, self.role, batch)
        errors += self._unseen(updates)
        for sent_id, cells in updates.items():
            try:
                result = self.store.update_cells(batch.target, sent_id, cells, agent=self.role.name, dry_run=True)
            except CorpusError as error:
                errors.append(f"sentence {sent_id}: {error}")
                continue
            if "head" in self.role.columns:
                errors += [
                    f"sentence {sent_id}: {issue.message}" for issue in result.issues if issue.code in TREE_ERRORS
                ]
        self._check_flags(answer.flags, batch, errors)
        self._retry(errors)
        self._retry(self._doubled(updates, batch))
        for resolution in answer.resolved:
            try:
                self.store.resolve_flag(batch.target, resolution.flag_id, resolution.status, resolution.note, agent=self.role.name)
            except CorpusError:
                pass  # an unknown or already closed flag is harmless
        for sent_id, cells in updates.items():
            result = self.store.update_cells(batch.target, sent_id, cells, agent=self.role.name)
            changed = [(edit.token_id, edit.column, edit.new) for edit in result.edits if edit.old != edit.new]
            self._follow_up(sent_id, changed, batch)
        self._apply_flags(answer.flags, batch)
        return answer

    def scores(self, answer: Scores, batch: Batch) -> Scores:
        scores, errors = parse_scores(answer.scores, batch)
        for sent_id, cells in scores.items():
            try:
                self.store.score_cells(batch.target, sent_id, cells, agent=self.role.name, dry_run=True)
            except CorpusError as error:
                errors.append(f"sentence {sent_id}: {error}")
        self._check_flags(answer.flags, batch, errors)
        self._retry(errors)
        for sent_id, cells in scores.items():
            self.store.score_cells(batch.target, sent_id, cells, agent=self.role.name)
        self._apply_flags(answer.flags, batch)
        return answer

