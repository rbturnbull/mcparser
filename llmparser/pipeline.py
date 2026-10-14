"""The rules → score → refine loop over a prediction source.

Before the first round, code settles what it can (see :mod:`llmparser.rules`),
including columns the trusted data never uses. Then each round:

1. **Score.** The scorer rates cells that a model or a person changed since it
   last scored them. Cells set by code are not re-checked: their confidence is
   based on the evidence.
2. **Refine.** Each role in :data:`~llmparser.agents.roles.REFINE_ORDER` works
   on cells of its column below the threshold and on flags raised for it.
   Sentences go from most to least confident, in batches of several sentences
   per task. After each role, the rules run again on the sentences it touched
   (e.g. XPOS from the new UPOS and FEATS, DEPREL for the new heads).

Rounds repeat until nothing is left to refine, a round changes nothing, or the
round limit is reached. A final scoring pass then re-scores the last changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio
from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

from .agents.evidence import EvidenceBuilder
from .agents.outputs import Batch
from .agents.prompts import TaskItem, task_prompt
from .agents.roles import REFINE_ORDER, ROLES
from .agents.team import AgentTeam, ModelUsage
from .corpus import CellUpdate, CorpusStore
from .corpus.conllu import EMPTY
from .corpus.models import CellRef
from .corpus.source import REQUIRED_COLUMNS
from .rules import PROGRAMMATIC_AGENTS, Rules, RulesReport
from .settings import Settings

EDIT_ACTIONS = frozenset(
    {"set", "split_word", "merge_words", "delete_word", "insert_word", "add_multiword", "remove_multiword"}
)
"""History actions that change annotation (as opposed to confidences and flags)."""

ALL = 1_000_000

TAG_DEPENDENT_COLUMNS = ("xpos", "feats")


class TaskResult(BaseModel):
    round: int
    sent_ids: list[str]
    role: str
    summary: str | None = None
    error: str | None = None


class RoundReport(BaseModel):
    round: int
    scored: int
    tasks: int
    edits: int
    open_flags: int
    confidence: float


class RunReport(BaseModel):
    rounds: list[RoundReport]
    final_scored: int = 0
    confidence: float
    rules: RulesReport
    tasks: list[TaskResult]
    total: ModelUsage
    usage: dict[str, ModelUsage]


@dataclass
class _CellHistory:
    reset: int = 0
    scored: dict[tuple[str, str], int] = field(default_factory=dict)
    changed: dict[tuple[str, str], tuple[int, str]] = field(default_factory=dict)
    """Last change of each cell: (history seq, agent)."""


class Pipeline:
    """Runs the rules → score → refine rounds over one writable source.

    ``sent_ids`` restricts the work to some sentences; ``progress`` receives a
    line of text after every agent task and round.
    """

    def __init__(
        self,
        store: CorpusStore,
        target: str,
        team: AgentTeam,
        settings: Settings,
        sent_ids: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.store = store
        self.target = target
        self.team = team
        self.settings = settings
        self.source = store.source(target)
        for sent_id in sent_ids or []:
            self.source.sentence(sent_id)
        self.sent_ids = set(sent_ids) if sent_ids else None
        self.progress = progress or (lambda message: None)
        self.roles = [name for name in REFINE_ORDER if settings.enabled(name)]
        self.columns = [ROLES[name].column for name in self.roles]
        self.tasks: list[TaskResult] = []
        self.evidence = EvidenceBuilder(store, target)
        self.rules = Rules(store, target, settings.rules)
        self.rules_report = RulesReport()

    # ------------------------------------------------------------ helpers

    def sentences(self, order: str = "file") -> list[str]:
        page = self.store.list_sentences(self.target, limit=ALL, order=order)
        return [s.sent_id for s in page.sentences if self.sent_ids is None or s.sent_id in self.sent_ids]

    def confidence(self) -> float:
        """Mean confidence over the cells of the enabled roles' columns."""
        scores = [cell.confidence for sent_id in self.sentences() for cell in self.cells(sent_id)]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def history_state(self) -> dict[str, _CellHistory]:
        """For each sentence, when (and by whom) each cell was last changed, and when it was last scored."""
        state: dict[str, _CellHistory] = {}
        for edit in self.source.history:
            cells = state.setdefault(edit.sent_id, _CellHistory())
            key = (edit.token_id or "", edit.column or "")
            if edit.agent == "scorer" and edit.action == "confidence":
                cells.scored[key] = edit.seq
            elif edit.action == "set":
                cells.changed[key] = (edit.seq, edit.agent)
            elif edit.action in EDIT_ACTIONS:
                cells.reset = edit.seq  # segmentation renumbers words: everything needs a new score
        return state

    def cells_to_score(self, sent_id: str, state: dict[str, _CellHistory] | None = None) -> list[CellRef]:
        """Cells that a model or person changed since the scorer last saw them.

        Skipped: cells whose last change came from code (rules or the baseline),
        required cells that are still empty (``_`` LEMMA, UPOS, HEAD, DEPREL),
        and XPOS and FEATS of words without a UPOS, which cannot be judged yet.
        """
        history = (state if state is not None else self.history_state()).get(sent_id, _CellHistory())
        untagged = {word.id for word in self.source.sentence(sent_id).words if word.upos == EMPTY}
        selected = []
        for cell in self.cells(sent_id):
            key = (cell.token_id, cell.column)
            if cell.column in REQUIRED_COLUMNS and cell.value == EMPTY:
                continue
            if cell.column in TAG_DEPENDENT_COLUMNS and cell.token_id in untagged:
                continue
            changed_at, changed_by = history.changed.get(key, (0, ""))
            if changed_by in PROGRAMMATIC_AGENTS and changed_at > history.reset:
                continue
            if history.scored.get(key, -1) < max(history.reset, changed_at):
                selected.append(cell)
        return selected

    def needs_scoring(self, sent_id: str) -> bool:
        return bool(self.cells_to_score(sent_id))

    def cells(self, sent_id: str) -> list[CellRef]:
        sentence = self.source.sentence(sent_id)
        return [
            CellRef(
                sent_id=sent_id,
                token_id=word.id,
                form=word.form,
                column=column,
                value=word.get(column),
                confidence=self.source.confidence(word, column),
            )
            for word in sentence.words
            for column in self.columns
        ]

    def edits_since(self, seq: int) -> int:
        """Edits by agents or people (not rules) after history entry ``seq``."""
        return sum(
            edit.seq > seq and edit.action in EDIT_ACTIONS and edit.agent not in PROGRAMMATIC_AGENTS
            for edit in self.source.history
        )

    def batches(self, items: list[TaskItem]) -> list[list[TaskItem]]:
        """Group items into tasks of at most ``batch_sentences`` sentences and ``batch_cells`` cells."""
        groups: list[list[TaskItem]] = []
        size = 0
        for item in items:
            cells = max(len(item.cells), 1)
            if groups and len(groups[-1]) < self.settings.batch_sentences and size + cells <= self.settings.batch_cells:
                groups[-1].append(item)
                size += cells
            else:
                groups.append([item])
                size = cells
        return groups

    def apply_rules(self, sent_ids: list[str]) -> None:
        report = self.rules.apply(sent_ids, columns=[*self.columns, "form"])
        for column, count in report.cells.items():
            self.rules_report.cells[column] = self.rules_report.cells.get(column, 0) + count
        self.rules_report.splits += report.splits

    async def _each(self, batches: list[list[TaskItem]], work: Callable[[list[TaskItem]], Awaitable[None]]) -> None:
        if self.settings.concurrency == 1:
            for batch in batches:
                await work(batch)
            return
        limiter = anyio.CapacityLimiter(self.settings.concurrency)

        async def limited(batch: list[TaskItem]) -> None:
            async with limiter:
                await work(batch)

        async with anyio.create_task_group() as group:
            for batch in batches:
                group.start_soon(limited, batch)

    async def _task(self, role: str, items: list[TaskItem]) -> None:
        sent_ids = [item.sent_id for item in items]
        result = TaskResult(round=self.store.round or 0, sent_ids=sent_ids, role=role)
        label = f"[round {result.round}] {role} {', '.join(sent_ids)}"
        try:
            spec = ROLES[role]
            for item in items:
                item.conllu = self.source.sentence(item.sent_id).to_conllu()
            evidence = self.evidence.build(spec, [(item.sent_id, item.cells) for item in items])
            answer = await self.team.run(role, task_prompt(spec, self.target, items, evidence), Batch(self.target, sent_ids))
            result.summary = answer.summary
            self.progress(f"{label}: {result.summary}")
        except UserError:
            raise  # configuration problems (e.g. a missing API key) stop the run
        except Exception as error:  # one failed task must not lose a long run
            result.error = f"{type(error).__name__}: {error}"
            self.progress(f"{label} FAILED: {result.error}")
        self.tasks.append(result)

    # ------------------------------------------------------------ phases

    async def score(self) -> int:
        """Score the cells that need it. Returns the number of sentences scored."""
        state = self.history_state()
        items = [TaskItem(sent_id, self.cells_to_score(sent_id, state)) for sent_id in self.sentences()]
        items = [item for item in items if item.cells]
        await self._each(self.batches(items), lambda batch: self._task("scorer", batch))
        return len(items)

    async def refine(self) -> int:
        """Run each role on low-confidence cells and open flags. Returns the number of tasks."""
        count = 0
        for role in self.roles:
            column = ROLES[role].column
            items = []
            for sent_id in self.sentences(order="confidence_desc"):
                cells = self.store.low_confidence(
                    self.target, threshold=self.settings.threshold, columns=[column], sent_id=sent_id, limit=ALL
                ).cells
                flags = self.store.list_flags(self.target, column=column, sent_id=sent_id, limit=ALL).flags
                if cells or flags:
                    items.append(TaskItem(sent_id, cells, flags))
            batches = self.batches(items)
            count += len(batches)
            await self._each(batches, lambda batch, role=role: self._task(role, batch))
            if items:
                self.apply_rules([item.sent_id for item in items])
        return count

    def settle_unused_columns(self) -> list[str]:
        """Settle XPOS/FEATS by rule when the trusted data never uses them.

        Some treebanks leave XPOS (or FEATS) empty throughout. Then ``_`` is the
        only right value, so it is set with confidence 1.0 and the role is not
        run, which saves a model task per sentence. Returns the settled columns.
        """
        trusted = [source for source in self.store.sources.values() if not source.writable]
        settled = []
        for column in ("xpos", "feats"):
            if column not in self.roles or not trusted or any(s.index.values[column] for s in trusted):
                continue
            self.roles.remove(column)
            self.columns.remove(column)
            settled.append(column)
            for sent_id in self.sentences():
                updates = [
                    CellUpdate(token_id=word.id, column=column, value=EMPTY, confidence=1.0,
                               reason=f"the trusted data never uses {column.upper()}")
                    for word in self.source.sentence(sent_id).words
                    if word.get(column) != EMPTY or self.source.confidence(word, column) < 1.0
                ]  # fmt: skip
                if updates:
                    self.store.update_cells(self.target, sent_id, updates, agent="rules")
            self.progress(f"{column.upper()} is never used in the trusted data: set to '_' without an agent.")
        return settled

    async def run(self, rounds: int | None = None, score_only: bool = False) -> RunReport:
        """Apply the rules, run up to ``rounds`` rounds (default from the settings), then a final scoring pass.

        With ``score_only`` only the rules and the scorer run, once.
        """
        rounds = rounds or self.settings.rounds
        reports: list[RoundReport] = []
        async with self.team:
            self.settle_unused_columns()
            self.apply_rules(self.sentences())
            self.progress(f"Rules settled {sum(self.rules_report.cells.values())} cells: {self.rules_report.cells}")
            for number in range(1, rounds + 1):
                self.store.round = number
                start = len(self.source.history)
                scored = await self.score()
                tasks = 0 if score_only else await self.refine()
                reports.append(
                    RoundReport(
                        round=number,
                        scored=scored,
                        tasks=tasks,
                        edits=self.edits_since(start),
                        open_flags=self.store.list_flags(self.target).total,
                        confidence=self.confidence(),
                    )
                )
                self.progress(f"Round {number}: {reports[-1].model_dump()}")
                if score_only or tasks == 0 or reports[-1].edits == 0:
                    break
            final_scored = 0 if score_only else await self.score()
        return RunReport(
            rounds=reports,
            final_scored=final_scored,
            confidence=self.confidence(),
            rules=self.rules_report,
            tasks=self.tasks,
            total=self.team.total(),
            usage=self.team.usage,
        )
