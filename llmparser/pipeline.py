"""The score → refine loop over a prediction source.

Each round:

1. **Score.** The scorer agent assigns a confidence to every cell of each
   sentence that has not been scored since it last changed.
2. **Refine.** Sentences are visited from most to least confident, so settled
   sentences are available as evidence for harder ones. In each sentence the
   role agents run in :data:`~llmparser.agents.roles.REFINE_ORDER` on cells below
   the threshold and on flags raised for their column.

Rounds repeat until nothing is left to refine, a round changes nothing, or the
round limit is reached. A final scoring pass then re-scores the last changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import anyio
from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

from .agents.prompts import task_prompt
from .agents.roles import REFINE_ORDER, ROLES
from .agents.team import AgentTeam
from .corpus import CorpusStore
from .corpus.models import CellRef, Flag
from .settings import Settings

EDIT_ACTIONS = frozenset(
    {"set", "split_word", "merge_words", "delete_word", "insert_word", "add_multiword", "remove_multiword"}
)
"""History actions that change annotation (as opposed to confidences and flags)."""

ALL = 1_000_000


class TaskResult(BaseModel):
    round: int
    sent_id: str
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
    tasks: list[TaskResult]
    requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


class Pipeline:
    """Runs the score → refine rounds over one writable source.

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

    # ------------------------------------------------------------ helpers

    def sentences(self, order: str = "file") -> list[str]:
        page = self.store.list_sentences(self.target, limit=ALL, order=order)
        return [s.sent_id for s in page.sentences if self.sent_ids is None or s.sent_id in self.sent_ids]

    def confidence(self) -> float:
        """Mean confidence over the cells of the enabled roles' columns."""
        scores = [cell.confidence for sent_id in self.sentences() for cell in self.cells(sent_id)]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def needs_scoring(self, sent_id: str) -> bool:
        """True if the sentence was never scored, or changed after it was last scored."""
        last_score = last_edit = 0
        for edit in self.source.history:
            if edit.sent_id != sent_id:
                continue
            if edit.agent == "scorer" and edit.action == "confidence":
                last_score = edit.seq
            elif edit.action in EDIT_ACTIONS:
                last_edit = edit.seq
        return last_score == 0 or last_edit > last_score

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
        return sum(edit.seq > seq and edit.action in EDIT_ACTIONS for edit in self.source.history)

    async def _each(self, sent_ids: list[str], work: Callable[[str], Awaitable[None]]) -> None:
        if self.settings.concurrency == 1:
            for sent_id in sent_ids:
                await work(sent_id)
            return
        limiter = anyio.CapacityLimiter(self.settings.concurrency)

        async def limited(sent_id: str) -> None:
            async with limiter:
                await work(sent_id)

        async with anyio.create_task_group() as group:
            for sent_id in sent_ids:
                group.start_soon(limited, sent_id)

    async def _task(self, role: str, sent_id: str, cells: list[CellRef], flags: list[Flag]) -> None:
        prompt = task_prompt(ROLES[role], self.target, sent_id, cells, flags)
        result = TaskResult(round=self.store.round or 0, sent_id=sent_id, role=role)
        try:
            result.summary = (await self.team.run(role, prompt)).summary
            self.progress(f"[round {result.round}] {sent_id} {role}: {result.summary}")
        except UserError:
            raise  # configuration problems (e.g. a missing API key) stop the run
        except Exception as error:  # one failed task must not lose a long run
            result.error = f"{type(error).__name__}: {error}"
            self.progress(f"[round {result.round}] {sent_id} {role} FAILED: {result.error}")
        self.tasks.append(result)

    # ------------------------------------------------------------ phases

    async def score(self) -> int:
        """Score every sentence that needs it. Returns the number scored."""
        targets = [sent_id for sent_id in self.sentences() if self.needs_scoring(sent_id)]

        async def work(sent_id: str) -> None:
            await self._task("scorer", sent_id, self.cells(sent_id), [])

        await self._each(targets, work)
        return len(targets)

    async def refine(self) -> int:
        """Run the role agents on low-confidence cells and open flags. Returns the number of tasks."""
        count = 0

        async def work(sent_id: str) -> None:
            nonlocal count
            for role in self.roles:
                column = ROLES[role].column
                cells = self.store.low_confidence(
                    self.target, threshold=self.settings.threshold, columns=[column], sent_id=sent_id, limit=ALL
                ).cells
                flags = self.store.list_flags(self.target, column=column, sent_id=sent_id, limit=ALL).flags
                if cells or flags:
                    count += 1
                    await self._task(role, sent_id, cells, flags)

        await self._each(self.sentences(order="confidence_desc"), work)
        return count

    async def run(self, rounds: int | None = None, score_only: bool = False) -> RunReport:
        """Run up to ``rounds`` rounds (default from the settings), then a final scoring pass.

        With ``score_only`` only the scorer runs, once.
        """
        rounds = rounds or self.settings.rounds
        reports: list[RoundReport] = []
        final_scored = 0
        async with self.team:
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
            if not score_only:
                final_scored = await self.score()
        return RunReport(
            rounds=reports,
            final_scored=final_scored,
            confidence=self.confidence(),
            tasks=self.tasks,
            requests=self.team.requests,
            input_tokens=self.team.input_tokens,
            output_tokens=self.team.output_tokens,
            cache_read_tokens=self.team.cache_read_tokens,
        )
