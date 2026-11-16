"""Anthropic Message Batches behind pydantic-ai's Anthropic client.

A batch is one-shot: independent requests go in, answers come out later (most within an
hour, at most 24 hours) at half the price. An agent task is a conversation of several
requests, so instead of batching a run, this module batches the *requests* and leaves the
agents, tools, retries and validators as they are:

- :class:`BatchingClient` is an :class:`anthropic.AsyncAnthropic` whose
  ``beta.messages.create`` does not call the API. It hands the request to a
  :class:`BatchDispatcher` and awaits the answer. Give it to pydantic-ai with
  ``AnthropicModel(name, provider=AnthropicProvider(anthropic_client=client))``.
- :class:`BatchDispatcher` queues the requests and submits a batch when every task in
  flight is waiting for its next request (or when the oldest request has waited too
  long, or a batch is full). It polls until the batch has ended, streams the results and
  resolves each request. A request that errored, expired or was cancelled raises
  :class:`BatchRequestError` in the task that sent it, so the usual handling of a failed
  task applies.

Run many tasks at once (``--concurrency``): each wave of simultaneous requests becomes one
batch, so the wall-clock time is about (dependent steps) x (batch latency), not the number
of tasks. Wrap each task in ``with client.task():`` so the dispatcher knows how many are in
flight; without it every queued request is submitted as soon as the queue is quiet.

Nothing here is specific to MCParser, so other projects using pydantic-ai with Anthropic
models can reuse it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from anthropic import AsyncAnthropic, NotGiven, Omit
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

logger = logging.getLogger("mcparser.batch")

MAX_BATCH_REQUESTS = 100_000
"""The most requests the API accepts in one batch."""
MAX_BATCH_BYTES = 200 * 1024 * 1024
"""Our limit on a batch's request size: the API's is 256 MB, and this leaves room for the envelope."""
UNSUPPORTED = ("speed", "fallbacks")
"""Parameters the Batches API rejects (``stream`` and ``max_tokens=0`` are rejected too)."""
_LOCAL_ONLY = ("stream", "timeout", "extra_headers", "extra_query", "extra_body", "betas")
"""Arguments of ``messages.create`` that are not fields of a batched request."""


class BatchRequestError(RuntimeError):
    """One request of a batch did not succeed (it errored, expired or was cancelled)."""


class BatchRecord(BaseModel):
    """What happened to one submitted batch (kept in the run report)."""

    id: str | None = None
    size: int
    betas: list[str] = []
    submitted: str
    ended: str | None = None
    seconds: float | None = None
    """Seconds from submission until the batch ended."""
    succeeded: int = 0
    errored: int = 0
    expired: int = 0
    canceled: int = 0
    error: str | None = None
    """Why the batch as a whole failed (a request that failed is counted in ``errored``)."""


class BatchesApi(Protocol):
    """The part of ``client.beta.messages.batches`` the dispatcher uses."""

    async def create(self, *, requests: list[dict[str, Any]], betas: Any = ...) -> Any: ...

    async def retrieve(self, message_batch_id: str, *, betas: Any = ...) -> Any: ...

    async def results(self, message_batch_id: str, *, betas: Any = ...) -> AsyncIterable[Any]: ...

    async def cancel(self, message_batch_id: str, *, betas: Any = ...) -> Any: ...


def _missing(value: object) -> bool:
    return isinstance(value, (Omit, NotGiven))


def batch_params(arguments: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Turn the arguments of ``beta.messages.create`` into a batched request's ``params`` and its betas.

    Omitted arguments and the ones that only matter to a live call (``timeout``, headers,
    ``stream``) are dropped. The betas travel as a header of the whole batch, so they are
    returned separately; ``extra_body`` is merged into the params. Arguments the Batches
    API does not accept raise :class:`~pydantic_ai.exceptions.UserError`.
    """
    given = {key: value for key, value in arguments.items() if not _missing(value)}
    if given.get("stream"):
        raise UserError("Batch mode cannot stream responses.")
    for name in UNSUPPORTED:
        if name in given:
            raise UserError(f"The Message Batches API does not support `{name}`; turn off batch mode to use it.")
    betas = set(given.get("betas") or ())
    betas.update(
        beta.strip() for beta in ((given.get("extra_headers") or {}).get("anthropic-beta") or "").split(",") if beta.strip()
    )
    params = {key: value for key, value in given.items() if key not in _LOCAL_ONLY}
    params.update(given.get("extra_body") or {})
    return params, tuple(sorted(betas))


@dataclass
class _Pending:
    custom_id: str
    params: dict[str, Any]
    betas: tuple[str, ...]
    size: int
    queued: float
    future: asyncio.Future[BetaMessage]


def _settle(item: _Pending, message: BetaMessage | None = None, error: Exception | None = None) -> None:
    """Resolve a request's future, unless its waiter has already gone away (a cancelled future)."""
    with suppress(asyncio.InvalidStateError):
        if error is not None:
            item.future.set_exception(error)
        else:
            item.future.set_result(message)  # type: ignore[arg-type]


@dataclass
class BatchDispatcher:
    """Queues requests and runs them as Message Batches (see the module docstring).

    ``batches`` is ``client.beta.messages.batches``. After a request is queued the
    dispatcher looks every ``settle_seconds``; it submits the queue when it did not grow
    since the last look and every task in flight is waiting, or when the oldest request has
    waited ``max_wait_seconds``, or when the queue is full. A submitted batch is polled
    every ``poll_seconds``, backing off by ``backoff`` up to ``max_poll_seconds``; a failing
    poll is retried ``poll_failures`` times. ``log`` receives one line per batch event.
    """

    batches: BatchesApi
    poll_seconds: float = 30.0
    max_poll_seconds: float = 300.0
    backoff: float = 1.5
    poll_failures: int = 5
    settle_seconds: float = 2.0
    max_wait_seconds: float = 600.0
    max_requests: int = MAX_BATCH_REQUESTS
    max_bytes: int = MAX_BATCH_BYTES
    log: Callable[[str], None] | None = None
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic
    records: list[BatchRecord] = field(default_factory=list)
    """One record per submitted batch, in submission order."""

    def __post_init__(self) -> None:
        self._pending: list[_Pending] = []
        self._active = 0
        self._counter = 0
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._jobs: set[asyncio.Task[None]] = set()

    def _say(self, line: str) -> None:
        logger.info(line)
        if self.log:
            self.log(line)

    @contextmanager
    def task(self) -> Iterator[None]:
        """Count one task in flight (a conversation that will send requests)."""
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._wake.set()

    async def submit(self, params: dict[str, Any], betas: tuple[str, ...] = ()) -> BetaMessage:
        """Queue one request and wait for its message."""
        self._counter += 1
        future: asyncio.Future[BetaMessage] = asyncio.get_running_loop().create_future()
        size = len(json.dumps(params, default=str))
        self._pending.append(_Pending(f"r{self._counter:09d}", params, betas, size, self.clock(), future))
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._loop())
        self._wake.set()
        return await future

    async def aclose(self) -> None:
        """Stop the dispatcher; batches still running are cancelled on the server."""
        tasks = [task for task in (self._loop_task, *self._jobs) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for item in self._pending:
            item.future.cancel()
        self._pending.clear()

    async def _loop(self) -> None:
        looked = -1
        while True:
            if not self._pending:
                self._wake.clear()
                looked = -1
                await self._wake.wait()
                continue
            await self.sleep(self.settle_seconds)
            count = len(self._pending)
            overdue = self.clock() - self._pending[0].queued >= self.max_wait_seconds
            if self._full() or overdue or (count == looked and count >= self._active):
                self._flush()
                looked = -1
            else:
                looked = count

    def _full(self) -> bool:
        return len(self._pending) >= self.max_requests or sum(item.size for item in self._pending) >= self.max_bytes

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        groups: dict[tuple[str, ...], list[_Pending]] = {}
        for item in pending:
            groups.setdefault(item.betas, []).append(item)
        for betas, items in groups.items():
            chunk: list[_Pending] = []
            size = 0
            for item in items:
                if chunk and (len(chunk) >= self.max_requests or size + item.size > self.max_bytes):
                    self._start(betas, chunk)
                    chunk, size = [], 0
                chunk.append(item)
                size += item.size
            self._start(betas, chunk)

    def _start(self, betas: tuple[str, ...], items: list[_Pending]) -> None:
        job = asyncio.create_task(self._process(betas, items))
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)

    async def _process(self, betas: tuple[str, ...], items: list[_Pending]) -> None:
        started = self.clock()
        record = BatchRecord(
            size=len(items), betas=list(betas), submitted=datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self.records.append(record)
        batch_id: str | None = None
        try:
            options: dict[str, Any] = {"betas": list(betas)} if betas else {}
            batch = await self.batches.create(
                requests=[{"custom_id": item.custom_id, "params": item.params} for item in items], **options
            )
            batch_id = record.id = batch.id
            self._say(f"batch {batch_id}: submitted {len(items)} requests")
            batch = await self._wait(batch, options)
            counts = batch.request_counts
            record.succeeded, record.errored = counts.succeeded, counts.errored
            record.expired, record.canceled = counts.expired, counts.canceled
            await self._resolve(batch_id, items, options)
        except asyncio.CancelledError:
            if batch_id is not None:
                with suppress(Exception):
                    await asyncio.shield(self.batches.cancel(batch_id, **options))
                self._say(f"batch {batch_id}: cancelled")
            record.error = "cancelled"
            for item in items:
                item.future.cancel()
            raise
        except Exception as error:
            record.error = f"{type(error).__name__}: {error}"
            self._say(f"batch {batch_id or '(not created)'}: failed: {record.error}")
            for item in items:
                _settle(item, error=BatchRequestError(f"batch failed: {record.error}"))
        finally:
            record.ended = datetime.now(timezone.utc).isoformat(timespec="seconds")
            record.seconds = round(self.clock() - started, 1)

    async def _wait(self, batch: Any, options: dict[str, Any]) -> Any:
        interval, failures = self.poll_seconds, 0
        while batch.processing_status != "ended":
            await self.sleep(interval)
            try:
                batch = await self.batches.retrieve(batch.id, **options)
                failures = 0
            except Exception as error:
                failures += 1
                if failures > self.poll_failures:
                    raise
                self._say(f"batch {batch.id}: could not poll ({type(error).__name__}), retrying")
            interval = min(interval * self.backoff, self.max_poll_seconds)
        counts = batch.request_counts
        self._say(
            f"batch {batch.id}: ended, {counts.succeeded} succeeded, {counts.errored} errored, "
            f"{counts.expired} expired, {counts.canceled} canceled"
        )
        return batch

    async def _resolve(self, batch_id: str, items: list[_Pending], options: dict[str, Any]) -> None:
        by_id = {item.custom_id: item for item in items}
        async for entry in await self.batches.results(batch_id, **options):
            item = by_id.pop(entry.custom_id, None)
            if item is None:
                continue
            result = entry.result
            if result.type == "succeeded":
                _settle(item, message=result.message)
            else:
                detail = result.error.error.message if result.type == "errored" else result.type
                _settle(item, error=BatchRequestError(f"request {result.type} in {batch_id}: {detail}"))
        for item in by_id.values():
            _settle(item, error=BatchRequestError(f"request {item.custom_id} has no result in {batch_id}"))


class _Messages:
    def __init__(self, dispatcher: BatchDispatcher):
        self._dispatcher = dispatcher

    async def create(self, **arguments: Any) -> BetaMessage:
        params, betas = batch_params(arguments)
        return await self._dispatcher.submit(params, betas)


class _Beta:
    def __init__(self, dispatcher: BatchDispatcher):
        self.messages = _Messages(dispatcher)


class BatchingClient(AsyncAnthropic):
    """An ``AsyncAnthropic`` whose ``beta.messages.create`` sends requests through Message Batches.

    Anything else on the client (``base_url``, ``timeout``, ...) is the real client's, which
    is what pydantic-ai reads from it. Pass ``dispatcher`` to supply one (tests do); by default
    it is built over this client's own ``beta.messages.batches``, with ``dispatcher_options``.
    """

    def __init__(
        self, dispatcher: BatchDispatcher | None = None, dispatcher_options: dict[str, Any] | None = None, **options: Any
    ):
        super().__init__(**options)
        # The parent's ``beta`` is a cached property; calling its function gives the real, live resource.
        live = AsyncAnthropic.beta.func(self)  # type: ignore[attr-defined]
        self.dispatcher = dispatcher or BatchDispatcher(live.messages.batches, **(dispatcher_options or {}))
        self._batching_beta = _Beta(self.dispatcher)

    @property
    def beta(self) -> _Beta:  # type: ignore[override]
        return self._batching_beta

    def task(self) -> Any:
        """``with client.task():`` around a conversation, so the dispatcher knows it is in flight."""
        return self.dispatcher.task()
