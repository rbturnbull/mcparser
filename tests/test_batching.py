"""The Message Batches layer and its wiring into the agent team. No network: a fake batches API stands in."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from anthropic import NOT_GIVEN, omit
from anthropic.resources.beta.messages.batches import AsyncBatches
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from mcparser.agents.batching import (
    BatchDispatcher,
    BatchingClient,
    BatchRequestError,
    batch_params,
)
from mcparser.agents.outputs import Batch
from mcparser.agents.team import AgentTeam, ModelUsage, TaskReport
from mcparser.pricing import estimate
from mcparser.settings import Settings

# ---------------------------------------------------------------- fakes


def message(text="ok", tool=None, input_tokens=1000, output_tokens=100):
    block = (
        BetaToolUseBlock(type="tool_use", id="toolu_1", name=tool[0], input=tool[1])
        if tool
        else BetaTextBlock(type="text", text=text, citations=None)
    )
    return BetaMessage(
        id="msg_1", type="message", role="assistant", model="claude-opus-5-5", content=[block],
        stop_reason="tool_use" if tool else "end_turn", stop_sequence=None,
        usage=BetaUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )  # fmt: skip


def result(outcome):
    """A batch result entry body: a BetaMessage succeeds; a string names another outcome."""
    if isinstance(outcome, BetaMessage):
        return SimpleNamespace(type="succeeded", message=outcome)
    if outcome == "errored":
        return SimpleNamespace(type="errored", error=SimpleNamespace(error=SimpleNamespace(message="overloaded")))
    return SimpleNamespace(type=outcome)


class FakeBatches:
    """``client.beta.messages.batches``: answers each request with ``respond(params)`` after ``polls`` polls."""

    def __init__(self, respond=lambda params: message(), polls=0, poll_errors=0, create_error=None, cancel_error=None, hang_create=False):
        self.respond, self.polls, self.poll_errors = respond, polls, poll_errors
        self.create_error, self.cancel_error, self.hang_create = create_error, cancel_error, hang_create
        self.created, self.retrieved, self.cancelled, self.extra_results = [], 0, [], []
        self._requests = {}

    @staticmethod
    def counts(**kwargs):
        return SimpleNamespace(**{"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, **kwargs})

    async def create(self, *, requests, betas=None):
        if self.create_error:
            raise self.create_error
        if self.hang_create:
            await asyncio.Event().wait()
        self.created.append({"requests": requests, "betas": betas})
        batch_id = f"msgbatch_{len(self.created)}"
        self._requests[batch_id] = requests
        return SimpleNamespace(
            id=batch_id, processing_status="in_progress", request_counts=self.counts(processing=len(requests))
        )

    async def retrieve(self, message_batch_id, *, betas=None):
        self.retrieved += 1
        if self.retrieved <= self.poll_errors:
            raise ConnectionError("network blip")
        done = self.retrieved - self.poll_errors > self.polls
        outcomes = [self._outcome(item) for item in self._requests[message_batch_id]]
        counts = self.counts(**{kind: sum(o.type == kind for o in outcomes) for kind in ("succeeded", "errored", "expired", "canceled")})
        return SimpleNamespace(
            id=message_batch_id, processing_status="ended" if done else "in_progress", request_counts=counts
        )

    def _outcome(self, item):
        outcome = self.respond(item["params"])
        return result(outcome) if outcome is not None else SimpleNamespace(type="canceled")

    async def results(self, message_batch_id, *, betas=None):
        async def stream():
            for item in self._requests[message_batch_id]:
                outcome = self.respond(item["params"])
                if outcome is not None:  # None: the batch has no result for this request
                    yield SimpleNamespace(custom_id=item["custom_id"], result=result(outcome))
            for entry in self.extra_results:
                yield entry

        return stream()

    async def cancel(self, message_batch_id, *, betas=None):
        self.cancelled.append(message_batch_id)
        if self.cancel_error:
            raise self.cancel_error


class VirtualTime:
    """Sleeping advances a virtual clock and yields once, so polling takes no real time."""

    def __init__(self):
        self.now, self.slept = 0.0, []

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    def clock(self):
        return self.now


def dispatcher(batches, time=None, **options):
    time = time or VirtualTime()
    options = {"poll_seconds": 10, "max_poll_seconds": 30, "backoff": 2, "settle_seconds": 1, **options}
    return BatchDispatcher(batches, sleep=time.sleep, clock=time.clock, **options), time


async def request(d, text="q", betas=()):
    """One task in flight sending one request."""
    with d.task():
        return await d.submit({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": text}]}, betas)


def text_of(response):
    return response.content[0].text


# ---------------------------------------------------------------- request translation


def test_batch_params_drops_what_only_a_live_call_needs():
    params, betas = batch_params(
        {
            "model": "m", "max_tokens": 9, "messages": [], "system": omit, "tools": omit, "tool_choice": NOT_GIVEN,
            "stream": False, "timeout": NOT_GIVEN, "extra_headers": {"anthropic-beta": "b-2, b-1", "x-other": "1"},
            "extra_query": None, "extra_body": {"thinking_extra": 1}, "betas": ["b-3", "b-1"], "cache_control": {"type": "ephemeral"},
        }
    )  # fmt: skip
    assert params == {
        "model": "m", "max_tokens": 9, "messages": [], "cache_control": {"type": "ephemeral"}, "thinking_extra": 1
    }  # fmt: skip
    assert betas == ("b-1", "b-2", "b-3")
    assert batch_params({"model": "m", "extra_headers": None, "betas": omit}) == ({"model": "m"}, ())


@pytest.mark.parametrize("arguments, message", [
    ({"stream": True}, "cannot stream"),
    ({"speed": "fast"}, "does not support `speed`"),
    ({"fallbacks": []}, "does not support `fallbacks`"),
])  # fmt: skip
def test_batch_params_rejects_what_batches_cannot_do(arguments, message):
    with pytest.raises(UserError, match=message):
        batch_params({"model": "m", **arguments})


# ---------------------------------------------------------------- the dispatcher


@pytest.mark.anyio
async def test_requests_of_one_wave_share_a_batch():
    batches = FakeBatches(respond=lambda params: message(params["messages"][0]["content"] + "!"), polls=2)
    lines = []
    d, time = dispatcher(batches, log=lines.append)
    replies = await asyncio.gather(*(request(d, f"q{i}") for i in range(3)))
    assert [text_of(reply) for reply in replies] == ["q0!", "q1!", "q2!"]
    assert len(batches.created) == 1 and len(batches.created[0]["requests"]) == 3
    assert {item["custom_id"] for item in batches.created[0]["requests"]} == {"r000000001", "r000000002", "r000000003"}
    assert batches.created[0]["betas"] is None
    (record,) = d.records
    assert (record.id, record.size, record.succeeded, record.errored, record.error) == ("msgbatch_1", 3, 3, 0, None)
    assert record.ended and record.seconds >= 20  # two polls of 10 and 20 seconds, plus the settling
    assert lines == ["batch msgbatch_1: submitted 3 requests", "batch msgbatch_1: ended, 3 succeeded, 0 errored, 0 expired, 0 canceled"]
    assert time.slept[:2] == [1, 1]  # the queue is looked at twice (once to see it, once to see it stopped growing)
    await d.aclose()


@pytest.mark.anyio
async def test_polling_backs_off_up_to_the_limit():
    batches = FakeBatches(polls=4)
    d, time = dispatcher(batches)
    await request(d)
    polls = time.slept[2:]  # after the two settling sleeps
    assert polls == [10, 20, 30, 30, 30]
    await d.aclose()


@pytest.mark.anyio
async def test_a_second_request_of_each_task_makes_a_second_wave():
    batches = FakeBatches()
    d, _ = dispatcher(batches)

    async def conversation(i):
        with d.task():
            first = await d.submit({"messages": [f"a{i}"]})
            second = await d.submit({"messages": [f"b{i}"]})
        return text_of(first), text_of(second)

    assert await asyncio.gather(*(conversation(i) for i in range(4))) == [("ok", "ok")] * 4
    assert [len(call["requests"]) for call in batches.created] == [4, 4]
    assert [record.size for record in d.records] == [4, 4]
    await d.aclose()


@pytest.mark.anyio
async def test_waits_for_tasks_that_are_still_preparing_their_request():
    batches = FakeBatches()
    d, time = dispatcher(batches, max_wait_seconds=1000)

    async def slow():
        with d.task():  # in flight, but its request is ready only after 6 virtual seconds
            while time.now < 6:
                await asyncio.sleep(0)
            return await d.submit({"messages": ["slow"]})

    replies = await asyncio.gather(request(d, "fast"), slow())
    assert [len(call["requests"]) for call in batches.created] == [2]
    assert len(replies) == 2
    await d.aclose()


@pytest.mark.anyio
async def test_submits_after_the_maximum_wait_even_if_a_task_is_still_preparing():
    batches = FakeBatches()
    d, time = dispatcher(batches, max_wait_seconds=3)
    stop = asyncio.Event()

    async def never_ready():
        with d.task():
            await stop.wait()

    blocked = asyncio.create_task(never_ready())
    await asyncio.sleep(0)
    reply = await request(d, "alone")
    stop.set()
    await blocked
    assert text_of(reply) == "ok" and [len(call["requests"]) for call in batches.created] == [1]
    assert time.now >= 3
    await d.aclose()


@pytest.mark.anyio
async def test_full_batches_are_split_by_count_and_size():
    batches = FakeBatches()
    d, _ = dispatcher(batches, max_requests=2)
    await asyncio.gather(*(request(d, f"q{i}") for i in range(5)))
    assert sorted(len(call["requests"]) for call in batches.created) == [1, 2, 2]
    await d.aclose()

    batches = FakeBatches()
    d, _ = dispatcher(batches, max_bytes=150)  # one request is about 80 bytes
    await asyncio.gather(*(request(d, f"q{i}") for i in range(3)))
    assert sorted(len(call["requests"]) for call in batches.created) == [1, 1, 1]
    await d.aclose()


@pytest.mark.anyio
async def test_betas_are_a_header_of_the_batch_so_different_betas_split_it():
    batches = FakeBatches()
    d, _ = dispatcher(batches)
    await asyncio.gather(request(d, "a", ("beta-1",)), request(d, "b", ("beta-1",)), request(d, "c"))
    by_betas = {tuple(call["betas"] or ()): len(call["requests"]) for call in batches.created}
    assert by_betas == {("beta-1",): 2, (): 1}
    assert {tuple(record.betas) for record in d.records} == {("beta-1",), ()}
    await d.aclose()


@pytest.mark.anyio
async def test_a_request_that_failed_raises_in_its_own_task_only():
    outcomes = {"ok": message("fine"), "bad": "errored", "late": "expired", "gone": "canceled", "lost": None}
    batches = FakeBatches(respond=lambda params: outcomes[params["messages"][0]["content"]])
    d, _ = dispatcher(batches)
    outputs = await asyncio.gather(*(request(d, name) for name in outcomes), return_exceptions=True)
    assert text_of(outputs[0]) == "fine"
    errors = [str(error) for error in outputs[1:]]
    assert errors[0] == "request errored in msgbatch_1: overloaded"
    assert errors[1] == "request expired in msgbatch_1: expired"
    assert errors[2] == "request canceled in msgbatch_1: canceled"
    assert "has no result in msgbatch_1" in errors[3]
    assert all(isinstance(error, BatchRequestError) for error in outputs[1:])
    (record,) = d.records
    assert (record.succeeded, record.errored, record.expired, record.canceled) == (1, 1, 1, 2)  # the lost one counts as canceled in the fake
    await d.aclose()


@pytest.mark.anyio
async def test_unknown_results_are_ignored_and_a_gone_waiter_does_not_break_the_batch():
    batches = FakeBatches(polls=3)
    batches.extra_results = [SimpleNamespace(custom_id="stranger", result=result(message()))]
    d, _ = dispatcher(batches)
    kept = asyncio.create_task(request(d, "kept"))
    dropped = asyncio.create_task(request(d, "dropped"))
    while not batches.created:  # wait for the submission, then abandon one request while the batch runs
        await asyncio.sleep(0)
    dropped.cancel()
    assert text_of(await kept) == "ok"
    with pytest.raises(asyncio.CancelledError):
        await dropped
    await d.aclose()


@pytest.mark.anyio
async def test_polling_survives_a_few_failures_but_not_many():
    batches = FakeBatches(poll_errors=2, polls=1)
    lines = []
    d, _ = dispatcher(batches, log=lines.append)
    assert text_of(await request(d)) == "ok"
    assert sum("could not poll (ConnectionError)" in line for line in lines) == 2
    await d.aclose()

    batches = FakeBatches(poll_errors=99)
    d, _ = dispatcher(batches, poll_failures=2)
    with pytest.raises(BatchRequestError, match="batch failed: ConnectionError: network blip"):
        await request(d)
    (record,) = d.records
    assert record.error == "ConnectionError: network blip" and record.id == "msgbatch_1"
    await d.aclose()


@pytest.mark.anyio
async def test_a_batch_that_cannot_be_created_fails_its_requests():
    d, _ = dispatcher(FakeBatches(create_error=RuntimeError("too big")))
    outputs = await asyncio.gather(request(d, "a"), request(d, "b"), return_exceptions=True)
    assert all("batch failed: RuntimeError: too big" in str(error) for error in outputs)
    assert d.records[0].id is None and d.records[0].error == "RuntimeError: too big"
    await d.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_error", [None, RuntimeError("cannot cancel")])
async def test_closing_cancels_a_running_batch_on_the_server(cancel_error):
    batches = FakeBatches(polls=10**6, cancel_error=cancel_error)
    lines = []
    d, _ = dispatcher(batches, log=lines.append)
    waiting = asyncio.create_task(request(d))
    while not batches.created:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    await d.aclose()
    assert batches.cancelled == ["msgbatch_1"]
    assert "batch msgbatch_1: cancelled" in lines
    assert d.records[0].error == "cancelled"
    with pytest.raises(asyncio.CancelledError):
        await waiting


@pytest.mark.anyio
async def test_closing_while_a_batch_is_being_created_has_nothing_to_cancel_yet():
    batches = FakeBatches(hang_create=True)
    d, _ = dispatcher(batches)
    waiting = asyncio.create_task(request(d))
    for _ in range(10):
        await asyncio.sleep(0)
    await d.aclose()
    assert batches.cancelled == [] and d.records[0].error == "cancelled" and d.records[0].id is None
    with pytest.raises(asyncio.CancelledError):
        await waiting


@pytest.mark.anyio
async def test_closing_with_requests_still_queued_cancels_them():
    d, _ = dispatcher(FakeBatches(), settle_seconds=10**6)
    waiting = asyncio.create_task(request(d))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await d.aclose()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await d.aclose()  # closing twice is harmless


@pytest.mark.anyio
async def test_without_task_tracking_a_quiet_queue_is_submitted():
    batches = FakeBatches()
    d, _ = dispatcher(batches)
    reply = await d.submit({"messages": ["x"]})
    assert text_of(reply) == "ok" and len(batches.created) == 1
    reply = await d.submit({"messages": ["y"]})  # the loop is idle again and restarts for the next request
    assert len(batches.created) == 2
    await d.aclose()


# ---------------------------------------------------------------- the client


@pytest.mark.anyio
async def test_the_client_sends_beta_messages_through_the_dispatcher():
    batches = FakeBatches()
    d, _ = dispatcher(batches)
    client = BatchingClient(api_key="test", dispatcher=d)
    with client.task():
        reply = await client.beta.messages.create(
            model="claude-x", max_tokens=5, messages=[], stream=False, timeout=NOT_GIVEN, betas=["beta-1"]
        )
    assert text_of(reply) == "ok"
    assert batches.created[0]["betas"] == ["beta-1"]
    assert batches.created[0]["requests"][0]["params"] == {"model": "claude-x", "max_tokens": 5, "messages": []}
    assert str(client.base_url).startswith("https://") and client.timeout is not None  # the real client's attributes
    await d.aclose()


def test_the_client_builds_a_dispatcher_over_its_own_batches_api():
    client = BatchingClient(api_key="test", dispatcher_options={"poll_seconds": 7, "max_wait_seconds": 9})
    assert isinstance(client.dispatcher.batches, AsyncBatches)
    assert (client.dispatcher.poll_seconds, client.dispatcher.max_wait_seconds) == (7, 9)


@pytest.mark.anyio
async def test_pydantic_ai_runs_an_agent_over_the_batching_client():
    seen = []

    def respond(params):
        seen.append(params)
        return message("hello from a batch")

    d, _ = dispatcher(FakeBatches(respond=respond))
    client = BatchingClient(api_key="test", dispatcher=d)
    model = AnthropicModel("claude-opus-5-5", provider=AnthropicProvider(anthropic_client=client))
    with override_allow_model_requests(True), client.task():
        output = await Agent(model, instructions="Be brief.").run("hi")
    assert output.output == "hello from a batch"
    assert seen[0]["model"] == "claude-opus-5-5" and seen[0]["system"]
    assert "stream" not in seen[0] and "timeout" not in seen[0]
    await d.aclose()


# ---------------------------------------------------------------- pricing


def test_batch_prices_are_half_with_one_hour_cache_writes():
    live = estimate("anthropic:claude-opus-5-5", 1_000_000, 600_000, 100_000, 10_000)
    batch = estimate("anthropic:claude-opus-5-5", 1_000_000, 600_000, 100_000, 10_000, batch=True)
    # 300k uncached at $4, 600k reads at $0.20, 100k one-hour writes at $8, 10k output at $20, all halved
    assert batch == pytest.approx((0.3 * 4 + 0.6 * 0.2 + 0.1 * 8 + 0.01 * 20) / 2)
    assert batch < live
    assert estimate("openai:gpt-5.6", 1000, 0, 0, 100, batch=True) == pytest.approx(estimate("openai:gpt-5.6", 1000, 0, 0, 100) / 2)
    assert estimate("local:llama", 10, 0, 0, 10, batch=True) is None


def test_batch_cache_writes_are_priced_by_the_cache_lifetime():
    args = ("anthropic:claude-opus-5-5", 1_000_000, 0, 1_000_000, 0)
    assert estimate(*args, batch=True, cache_ttl="5m") == pytest.approx(1.25 * 4 / 2)  # a 5-minute write, halved
    assert estimate(*args, batch=True, cache_ttl="1h") == pytest.approx(2 * 4 / 2)  # a one-hour write, halved
    assert estimate(*args, batch=True) == estimate(*args, batch=True, cache_ttl="1h")
    assert estimate(*args, cache_ttl="1h") == pytest.approx(1.25 * 4)  # the live API always writes for 5 minutes


def test_model_usage_prices_batches():
    from pydantic_ai.usage import RunUsage

    usage = RunUsage(requests=1, input_tokens=1_000_000, output_tokens=0)
    written = RunUsage(requests=1, input_tokens=1_000_000, cache_write_tokens=1_000_000)
    live, batch = ModelUsage(), ModelUsage()
    live.add(usage, "claude-sonnet-5-5")
    batch.add(usage, "claude-sonnet-5-5", batch=True)
    assert (live.cost, batch.cost) == (pytest.approx(2.0), pytest.approx(1.0))
    short, long = ModelUsage(), ModelUsage()
    short.add(written, "claude-sonnet-5-5", batch=True, cache_ttl="5m")
    long.add(written, "claude-sonnet-5-5", batch=True, cache_ttl="1h")
    assert (short.cost, long.cost) == (pytest.approx(1.25), pytest.approx(2.0))


# ---------------------------------------------------------------- settings and the team


def test_non_anthropic_models_are_listed():
    settings = Settings(strong_model="openai:gpt-5.6", roles={"upos": {"model": "ollama:llama"}})
    found = settings.non_anthropic_models()
    assert ("lemma", "openai:gpt-5.6") in found and ("upos", "ollama:llama") in found
    assert not any(role == "scorer" for role, _ in found)
    assert Settings().non_anthropic_models() == []
    # with cheap_first, round 1 runs the efficient roles on the light model; the scorer is never downgraded
    cheap = Settings(cheap_first=True, light_model="google-gla:g").non_anthropic_models()
    assert {role for role, _ in cheap} == {"upos", "feats", "xpos"}


def test_batch_mode_needs_anthropic_models(store):
    with pytest.raises(UserError, match="Batch mode only works with Anthropic models, but .* uses openai:gpt-5.6"):
        AgentTeam(store, "pred", Settings(batch=True, strong_model="openai:gpt-5.6"))
    AgentTeam(store, "pred", Settings(batch=False, strong_model="openai:gpt-5.6"))
    # models given explicitly are used as they are, so they need not be Anthropic's
    AgentTeam(store, "pred", Settings(batch=True, model="openai:gpt-5.6"), models={"*": "test"})


def test_batch_mode_needs_an_api_key(store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    team = AgentTeam(store, "pred", Settings(batch=True))
    with pytest.raises(UserError, match="Set the ANTHROPIC_API_KEY"):
        team.model("lemma")


def test_team_builds_batching_models(store, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    lines = []
    team = AgentTeam(store, "pred", Settings(batch=True, batch_poll_seconds=5, batch_max_wait_seconds=60), on_batch=lines.append)
    model = team.model("lemma")
    assert isinstance(model, AnthropicModel) and team.model_name(model) == "claude-opus-5-5"
    assert team.model("lemma") is model and team.model("upos") is not model
    client = team._batching
    assert isinstance(client, BatchingClient) and team.model("xpos")._provider.client is client
    assert (client.dispatcher.poll_seconds, client.dispatcher.max_wait_seconds) == (5, 60)
    assert client.dispatcher.log == lines.append
    assert team.batched("lemma") and not AgentTeam(store, "pred", Settings(), models={}).batched("lemma")
    assert not AgentTeam(store, "pred", Settings(batch=True), models={"lemma": "x"}).batched("lemma")
    assert team.batch_records() == []
    assert team.model_settings("lemma")["cache"] == "1h"
    for ttl, expected in (("5m", "5m"), ("off", False)):
        assert AgentTeam(store, "pred", Settings(batch=True, batch_cache=ttl)).model_settings("lemma")["cache"] == expected
    assert AgentTeam(store, "pred", Settings(batch=True, cache=False)).model_settings("lemma")["cache"] is False
    assert AgentTeam(store, "pred", Settings()).model_settings("lemma")["cache"] is True
    assert AgentTeam(store, "pred", Settings(batch=False)).model("lemma") == "anthropic:claude-opus-5-5"


@pytest.mark.anyio
async def test_a_team_task_runs_through_a_batch_and_is_priced_as_one(store, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    batches = FakeBatches(respond=lambda params: message(tool=("final_result", {"summary": "done"})), polls=1)
    team = AgentTeam(store, "pred", Settings(batch=True))
    team._batching = BatchingClient(api_key="test-key", dispatcher=dispatcher(batches)[0])
    with override_allow_model_requests(True):
        async with team:
            report = await team.run("segmentation", "go", Batch("pred", ["p1"]))
    assert isinstance(report, TaskReport) and report.summary == "done"
    (call,) = batches.created
    params = call["requests"][0]["params"]
    assert params["model"] == "claude-opus-5-5" and params["tools"] and params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    (record,) = team.batch_records()
    assert (record.id, record.size, record.succeeded) == ("msgbatch_1", 1, 1) and record.ended
    total = team.total()
    assert total.requests == 1 and total.cost == pytest.approx(estimate("claude-opus-5-5", 1000, 0, 0, 100, batch=True))
    assert json.loads(record.model_dump_json())["id"] == "msgbatch_1"


def test_pipeline_forwards_batch_events_to_its_progress_callback(store):
    from mcparser.pipeline import Pipeline

    lines = []
    team = AgentTeam(store, "pred", Settings(), models={"*": "test"})
    Pipeline(store, "pred", team, Settings(), progress=lines.append)
    assert team.on_batch == lines.append
    own = lambda line: None  # noqa: E731
    team = AgentTeam(store, "pred", Settings(), models={"*": "test"}, on_batch=own)
    Pipeline(store, "pred", team, Settings(), progress=lines.append)
    assert team.on_batch is own
