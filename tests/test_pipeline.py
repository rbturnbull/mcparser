import pytest

from llmparser.agents.team import AgentTeam
from llmparser.corpus import CorpusError, CorpusStore
from llmparser.pipeline import Pipeline
from llmparser.settings import Settings
from scripted import cells, nothing, scripted, sent_id


def scorer(low=lambda cell: False):
    """Scores every cell 0.9, or 0.3 when ``low(cell)``."""

    def decide(prompt):
        updates = [
            {"token_id": c["token_id"], "column": c["column"], "confidence": 0.3 if low(c) else 0.9} for c in cells(prompt)
        ]
        return [("update_cells", {"source": "pred", "sent_id": sent_id(prompt), "updates": updates})]

    return scripted(decide, summary="scored")


def upos_fixer(prompt):
    return [
        (
            "update_cells",
            {
                "source": "pred",
                "sent_id": sent_id(prompt),
                "updates": [{"token_id": c["token_id"], "column": "upos", "value": "ADP", "confidence": 0.9} for c in cells(prompt)],
            },
        )
    ]


def pipeline(store, models, progress=None, **settings):
    settings = Settings(**settings)
    return Pipeline(store, "pred", AgentTeam(store, "pred", settings, models=models), settings, progress=progress)


@pytest.mark.anyio
async def test_score_then_refine_until_stable(store):
    messages = []
    low_upos_of_al = lambda c: c["form"] == "al" and c["column"] == "upos" and c["value"] == "_"  # noqa: E731
    models = {"scorer": scorer(low_upos_of_al), "upos": scripted(upos_fixer, summary="fixed al"), "*": scripted(nothing)}
    report = await pipeline(store, models, progress=messages.append).run()

    first, second = report.rounds
    assert (first.scored, first.tasks, first.edits) == (2, 1, 1)
    assert (second.scored, second.tasks, second.edits) == (1, 0, 0)
    assert report.final_scored == 0
    assert [(t.round, t.sent_id, t.role, t.summary) for t in report.tasks] == [
        (1, "p1", "scorer", "scored"),
        (1, "p2", "scorer", "scored"),
        (1, "p1", "upos", "fixed al"),
        (2, "p1", "scorer", "scored"),
    ]
    assert store.source("pred").sentence("p1").token("2").upos == "ADP"
    assert report.confidence == 0.9
    assert report.requests == 8
    assert "[round 1] p1 upos: fixed al" in messages
    assert any(message.startswith("Round 2:") for message in messages)


@pytest.mark.anyio
async def test_stops_when_a_round_changes_nothing_and_rescores_at_the_end(store):
    models = {"scorer": scorer(lambda c: c["column"] == "lemma"), "*": scripted(nothing)}
    report = await pipeline(store, models, rounds=3).run()
    assert len(report.rounds) == 1
    assert report.rounds[0].tasks == 2  # the lemma agent on p1 and p2
    assert report.rounds[0].edits == 0
    assert report.final_scored == 0


@pytest.mark.anyio
async def test_round_limit_then_final_scoring(store):
    calls = []

    def lemma(prompt):
        calls.append(sent_id(prompt))
        return [("set_cell", {"source": "pred", "sent_id": sent_id(prompt), "token_id": "1", "column": "lemma", "value": f"x{len(calls)}"})]

    models = {"scorer": scorer(lambda c: c["column"] == "lemma"), "lemma": scripted(lemma), "*": scripted(nothing)}
    report = await pipeline(store, models, rounds=1).run()
    assert len(report.rounds) == 1
    assert report.final_scored == 2
    assert calls == ["p1", "p2"]


@pytest.mark.anyio
async def test_flags_are_sent_to_the_responsible_role(store):
    store.flag_issue("pred", "p2", "lemma", "should be 'dog'?", token_id="2", agent="upos")
    prompts = []

    def lemma(prompt):
        prompts.append(prompt)
        return [("resolve_flag", {"source": "pred", "flag_id": 1, "status": "dismissed", "note": "already dog"})]

    models = {"scorer": scorer(), "lemma": scripted(lemma), "*": scripted(nothing)}
    report = await pipeline(store, models).run()
    assert "flag 1 (word 2, from upos): should be 'dog'?" in prompts[0]
    assert store.list_flags("pred", status="dismissed").flags[0].resolved_by == "lemma"
    assert report.rounds[0].open_flags == 0


@pytest.mark.anyio
async def test_score_only_and_sentence_filter(store):
    settings = Settings()
    team = AgentTeam(store, "pred", settings, models={"*": scorer()})
    report = await Pipeline(store, "pred", team, settings, sent_ids=["p2"]).run(score_only=True)
    assert [(r.scored, r.tasks) for r in report.rounds] == [(1, 0)]
    assert [t.sent_id for t in report.tasks] == ["p2"]
    with pytest.raises(CorpusError, match="No sentence 'zz'"):
        Pipeline(store, "pred", team, settings, sent_ids=["zz"])


@pytest.mark.anyio
async def test_agent_failures_are_recorded_and_the_run_continues(store):
    def loop_forever(prompt):
        return [("get_sentence", {"source": "pred", "sent_id": sent_id(prompt)})]

    messages = []
    model = scripted(loop_forever)
    original = model.function
    model.function = lambda messages, info: original(messages[:1], info)  # always asks for another tool call
    report = await pipeline(store, {"*": model}, progress=messages.append, request_limit=2, rounds=1).run(score_only=True)
    assert len(report.tasks) == 2
    assert all(task.error and task.error.startswith("UsageLimitExceeded") for task in report.tasks)
    assert any("p1 scorer FAILED" in message for message in messages)


@pytest.mark.anyio
async def test_concurrent_sentences(store):
    models = {"scorer": scorer(lambda c: c["column"] == "upos"), "upos": scripted(upos_fixer), "*": scripted(nothing)}
    report = await pipeline(store, models, concurrency=2).run()
    assert sorted((t.sent_id, t.role) for t in report.tasks if t.round == 1) == [
        ("p1", "scorer"), ("p1", "upos"), ("p2", "scorer"), ("p2", "upos"),
    ]  # fmt: skip
    assert report.rounds[0].tasks == 2


def test_helpers(store, tmp_path):
    p = Pipeline(store, "pred", AgentTeam(store, "pred", Settings()), Settings(roles={"xpos": {"enabled": False}}))
    assert p.roles == ["segmentation", "upos", "lemma", "feats", "head", "deprel"]
    assert p.needs_scoring("p1")
    assert [c.column for c in p.cells("p2")][:6] == ["form", "upos", "lemma", "feats", "head", "deprel"]
    assert p.sentences(order="confidence_desc") == ["p1", "p2"]

    store.update_cells("pred", "p1", [{"token_id": "1", "column": "upos", "confidence": 0.9}], agent="scorer")
    assert not p.needs_scoring("p1")
    store.set_cell("pred", "p1", "1", "upos", "VERB", agent="upos")
    assert p.needs_scoring("p1")
    store.flag_issue("pred", "p1", "upos", "x", agent="scorer")
    assert p.edits_since(0) == 1

    empty = tmp_path / "empty.conllu"
    empty.write_text("", encoding="utf-8")
    other = CorpusStore()
    other.add_source("pred", empty, mode="rw")
    assert Pipeline(other, "pred", AgentTeam(other, "pred", Settings()), Settings()).confidence() == 0.0
