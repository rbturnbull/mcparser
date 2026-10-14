import pytest

from mcparser.agents.prompts import TaskItem
from mcparser.agents.team import AgentTeam
from mcparser.corpus import CorpusError, CorpusStore
from mcparser.pipeline import Pipeline
from mcparser.rules import RuleSettings
from mcparser.settings import Settings
from scripted import cells, scored, scripted, sentences, values

# The fixture predictions: p1 "Vino al mercado ." (al has no LEMMA/UPOS), p2 "the dog".
# With a high min_count the tiny training data never triggers a rule, so the agents do the work.
NO_RULES = RuleSettings(min_count=50)


def pipeline(store, models, progress=None, **settings):
    settings = Settings(rules=NO_RULES, **settings)
    return Pipeline(store, "pred", AgentTeam(store, "pred", settings, models=models), settings, progress=progress)


@pytest.mark.anyio
async def test_score_then_refine_until_stable(store):
    messages = []
    models = {
        "scorer": scripted(scored(0.9)),
        "upos": scripted(values("upos", "ADP"), summary="fixed al"),
        "lemma": scripted(values("lemma", "al")),
        "*": scripted(),
    }
    report = await pipeline(store, models, progress=messages.append).run()

    first, second = report.rounds
    # one task per role, both sentences in the scorer's batch; "al" goes straight to the agents
    assert (first.scored, first.tasks, first.edits) == (2, 4, 2)
    assert (second.scored, second.tasks, second.edits) == (1, 0, 0)
    assert report.final_scored == 0
    assert [(t.round, t.sent_ids, t.role) for t in report.tasks] == [
        (1, ["p1", "p2"], "scorer"),
        (1, ["p1"], "upos"),
        (1, ["p1"], "lemma"),
        (1, ["p1"], "feats"),
        (1, ["p1"], "xpos"),
        (2, ["p1"], "scorer"),
    ]
    assert store.source("pred").sentence("p1").token("2").upos == "ADP"
    assert report.confidence == 0.902  # Vino's DEPREL "root" was confirmed by rule (0.99) once its head scored 0.9
    assert report.total.requests == 6  # one request per task
    assert "[round 1] upos p1: fixed al" in messages
    assert messages[0].startswith("Rules settled 0 cells")
    assert {edit.agent for edit in store.get_history("pred", limit=100)} == {"scorer", "upos", "lemma", "rules"}


@pytest.mark.anyio
async def test_scorer_sees_only_changed_cells_with_evidence(store):
    prompts = []

    def score(prompt):
        prompts.append(prompt)
        return scored(0.9)(prompt)

    models = {"scorer": scripted(score), "upos": scripted(values("upos", "ADP")), "lemma": scripted(values("lemma", "al")), "*": scripted()}
    await pipeline(store, models).run()
    second_round = prompts[1]
    assert [(c["token_id"], c["column"]) for c in cells(second_round)] == [("2", "upos"), ("2", "lemma"), ("2", "feats"), ("2", "xpos")]
    assert "Analyses in the trusted data" in second_round


@pytest.mark.anyio
async def test_rules_settle_cells_and_the_scorer_skips_them(train_path, pred_path):
    store = CorpusStore()
    store.add_source("train", train_path)
    store.add_source("pred", pred_path, mode="rw")
    settings = Settings(rules=RuleSettings(min_count=1, use_predictions=False))
    models = {"scorer": scripted(scored(0.9)), "*": scripted()}
    p = Pipeline(store, "pred", AgentTeam(store, "pred", settings, models=models), settings)
    report = await p.run(score_only=True)
    assert report.rules.cells["xpos"] >= 1
    p2 = store.source("pred").sentence("p2")
    assert p2.token("1").xpos == "DT"  # filled by the morphology rule
    assert ("1", "xpos") not in [(c.token_id, c.column) for c in p.cells_to_score("p2")]


@pytest.mark.anyio
async def test_stops_when_a_round_changes_nothing(store):
    models = {"scorer": scripted(scored(lambda c: 0.3 if c["column"] == "lemma" else 0.9)), "*": scripted()}
    report = await pipeline(store, models, rounds=3).run()
    assert len(report.rounds) == 1
    assert report.rounds[0].tasks == 4  # upos, xpos, lemma (both sentences in one batch), feats
    assert report.rounds[0].edits == 0
    assert report.final_scored == 0


@pytest.mark.anyio
async def test_round_limit_then_final_scoring_and_batches(store):
    seen = []

    def lemma(prompt):
        seen.append(sentences(prompt))
        return values("lemma", lambda c: f"x{c['token_id']}")(prompt)

    models = {"scorer": scripted(scored(lambda c: 0.3 if c["column"] == "lemma" else 0.9)), "lemma": scripted(lemma), "*": scripted()}
    report = await pipeline(store, models, rounds=1, batch_sentences=1, final_scoring=True).run()
    assert len(report.rounds) == 1
    assert report.final_scored == 2
    assert {"scorer", "lemma"} <= set(report.roles) and report.roles["lemma"].tasks == 2
    assert all(task.usage is not None and task.usage.requests == 1 for task in report.tasks)
    assert seen == [["p2"], ["p1"]]  # one sentence per task; p2 is more confident, so it goes first


@pytest.mark.anyio
async def test_flags_are_sent_to_the_responsible_role(store):
    store.flag_issue("pred", "p2", "lemma", "should be 'dog'?", token_id="2", agent="upos")
    prompts = []

    def lemma(prompt):
        prompts.append(prompt)
        return {"resolved": [{"flag_id": 1, "status": "dismissed", "note": "already dog"}]}

    models = {"scorer": scripted(scored(0.9)), "lemma": scripted(lemma), "*": scripted()}
    report = await pipeline(store, models).run()
    assert "flag 1 (word 2, from upos): should be 'dog'?" in prompts[0]
    assert store.list_flags("pred", status="dismissed").flags[0].resolved_by == "lemma"
    assert report.rounds[0].open_flags == 0


@pytest.mark.anyio
async def test_score_only_and_sentence_filter(store):
    settings = Settings(rules=NO_RULES)
    team = AgentTeam(store, "pred", settings, models={"*": scripted(scored(0.9))})
    report = await Pipeline(store, "pred", team, settings, sent_ids=["p2"]).run(score_only=True)
    assert [(r.scored, r.tasks) for r in report.rounds] == [(1, 0)]
    assert [t.sent_ids for t in report.tasks] == [["p2"]]
    with pytest.raises(CorpusError, match="No sentence 'zz'"):
        Pipeline(store, "pred", team, settings, sent_ids=["zz"])


@pytest.mark.anyio
async def test_agent_failures_are_recorded_and_the_run_continues(store):
    messages = []
    model = scripted(lambda prompt: {"scores": "bad line"})
    report = await pipeline(store, {"*": model}, progress=messages.append, rounds=1).run(score_only=True)
    assert len(report.tasks) == 1
    assert report.tasks[0].error.startswith("UnexpectedModelBehavior")
    assert "expected SENTENCE TOKEN_ID COLUMN CONFIDENCE" in model.retries[0]
    assert report.tasks[0].retries == []  # a failed task has no result to read them from
    assert any("scorer p1, p2 FAILED" in message for message in messages)


@pytest.mark.anyio
async def test_concurrent_tasks(store):
    models = {
        "scorer": scripted(scored(lambda c: 0.3 if c["column"] == "upos" else 0.9)),
        "upos": scripted(values("upos", "ADP")),
        "*": scripted(),
    }
    report = await pipeline(store, models, concurrency=2, batch_sentences=1).run()
    assert sorted((t.sent_ids[0], t.role) for t in report.tasks if t.round == 1) == [
        ("p1", "feats"), ("p1", "lemma"), ("p1", "scorer"), ("p1", "upos"), ("p1", "xpos"),
        ("p2", "scorer"), ("p2", "upos"),
    ]  # fmt: skip
    assert report.rounds[0].tasks == 5


@pytest.mark.anyio
async def test_unused_columns_are_settled_without_agents(store, tmp_path):
    no_xpos = tmp_path / "noxpos.conllu"
    no_xpos.write_text("# sent_id = n1\n1\tdog\tdog\tNOUN\t_\tNumber=Sing\t0\troot\t_\t_\n\n", encoding="utf-8")
    other = CorpusStore()
    other.add_source("train", no_xpos)
    other.add_source("pred", store.source("pred").path, mode="rw")
    other.set_cell("pred", "p2", "1", "xpos", "DT")
    messages = []
    p = pipeline(other, {"*": scripted(scored(0.9))}, progress=messages.append, rounds=1)
    report = await p.run(score_only=True)
    assert "xpos" not in p.roles and "feats" in p.roles
    assert all(t.role == "scorer" for t in report.tasks)
    word = other.source("pred").sentence("p2").token("1")
    assert (word.xpos, other.source("pred").confidence(word, "xpos")) == ("_", 1.0)
    assert "XPOS is never used in the trusted data" in messages[0]
    assert p.settle_unused_columns() == []  # already settled
    fresh = Pipeline(other, "pred", p.team, Settings(), sent_ids=["p2"])
    assert fresh.settle_unused_columns() == ["xpos"]  # nothing left to change in p2


def test_helpers(store, tmp_path):
    p = Pipeline(store, "pred", AgentTeam(store, "pred", Settings()), Settings(roles={"xpos": {"enabled": False}}))
    assert p.roles == ["segmentation", "upos", "lemma", "head", "deprel", "feats"]
    assert [c.column for c in p.cells("p2")][:6] == ["form", "upos", "lemma", "head", "deprel", "feats"]
    assert p.sentences(order="confidence_desc") == ["p1", "p2"]

    keys = lambda: [(c.token_id, c.column) for c in p.cells_to_score("p1")]  # noqa: E731
    assert len(keys()) == 4 * 6 - 3 and ("2", "lemma") not in keys()  # al: empty LEMMA/UPOS, FEATS needs UPOS
    store.score_cells("pred", "p1", {"1:upos": 0.9}, agent="scorer")
    assert ("1", "upos") not in keys() and len(keys()) == 20
    store.set_cell("pred", "p1", "1", "upos", "VERB", agent="upos")
    assert ("1", "upos") in keys()
    store.set_cell("pred", "p1", "1", "lemma", "ir", agent="rules")
    assert ("1", "lemma") not in keys()  # set by code: not re-checked
    store.score_cells("pred", "p1", {f"{i}:*": 0.9 for i in "1234"}, agent="scorer")
    assert not p.needs_scoring("p1")
    store.split_word("pred", "p1", "3", ["merca", "do"], agent="segmentation")
    assert len(keys()) == 5 * 6 - 3 - 4  # everything again, except the unannotated cells of "al" and new "do"
    store.flag_issue("pred", "p1", "upos", "x", agent="scorer")
    assert p.edits_since(0) == 2  # rules' edits do not count

    item = lambda n: TaskItem("s", [None] * n)  # noqa: E731
    p.settings.batch_sentences, p.settings.batch_cells = 2, 5
    assert [len(batch) for batch in p.batches([item(2), item(2), item(2), item(6), item(0), item(1)])] == [2, 1, 1, 2]

    empty = tmp_path / "empty.conllu"
    empty.write_text("", encoding="utf-8")
    other = CorpusStore()
    other.add_source("pred", empty, mode="rw")
    assert Pipeline(other, "pred", AgentTeam(other, "pred", Settings()), Settings()).confidence() == 0.0
