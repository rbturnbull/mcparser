import json

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

from mcparser.agents.outputs import Annotation, Applier, Batch, FlagRequest, FlagResolution, Scores, parse_scores, parse_values
from mcparser.agents.prompts import TaskItem, instructions, sentence_block, task_prompt
from mcparser.agents.roles import READ_TOOLS, ROLES, SEGMENTATION_TOOLS
from mcparser.agents.team import AgentTeam, TaskReport
from mcparser.agents.toolset import RoleToolset
from mcparser.corpus.confidence import read_confidence
from mcparser.corpus.models import CellRef, Flag
from mcparser.settings import RoleSettings, Settings
from scripted import scripted, values

# ---------------------------------------------------------------- roles and prompts


def test_roles():
    assert ROLES["upos"].column == "upos"
    assert SEGMENTATION_TOOLS <= ROLES["segmentation"].tools and "update_cells" in ROLES["segmentation"].tools
    assert ROLES["lemma"].tools == READ_TOOLS == ROLES["scorer"].tools
    assert "relation_stats" in READ_TOOLS
    assert ROLES["segmentation"].edits_with_tools and not ROLES["scorer"].edits_with_tools
    assert ROLES["scorer"].is_scorer and not ROLES["head"].is_scorer


def test_instructions():
    text = instructions(ROLES["head"], "pred", ["train", "silver"], "Mind the particles.")
    assert "conventions of the trusted treebank\n(train, silver)" in text
    assert '"s1 3 4 0.95"' in text and "any other HEAD value" in text
    assert text.endswith("Mind the particles.")
    scorer = instructions(ROLES["scorer"], "pred", [])
    assert "treebank\n(none)" in scorer and '"s1 * * 0.95"' in scorer
    assert "split_word" in instructions(ROLES["segmentation"], "pred", ["train"])
    assert "Relations used in the treebank (with counts; use its subtypes):\nroot 3" in instructions(ROLES["deprel"], "pred", [], relations="root 3")
    assert "Relations used" not in instructions(ROLES["upos"], "pred", [], relations="root 3")


def test_task_prompts():
    cells = [CellRef(sent_id="s", token_id="2", form="al", column="upos", value="_", confidence=0.0)]
    flags = [
        Flag(id=3, sent_id="s", token_id="2", column="upos", reason="looks like ADP", suggestion="ADP", raised_by="lemma", created="t"),
        Flag(id=4, sent_id="s", column="upos", reason="check all", raised_by="scorer", created="t"),
    ]
    conllu = "# sent_id = s\n1\tVino\tvenir\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No|Conf=upos:0.80"
    text = task_prompt(ROLES["upos"], "pred", [TaskItem("s", cells, flags, conllu), TaskItem("t", [], flags[1:])], "EVIDENCE")
    assert text.splitlines() == [
        'Source "pred". 2 sentence(s).',
        "",
        "## s1 = sentence s",
        "# sent_id = s",
        "1\tVino\tvenir\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No",
        "UPOS cells to annotate or review: 2 at 0.00",
        "Open flags for UPOS:",
        "- flag 3 (word 2, from lemma): looks like ADP Suggested: 'ADP'.",
        "- flag 4 (whole sentence, from scorer): check all",
        "",
        "## s2 = sentence t",
        "Open flags for UPOS:",
        "- flag 4 (whole sentence, from scorer): check all",
        "",
        "EVIDENCE",
    ]
    scorer = task_prompt(ROLES["scorer"], "pred", [TaskItem("s", cells, flags)])
    assert "Cells to score: UPOS 2" in scorer and "Open flags" not in scorer
    assert sentence_block("1\ta\t_\t_\t_\t_\t0\troot\t_\tConf=upos:0.10") == "1\ta\t_\t_\t_\t_\t0\troot"
    assert sentence_block("1-2\tal\t_\t_\t_\t_\t_\t_\t_\t_") == "1-2\tal"
    head = CellRef(sent_id="s", token_id="2", form="al", column="head", value="_", confidence=0.0)
    label = CellRef(sent_id="s", token_id="3", form="x", column="deprel", value="obl", confidence=0.4)
    syntax = task_prompt(ROLES["syntax"], "pred", [TaskItem("s", [head, label], flags[:1])])
    assert "HEAD cells to annotate or review: 2 at 0.00\nDEPREL cells to annotate or review: 3 at 0.40" in syntax
    assert "Open flags for HEAD/DEPREL:" in syntax
    assert '"s1 3 4 nsubj 0.95 0.7"' in instructions(ROLES["syntax"], "pred", ["train"], relations="root 3")


def test_compact_cell_lists():
    def ref(token_id, column="upos", confidence=0.0):
        return CellRef(sent_id="s", token_id=token_id, form="x", column=column, value="_", confidence=confidence)

    many = [ref("1"), ref("2"), ref("3"), ref("5"), ref("6", confidence=0.6), ref("7", confidence=0.6), ref("7.1")]
    assert "UPOS cells to annotate or review: 1..3 5 at 0.00, 6..7 at 0.60, 7.1 at 0.00" in task_prompt(
        ROLES["upos"], "pred", [TaskItem("s", many)]
    )
    scored = [ref("1"), ref("2"), ref("1", "head"), ref("4", "head")]
    assert "Cells to score: UPOS 1..2; HEAD 1 4" in task_prompt(ROLES["scorer"], "pred", [TaskItem("s", scored)])


# ---------------------------------------------------------------- toolset permissions


def toolset(role):
    return RoleToolset(wrapped=None, role=ROLES[role], target="pred")


@pytest.mark.parametrize(
    ("role", "name", "args", "message"),
    [
        ("lemma", "update_cells", {"source": "pred"}, "not available"),
        ("lemma", "split_word", {"source": "pred"}, "not available"),
        ("segmentation", "split_word", {"source": "train"}, "only edit the source 'pred'"),
        ("segmentation", "update_cells", {"source": "pred", "updates": [{"column": "upos", "value": "X"}]}, "only change form"),
        ("segmentation", "update_cells", {"source": "pred", "updates": [{"column": "pos", "value": "X"}]}, "flag_issue with column='upos'"),
    ],
)
def test_toolset_refuses(role, name, args, message):
    with pytest.raises(ModelRetry, match=message):
        toolset(role).check(name, args)


def test_toolset_normalises_updates():
    normalise = RoleToolset.normalise
    assert normalise("split_word", {"x": 1}) == {"x": 1}
    assert normalise("update_cells", {"updates": '[{"column": "form"}]'})["updates"] == [{"column": "form"}]
    for bad in ("not json", '["x"]', None, {"column": "form"}):
        with pytest.raises(ModelRetry, match="must be a list"):
            normalise("update_cells", {"updates": bad})


def test_scorer_column_rules():
    with pytest.raises(ModelRetry, match="only send confidences"):
        toolset("scorer")._check_column("upos", has_value=True)
    with pytest.raises(ModelRetry, match="only change form"):
        toolset("scorer")._check_column("misc", has_value=False)


def test_toolset_allows():
    toolset("segmentation").check("update_cells", {"source": "pred", "updates": [{"column": "FORM", "confidence": 0.9}]})
    toolset("segmentation").check("update_cells", {"source": "pred", "updates": None})
    toolset("segmentation").check("split_word", {"source": "pred"})
    toolset("segmentation").check("resolve_flag", {"source": "train"})
    toolset("head").check("find_arcs", {})


# ---------------------------------------------------------------- compact answers


def batch(*ids):
    return Batch("pred", list(ids))


def test_parse_values():
    updates, errors = parse_values("p1 2 a el 0.9\n# comment\n\np1 3 x\nzz 1 X 0.5\np1 1 X high\np2 1 DET 1.5", ROLES["lemma"], batch("p1", "p2"))
    assert [(u.token_id, u.column, u.value, u.confidence) for u in updates["p1"]] == [("2", "lemma", "a el", 0.9)]
    assert updates["p2"][0].confidence == 1.0
    assert errors == [
        "line 2: expected SENTENCE TOKEN_ID VALUE CONFIDENCE, got 'p1 3 x'",
        "line 3: sentence 'zz' is not in this task",
        "line 4: confidence 'high' is not a number",
    ]


def test_batch_labels():
    labelled = batch("long-id-1", "long-id-2")
    assert labelled.label("long-id-2") == "s2"
    assert [labelled.resolve(name) for name in ("s1", "long-id-2", "s3", "s0", "x1", "s")] == [
        "long-id-1", "long-id-2", None, None, None, None
    ]
    updates, errors = parse_values("s2 1 X 0.9", ROLES["upos"], labelled)
    assert list(updates) == ["long-id-2"] and not errors


def test_parse_scores_wildcards_cover_listed_cells():
    listed = Batch("pred", ["p1", "p2"], cells={"p1": [("1", "upos"), ("1", "lemma"), ("2", "upos")], "p2": [("1", "head")]})
    scores, errors = parse_scores("s1 * * 0.95\ns1 1 * 0.6\ns1 2 upos 0.3\ns2 * head 0.8\ns2 4 * 0.5", listed)
    assert scores == {"p1": {"1:upos": 0.6, "1:lemma": 0.6, "2:upos": 0.3}, "p2": {"1:head": 0.8}}
    assert errors == ["line 5: no listed cell matches 4 *"]
    scores, errors = parse_scores("p1 * * 0.9", batch("p1"))
    assert errors == ["line 1: TOKEN_ID * is only allowed for the cells listed in the task"]


def test_parse_scores():
    scores, errors = parse_scores("p1 1 * 0.9\np1 2 POS 0.4\np1 2\nzz 1 upos 0.5\np1 1 upos x", batch("p1"))
    assert scores == {"p1": {"1:*": 0.9, "2:upos": 0.4}}
    assert [e.split(":")[0] for e in errors] == ["line 3", "line 4", "line 5"]


def test_applier_annotation(store):
    applier = Applier(store, ROLES["upos"])
    answer = Annotation(
        values="p1 2 ADP 0.9",
        flags=[FlagRequest(sent_id="p1", token_id="2", column="lemma", reason="should be 'a'", suggestion="a")],
        resolved=[FlagResolution(flag_id=99, status="dismissed")],
    )
    assert applier.annotation(answer, batch("p1")) is answer
    assert store.source("pred").sentence("p1").token("2").upos == "ADP"
    assert store.list_flags("pred").flags[0].raised_by == "upos"
    with pytest.raises(ModelRetry, match="sentence p1: 'NOUNY' is not a universal POS tag"):
        applier.annotation(Annotation(values="p1 2 NOUNY 0.9"), batch("p1"))
    applier.annotation(Annotation(values="", flags=[FlagRequest(sent_id="s1", column="lemma", reason="y")]), batch("p1"))
    assert store.list_flags("pred").flags[-1].sent_id == "p1"
    with pytest.raises(ModelRetry, match="flag on sentence 'p2', which is not in this task"):
        applier.annotation(Annotation(values="", flags=[FlagRequest(sent_id="p2", column="lemma", reason="x")]), batch("p1"))
    flag = store.flag_issue("pred", "p1", "upos", "check", agent="scorer")
    applier.annotation(Annotation(values="", resolved=[FlagResolution(flag_id=flag.id, status="resolved", note="ok")]), batch("p1"))
    assert store.list_flags("pred", status="resolved").flags[0].resolved_by == "upos"


def test_applier_rejects_broken_trees(store):
    applier = Applier(store, ROLES["head"])
    with pytest.raises(ModelRetry, match="Words 1 -> 2 form a cycle"):
        applier.annotation(Annotation(values="p2 1 2 0.9\np2 2 1 0.9"), batch("p2"))
    assert store.source("pred").sentence("p2").token("2").head == "0"  # nothing applied
    applier.annotation(Annotation(values="p2 1 2 0.95"), batch("p2"))
    assert read_confidence(store.source("pred").sentence("p2").token("1"))["head"] == 0.95


def test_parse_syntax():
    updates, errors = parse_values(
        "s1 2 3 obl 0.9\ns1 3 1 obl 0.8 0.6\ns1 4 1\nzz 1 0 root 0.9\ns1 1 0 root high", ROLES["syntax"], batch("p1")
    )
    assert [(u.token_id, u.column, u.value, u.confidence) for u in updates["p1"]] == [
        ("2", "head", "3", 0.9), ("2", "deprel", "obl", 0.9), ("3", "head", "1", 0.8), ("3", "deprel", "obl", 0.6),
    ]
    assert [e.split(":")[0] for e in errors] == ["line 3", "line 4", "line 5"]
    assert "expected SENTENCE TOKEN_ID HEAD DEPREL CONFIDENCE" in errors[0]


def test_applier_joint_syntax(store):
    applier = Applier(store, ROLES["syntax"])
    head_flag = store.flag_issue("pred", "p2", "head", "the head?", token_id="1", agent="scorer")
    label_flag = store.flag_issue("pred", "p2", "deprel", "the label?", token_id="2", agent="scorer")
    with pytest.raises(ModelRetry, match="form a cycle"):
        applier.annotation(Annotation(values="s1 1 2 det 0.9\ns1 2 1 root 0.9"), batch("p2"))
    with pytest.raises(ModelRetry, match="never occurs in the treebank's DEPREL"):
        applier.annotation(Annotation(values="s1 1 2 article 0.9"), batch("p2"))
    store.set_cell("pred", "p2", "1", "head", "0", confidence=0.5)
    store.set_cell("pred", "p2", "2", "head", "1", confidence=0.5)
    applier.annotation(Annotation(values="s1 1 2 det 0.9 0.8\ns1 2 0 root 0.95"), batch("p2"))
    words = store.source("pred").sentence("p2").words
    assert [(w.head, w.deprel) for w in words] == [("2", "det"), ("0", "root")]
    assert read_confidence(words[0])["deprel"] == 0.8  # not marked for review: the same answer set it
    flags = {f.id: f for f in store.list_flags("pred", status="all").flags}
    assert flags[head_flag.id].status == "resolved" and flags[label_flag.id].status == "open"  # p2:2 deprel unchanged


def test_doubled_relations_are_questioned_once(store, monkeypatch):
    monkeypatch.setattr(store, "single_relations", lambda: {"obl": (100, 1)})
    applier = Applier(store, ROLES["deprel"])
    task = batch("p1")
    with pytest.raises(ModelRetry, match=r"s1: word 1 would have 3 'obl' dependents \(2 al, 3 mercado, 4 \.\)"):
        applier.annotation(Annotation(values="p1 4 obl 0.9"), task)
    assert store.source("pred").sentence("p1").token("4").deprel == "punct"  # nothing applied
    applier.annotation(Annotation(values="p1 4 obl 0.9"), task)  # confirmed: accepted
    assert store.source("pred").sentence("p1").token("4").deprel == "obl"
    assert Applier(store, ROLES["upos"])._doubled({}, task) == []


def test_applier_scores(store):
    applier = Applier(store, ROLES["scorer"])
    applier.scores(Scores(scores="p2 1 * 0.9\np2 1 upos 0.4", flags=[FlagRequest(sent_id="p2", column="upos", reason="x")]), batch("p2"))
    assert read_confidence(store.source("pred").sentence("p2").token("1"))["upos"] == 0.4
    assert store.list_flags("pred").total == 1
    with pytest.raises(ModelRetry, match="No token with id '9'"):
        applier.scores(Scores(scores="p2 9 upos 0.5"), batch("p2"))


# ---------------------------------------------------------------- team


@pytest.mark.anyio
async def test_team_applies_compact_answers_and_retries(store):
    attempts = []

    def answer(prompt):
        attempts.append(len(model.retries))
        return {"values": "p1 2 NOUNY 0.7" if not model.retries else "p1 2 ADP 0.7", "summary": "tagged al"}

    model = scripted(answer, tools=lambda prompt: [("analyse_form", {"form": "el"})])
    team = AgentTeam(store, "pred", Settings(), models={"upos": model})
    async with team:
        report = await team.run("upos", "## s1 = sentence p1\n- word 2 (al): UPOS = '_', confidence 0.00", batch("p1"))
    assert report.summary == "tagged al"
    assert attempts == [0, 1]
    assert "not a universal POS tag" in model.retries[0]
    assert any(text.startswith("2 matches. By lemma upos xpos feats: el DET") for text in model.returns)
    assert store.source("pred").sentence("p1").token("2").upos == "ADP"
    assert [(e.agent, e.column) for e in store.get_history("pred")] == [("upos", "upos")]
    usage = team.total()
    assert usage.requests == 3 and usage.input_tokens > 0 and usage.output_tokens > 0 and usage.cost is None
    logged = []
    async with team:
        await team.run("upos", "## s1 = sentence p1", batch("p1"), retries=logged)
    assert len(logged) == 1 and "not a universal POS tag" in logged[0]


@pytest.mark.anyio
async def test_segmentation_edits_with_tools(store):
    calls = lambda prompt: [("split_word", {"source": "pred", "sent_id": "p1", "token_id": "2", "forms": ["a", "el"], "multiword": True})]  # noqa: E731
    model = scripted(tools=calls, summary="split al")
    seen = {}
    original = model.function

    def spy(messages, info):
        seen.setdefault("tools", {tool.name: tool for tool in info.function_tools})
        return original(messages, info)

    model.function = spy
    team = AgentTeam(store, "pred", Settings(), models={"segmentation": model})
    async with team:
        report = await team.run("segmentation", "go", batch("p1"))
    assert isinstance(report, TaskReport) and report.summary == "split al"
    assert "agent" not in json.dumps(seen["tools"]["split_word"].parameters_json_schema["properties"])
    assert {edit.agent for edit in store.get_history("pred")} == {"segmentation"}
    assert any(text.startswith("OK: 1 change(s) saved.") for text in model.returns)


def test_team_models_and_agents(store):
    settings = Settings(strong_model="global", roles={"lemma": RoleSettings(model="lemma-model", instructions="Extra.")})
    team = AgentTeam(store, "pred", settings, models={"head": TestModel()})
    assert team.trusted == ["train"]
    assert isinstance(team.model("head"), TestModel)
    assert team.model("lemma") == "lemma-model"
    assert team.model("deprel") == "global"
    assert AgentTeam(store, "pred", settings, models={"*": "star"}).model("upos") == "star"
    agent = team.agent("lemma")
    assert team.agent("lemma") is agent
    assert agent.name == "lemma"
    assert team.model_settings("lemma") == {"cache": True}
    assert AgentTeam(store, "pred", Settings(strong_thinking="low")).model_settings("head") == {"cache": True, "thinking": "low"}
    assert team.model_name(TestModel()) == "test"
    assert team.relations().startswith("det 3, root 3")
    assert values("upos", "X")("") == {"values": ""}


def test_applier_rejects_values_the_treebank_never_uses(store):
    with pytest.raises(ModelRetry, match="SCONJ never occurs in the treebank's XPOS"):
        Applier(store, ROLES["xpos"]).annotation(Annotation(values="p2 1 SCONJ 0.9"), batch("p2"))
    with pytest.raises(ModelRetry, match="Case=Abl never occurs in the treebank's FEATS"):
        Applier(store, ROLES["feats"]).annotation(Annotation(values="p2 1 Case=Abl|Number=Sing 0.9"), batch("p2"))
    Applier(store, ROLES["xpos"]).annotation(Annotation(values="p2 1 DT 0.9\np2 2 _ 0.9"), batch("p2"))
    assert store.source("pred").sentence("p2").token("1").xpos == "DT"


def test_applier_accepts_any_value_without_trusted_data(pred_path):
    from mcparser.corpus import CorpusStore

    only = CorpusStore()
    only.add_source("pred", pred_path, mode="rw")
    Applier(only, ROLES["xpos"]).annotation(Annotation(values="p2 1 ANYTHING 0.9"), batch("p2"))


def test_changing_a_key_marks_dependents_and_closes_flags(store):
    store.update_cells("pred", "p2", [
        {"token_id": "1", "column": "xpos", "value": "DT", "confidence": 0.97},
        {"token_id": "1", "column": "feats", "value": "Definite=Def", "confidence": 0.97},
        {"token_id": "2", "column": "xpos", "value": "NN", "confidence": 0.97},
    ], agent="rules")  # fmt: skip
    flag = store.flag_issue("pred", "p2", "upos", "should be PRON", token_id="1", agent="rules")
    other = store.flag_issue("pred", "p2", "upos", "check", token_id="2", agent="rules")
    Applier(store, ROLES["upos"]).annotation(Annotation(values="p2 1 PRON 0.9\np2 2 NOUN 0.9"), batch("p2"))
    sentence = store.source("pred").sentence("p2")
    assert read_confidence(sentence.token("1")) == {"upos": 0.9, "xpos": 0.0, "feats": 0.0}
    assert read_confidence(sentence.token("2"))["xpos"] == 0.97  # unchanged UPOS: dependents keep their scores
    flags = {f.id: f for f in store.list_flags("pred", status="all").flags}
    assert (flags[flag.id].status, flags[flag.id].note) == ("resolved", "upos changed it to PRON")
    assert flags[other.id].status == "open"
    Applier(store, ROLES["head"]).annotation(Annotation(values="p2 2 0 0.9\np2 1 2 0.9"), batch("p2"))  # no change
