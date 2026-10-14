import json

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

from llmparser.agents.outputs import Annotation, Applier, Batch, FlagRequest, FlagResolution, Scores, parse_scores, parse_values
from llmparser.agents.prompts import TaskItem, instructions, sentence_block, task_prompt
from llmparser.agents.roles import COLUMN_ROLES, READ_TOOLS, ROLES, SEGMENTATION_TOOLS
from llmparser.agents.team import AgentTeam, TaskReport
from llmparser.agents.toolset import RoleToolset
from llmparser.corpus.confidence import read_confidence
from llmparser.corpus.models import CellRef, Flag
from llmparser.settings import RoleSettings, Settings
from scripted import scripted, values

# ---------------------------------------------------------------- roles and prompts


def test_roles():
    assert ROLES["upos"].column == "upos"
    assert SEGMENTATION_TOOLS <= ROLES["segmentation"].tools and "update_cells" in ROLES["segmentation"].tools
    assert ROLES["lemma"].tools == READ_TOOLS == ROLES["scorer"].tools
    assert "relation_stats" in READ_TOOLS
    assert ROLES["segmentation"].edits_with_tools and not ROLES["scorer"].edits_with_tools
    assert ROLES["scorer"].is_scorer and not ROLES["head"].is_scorer
    assert COLUMN_ROLES["form"] == "segmentation"
    assert "scorer" not in COLUMN_ROLES.values()


def test_instructions():
    text = instructions(ROLES["head"], "pred", ["train", "silver"], "Mind the particles.")
    assert "conventions of the trusted treebank\n(train, silver)" in text
    assert '"12841 3 4 0.95"' in text and "any other HEAD value" in text
    assert text.endswith("Mind the particles.")
    scorer = instructions(ROLES["scorer"], "pred", [])
    assert "treebank\n(none)" in scorer and '"SENT_ID TOKEN_ID COLUMN CONFIDENCE"' in scorer
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
        "## Sentence s",
        "# sent_id = s",
        "1\tVino\tvenir\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No",
        "UPOS cells to annotate or review (low confidence):",
        "- word 2 (al): UPOS = '_', confidence 0.00",
        "Open flags for UPOS:",
        "- flag 3 (word 2, from lemma): looks like ADP Suggested: 'ADP'.",
        "- flag 4 (whole sentence, from scorer): check all",
        "",
        "## Sentence t",
        "Open flags for UPOS:",
        "- flag 4 (whole sentence, from scorer): check all",
        "",
        "EVIDENCE",
    ]
    scorer = task_prompt(ROLES["scorer"], "pred", [TaskItem("s", cells, flags)])
    assert "Score these cells:" in scorer and "Open flags" not in scorer
    assert sentence_block("1\ta\t_\t_\t_\t_\t0\troot\t_\tConf=upos:0.10") == "1\ta\t_\t_\t_\t_\t0\troot\t_\t_"


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
        "line 2: expected SENT_ID TOKEN_ID VALUE CONFIDENCE, got 'p1 3 x'",
        "line 3: sentence 'zz' is not in this task",
        "line 4: confidence 'high' is not a number",
    ]


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
        report = await team.run("upos", "## Sentence p1\n- word 2 (al): UPOS = '_', confidence 0.00", batch("p1"))
    assert report.summary == "tagged al"
    assert attempts == [0, 1]
    assert "not a universal POS tag" in model.retries[0]
    assert any(text.startswith("2 matches. By lemma upos xpos feats: el DET") for text in model.returns)
    assert store.source("pred").sentence("p1").token("2").upos == "ADP"
    assert [(e.agent, e.column) for e in store.get_history("pred")] == [("upos", "upos")]
    usage = team.total()
    assert usage.requests == 3 and usage.input_tokens > 0 and usage.output_tokens > 0 and usage.cost is None


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
