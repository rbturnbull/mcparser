import json

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

from llmparser.agents.prompts import instructions, task_prompt
from llmparser.agents.roles import COLUMN_ROLES, READ_TOOLS, ROLES, SEGMENTATION_TOOLS
from llmparser.agents.team import AgentTeam
from llmparser.agents.toolset import RoleToolset
from llmparser.corpus.models import CellRef, Flag
from llmparser.settings import RoleSettings, Settings
from scripted import scripted

# ---------------------------------------------------------------- roles and prompts


def test_roles():
    assert ROLES["upos"].column == "upos"
    assert SEGMENTATION_TOOLS <= ROLES["segmentation"].tools
    assert not SEGMENTATION_TOOLS & ROLES["lemma"].tools
    assert ROLES["scorer"].tools == READ_TOOLS | {"flag_issue", "resolve_flag", "update_cells"}
    assert ROLES["scorer"].is_scorer and not ROLES["head"].is_scorer
    assert COLUMN_ROLES["form"] == "segmentation"
    assert "scorer" not in COLUMN_ROLES.values()


def test_instructions():
    text = instructions(ROLES["head"], "pred", ["train", "silver"], "Mind the particles.")
    assert 'The source "pred"' in text
    assert "(train, silver)" in text
    assert "only change HEAD" in text
    assert text.endswith("Mind the particles.")
    assert "(none)" in instructions(ROLES["scorer"], "pred", [])
    assert "only change the confidence of cells" in instructions(ROLES["scorer"], "pred", [])


def test_task_prompts():
    cells = [CellRef(sent_id="s", token_id="2", form="al", column="upos", value="_", confidence=0.0)]
    flags = [
        Flag(id=3, sent_id="s", token_id="2", column="upos", reason="looks like ADP", suggestion="ADP", raised_by="lemma", created="t"),
        Flag(id=4, sent_id="s", column="upos", reason="check all", raised_by="scorer", created="t"),
    ]
    text = task_prompt(ROLES["upos"], "pred", "s", cells, flags)
    assert text.splitlines() == [
        'Sentence "s" in source "pred".',
        "These UPOS cells need annotation or review (low confidence):",
        "- word 2 (al): UPOS = '_', confidence 0.00",
        "Open flags for UPOS:",
        "- flag 3 (word 2, from lemma): looks like ADP Suggested: 'ADP'.",
        "- flag 4 (whole sentence, from scorer): check all",
        "You may also correct any other UPOS value in this sentence.",
    ]
    assert "Score these cells" in task_prompt(ROLES["scorer"], "pred", "s", cells, [])
    assert "Open flags" not in task_prompt(ROLES["upos"], "pred", "s", cells, [])
    assert "need annotation" not in task_prompt(ROLES["upos"], "pred", "s", [], flags)


# ---------------------------------------------------------------- toolset permissions


def toolset(role):
    return RoleToolset(wrapped=None, role=ROLES[role], target="pred")


@pytest.mark.parametrize(
    ("role", "name", "args", "message"),
    [
        ("lemma", "split_word", {"source": "pred"}, "not available"),
        ("lemma", "set_cell", {"source": "train", "column": "lemma"}, "only edit the source 'pred'"),
        ("lemma", "set_cell", {"source": "pred", "column": "upos"}, "only change lemma"),
        ("lemma", "update_cells", {"source": "pred", "updates": [{"column": "pos", "value": "X"}]}, "flag_issue with column='upos'"),
        ("scorer", "update_cells", {"source": "pred", "updates": [{"column": "upos", "value": "X"}]}, "only send confidences"),
        ("scorer", "update_cells", {"source": "pred", "updates": [{"column": "misc", "confidence": 0.5}]}, "only change form"),
    ],
)
def test_toolset_refuses(role, name, args, message):
    with pytest.raises(ModelRetry, match=message):
        toolset(role).check(name, args)


def test_toolset_normalises_updates():
    normalise = RoleToolset.normalise
    assert normalise("set_cell", {"x": 1}) == {"x": 1}
    assert normalise("update_cells", {"updates": '[{"column": "lemma"}]'})["updates"] == [{"column": "lemma"}]
    for bad in ("not json", '["x"]', None, {"column": "lemma"}):
        with pytest.raises(ModelRetry, match="must be a list"):
            normalise("update_cells", {"updates": bad})


def test_toolset_allows():
    toolset("lemma").check("set_cell", {"source": "pred", "column": "LEMMA"})
    toolset("lemma").check("update_cells", {"source": "pred", "updates": None})
    toolset("scorer").check("update_cells", {"source": "pred", "updates": [{"column": "head", "confidence": 0.4}]})
    toolset("segmentation").check("split_word", {"source": "pred"})
    toolset("head").check("resolve_flag", {"source": "train"})
    toolset("head").check("find_arcs", {})


# ---------------------------------------------------------------- team


@pytest.mark.anyio
async def test_team_runs_role_agents_through_mcp(store):
    seen = {}

    def decide(prompt):
        return [
            ("update_cells", {"source": "pred", "sent_id": "p1", "updates": [{"token_id": "2", "column": "upos", "value": "ADP", "confidence": 0.7}]}),
            ("set_cell", {"source": "pred", "sent_id": "p1", "token_id": "2", "column": "lemma", "value": "a"}),
            ("flag_issue", {"source": "pred", "sent_id": "p1", "column": "lemma", "reason": "al = a + el"}),
        ]

    model = scripted(decide, summary="tagged al")
    original = model.function

    def spy(messages, info):
        seen.setdefault("tools", {tool.name: tool for tool in info.function_tools})
        return original(messages, info)

    model.function = spy
    team = AgentTeam(store, "pred", Settings(), models={"upos": model})
    async with team:
        report = await team.run("upos", "go")
    assert report.summary == "tagged al"
    assert team.requests == 2 and team.input_tokens > 0 and team.output_tokens > 0
    tools = seen["tools"]
    assert "split_word" not in tools and "save" not in tools and "update_cells" in tools
    assert "agent" not in json.dumps(tools["update_cells"].parameters_json_schema["properties"])
    history = store.get_history("pred")
    assert [(edit.action, edit.agent, edit.column) for edit in history] == [("flag", "upos", "lemma"), ("set", "upos", "upos")]
    assert any("only change upos" in text for text in model.returns)


def test_team_models_and_agents(store):
    settings = Settings(model="global", roles={"lemma": RoleSettings(model="lemma-model", instructions="Extra.")})
    team = AgentTeam(store, "pred", settings, models={"head": TestModel()})
    assert team.trusted == ["train"]
    assert isinstance(team.model("head"), TestModel)
    assert team.model("lemma") == "lemma-model"
    assert team.model("upos") == "global"
    assert AgentTeam(store, "pred", settings, models={"*": "star"}).model("upos") == "star"
    agent = team.agent("lemma")
    assert team.agent("lemma") is agent
    assert agent.name == "lemma"
