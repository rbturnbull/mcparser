# The agents

There is one agent per CoNLL-U element, plus the scorer. Each is a
[pydantic-ai](https://ai.pydantic.dev) agent whose only tools are the
[MCP server's](mcp.md) tools, connected in-process.

| Role | Writes | Typical evidence |
|---|---|---|
| `segmentation` | FORM, multiword tokens (`split_word`, `merge_words`, …) | `find_multiword`, `analyse_form` |
| `upos` | UPOS | `analyse_form`, `find_tokens`, `allowed_values` |
| `xpos` | XPOS | `allowed_values(column="xpos", upos=...)`, `analyse_form` |
| `lemma` | LEMMA | `analyse_form`, `find_tokens(lemma=...)`, `find_tokens(root=...)` |
| `feats` | FEATS | `feature_inventory(upos=...)`, `analyse_form` |
| `head` | HEAD | `find_arcs`, `search_sentences`, `validate` |
| `deprel` | DEPREL | `find_arcs(dependent_upos=..., head_upos=...)`, `allowed_values` |
| `deps` | DEPS (off by default) | `find_arcs`, the trusted data's enhanced graphs |
| `scorer` | confidences only | all of the above |

## What an agent sees

Every agent can read everything: the whole sentence, its neighbours
(`get_context`), any other sentence in the prediction file, and the trusted
data. Its task prompt lists the cells it should work on and the open flags
for its column:

```text
Sentence "14.15" in source "pred".
These UPOS cells need annotation or review (low confidence):
- word 1 (NIM.LÀL): UPOS = '_', confidence 0.02
...
Open flags for UPOS:
- flag 3 (whole sentence, from scorer): Empty UPOS everywhere. Suggested: 'NOUN NUM NOUN NUM NOUN NOUN VERB'.
You may also correct any other UPOS value in this sentence.
```

The instructions (see {mod}`llmparser.agents.prompts`) describe the team, the
evidence tools for that role, the confidence scale (0.95 = clear and well
attested, 0.8 = probable, 0.6 = plausible, below 0.5 = a guess), and the rule
that problems in other columns are flagged, not fixed.

## What an agent may do

{class}`~llmparser.agents.toolset.RoleToolset` wraps the MCP connection for
each role:

- only the role's tools are listed (only the segmentation agent sees
  `split_word`; nobody sees `save` or `reload`);
- writes are allowed only to the prediction source and only to the role's own
  column; the scorer may only send confidences;
- every write is signed with the role's name, so the history and flags show
  who did what (the `agent` argument is hidden from the model);
- a refused call goes back to the model as a retry message explaining what to
  do instead (for example "call `flag_issue` with column='upos'").

## Models

Every role uses the global `model` unless the settings or `--model-for` give
it its own. Any pydantic-ai model name works (`anthropic:...`, `openai:...`,
`google-gla:...`, a local `ollama:...`). Cheaper models can suit the simpler
roles; HEAD and DEPREL benefit most from a strong model.

## Using the agents from Python

```python
import anyio
from llmparser.agents.team import AgentTeam
from llmparser.corpus import CorpusStore
from llmparser.pipeline import Pipeline
from llmparser.settings import Settings

store = CorpusStore()
store.add_source("train", "train.conllu")
store.add_source("pred", "pred.conllu", mode="rw")
settings = Settings(rounds=2)
team = AgentTeam(store, "pred", settings)
report = anyio.run(Pipeline(store, "pred", team, settings).run)
```

`AgentTeam(..., models={"*": some_model})` substitutes models. The test suite
uses this to run the whole pipeline with scripted
{class}`~pydantic_ai.models.function.FunctionModel` agents.
