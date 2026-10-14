# The agents

There is one agent per CoNLL-U element, plus the scorer. Each is a
[pydantic-ai](https://ai.pydantic.dev) agent whose only tools are the
[MCP server's](mcp.md) tools, connected in-process.

| Role | Writes | Typical evidence |
|---|---|---|
| `segmentation` | FORM, multiword tokens (`split_word`, `merge_words`, …) | `find_multiword`, `analyse_form` |
| `upos` | UPOS | `analyse_form`, `find_tokens`, `allowed_values` |
| `xpos` | XPOS | `allowed_values(column="xpos", upos=...)`, `analyse_form` |
| `lemma` | LEMMA | `analyse_form`, `find_tokens(suffix=...)` analogies, `find_tokens(root=...)` |
| `feats` | FEATS | `feature_inventory(upos=...)`, `analyse_form` |
| `head` | HEAD | similar treebank sentences, fixed expressions, `find_arcs`, `relation_stats` |
| `deprel` | DEPREL | `relation_stats` for the word's actual head, `find_arcs`, `allowed_values` |
| `deps` | DEPS (off by default) | `find_arcs`, the trusted data's enhanced graphs |
| `scorer` | confidences only | all of the above, plus relation counts for each current arc |

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

The instructions (see {mod}`mcparser.agents.prompts`) describe the team, the
evidence tools for that role, the confidence scale (0.95 = clear and well
attested, 0.8 = probable, 0.6 = plausible, below 0.5 = a guess), and the rule
that problems in other columns are flagged, not fixed.

## What an agent may do

{class}`~mcparser.agents.toolset.RoleToolset` wraps the MCP connection for
each role:

- only the role's tools are listed (the search tools for every role; the
  editing tools only for segmentation), and results come back as compact text
  lines instead of JSON;
- writes are allowed only to the prediction source and only to the role's own
  column; the scorer may only send confidences;
- every write is signed with the role's name, so the history and flags show
  who did what (the `agent` argument is hidden from the model);
- a refused call goes back to the model as a retry message explaining what to
  do instead (for example "call `flag_issue` with column='upos'").

## Answers

Every role except segmentation answers in one request with compact lines,
`SENTENCE TOKEN_ID VALUE CONFIDENCE`, e.g. `s1 3 NOUN 0.95`. The sentences of a
task are labelled `s1`, `s2`, ... in the prompt; a full `sent_id` is accepted
too. The scorer answers `SENTENCE TOKEN_ID COLUMN CONFIDENCE`, where `*` stands
for all the listed cells of a word (`s1 3 * 0.9`) or of the sentence
(`s1 * * 0.95`), and later lines override earlier ones. Answers also include any flags for other columns and the flags it closed. Code
checks the whole answer first: an unknown sentence or word, an invalid tag or,
for HEAD, a cycle or wrong number of roots is sent back to the model to fix;
only a valid answer is applied. For HEAD and DEPREL, code also asks once when
an answer gives a word two dependents of a relation the treebank almost never
repeats (in Coptic Scriptorium, 4 of 3,491 words have two `nsubj`; a noun
subject next to a pronoun subject is `dislocated`); sending the same answer
again confirms it. The reasons for every retry are kept in the run report.
The segmentation agent still edits through the
tools, since splitting and merging renumber the words.

## Models and reasoning

Roles use one of three tiers. The **strong** model (default Opus 5.5) handles
segmentation, lemmas, HEAD, DEPREL and DEPS. The **efficient** model (default
Sonnet 5.5) handles UPOS, FEATS and the scorer. The **light** model (default
Haiku 5.5) handles XPOS, which is close to a lookup once UPOS and FEATS are
known. Change the split with `--efficient-roles` / `--light-roles`, the models
with `--strong-model` / `--efficient-model` / `--light-model`, or give one role
its own model with `--model-for` or in the settings file. Any pydantic-ai model name
works (`anthropic:...`, `openai:...`, `google-gla:...`, `ollama:...`).

Reasoning ("thinking") is set per tier with `--strong-thinking` /
`--efficient-thinking` / `--light-thinking` (`off`, `on`, `minimal`, `low`, `medium`, `high`,
`xhigh`) or per role in the settings. More reasoning costs output tokens and has
not been measured here yet; HEAD and DEPREL are the likeliest to benefit. Left
unset, each model uses its own default.

## Using the agents from Python

```python
import anyio
from mcparser.agents.team import AgentTeam
from mcparser.corpus import CorpusStore
from mcparser.pipeline import Pipeline
from mcparser.settings import Settings

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
