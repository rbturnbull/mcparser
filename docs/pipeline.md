# The pipeline

```text
                         ┌────────────────────────────── round n ──────────────────────────────┐
prediction ──▶ rules ──▶ │ score: scorer rates cells a model changed since it last scored them  │──▶ stop? ──▶ (final scoring)
(+ baseline)             │ refine: each role in turn, batches of sentences (most confident      │      │
                         │ first), cells below the threshold + open flags; rules after each role│◀─────┘ next round
                         └──────────────────────────────────────────────────────────────────────┘
```

## Round 0: the baseline

`mcparser init` creates the prediction file. When trusted sources are given,
it also applies a **frequency baseline** that needs no LLM:

1. **Segmentation.** A surface token that the trusted data splits into a
   multiword token more often than it leaves it whole is split the same way,
   e.g. Arabic *بشيه* → *ب* + *شيه*.
2. **Morphology.** Each word gets the most frequent (LEMMA, UPOS, XPOS, FEATS)
   analysis of the same form. If there's no exact match, the lookup ignores
   case and diacritics.
3. **Confidence** reflects the evidence: the share of examples agreeing,
   discounted when there are fewer than three, and capped at 0.9.

The baseline only fills empty cells, so running it again changes nothing.
HEAD and DEPREL are left for the agents. You can also skip `init` and give
`run` any CoNLL-U file, for example the output of another parser.

## Each round

**Score.** The scorer agent rates every cell of each sentence that has never
been scored, or that changed after it was last scored. It sends only
confidences, never values, and flags cells it's fairly sure are wrong.

**Refine.** Sentences are visited **from most to least confident**, so the
sentences that settle first can serve as evidence (with `min_confidence`) for
the harder ones. In each sentence, the role agents run in this order:

```text
segmentation → upos → lemma → syntax → feats → xpos → (deps)
                            (or head → deprel with --separate-syntax)
```

Later columns build on earlier ones. Syntax comes before FEATS because case
and definiteness depend on a word's function, and XPOS comes last because many
treebanks encode UPOS and FEATS in it. When a model changes UPOS, the word's
XPOS and FEATS are marked for review; FEATS → XPOS and HEAD → DEPREL likewise. A role runs on a sentence only if it has
cells below the threshold (default 0.8) or open flags for its column.

**Stop.** The loop ends when no role had anything to do, when a round made no
edits, or when the round limit (default 3) is reached. With `--final-score`
(setting `final_scoring`), a last scoring pass re-scores whatever the last round
changed. It changes no annotation and cost 14 of 45 tasks in a Coptic run, so it
is off by default; the next `run` scores those cells anyway.

## Settings

`mcparser settings` prints the defaults as YAML. Save them, edit them, and
pass the file with `--config`:

```yaml
strong_model: anthropic:claude-opus-5-5       # segmentation, lemma, syntax, deps
efficient_model: anthropic:claude-sonnet-5-5  # efficient_roles
light_model: anthropic:claude-haiku-5-5       # light_roles
efficient_roles: [upos, feats, scorer]
light_roles: [xpos]
strong_thinking: low          # reasoning effort: false, true, minimal, low, medium, high, xhigh
threshold: 0.8                # refine cells below this confidence
rounds: 3                     # maximum rounds
cheap_first: false            # round 1 one tier down (Opus -> Sonnet, Sonnet -> Haiku); later rounds use the role's own tier
batch_sentences: 8            # sentences per agent task
batch_cells: 120              # cells per agent task
concurrency: 1                # tasks run at the same time
request_limit: 60             # model requests per agent task
cache: true                   # prompt caching (Anthropic, Bedrock, OpenRouter, OpenAI GPT-5.6+; others cache implicitly)
batch: false                  # Anthropic Message Batches: half the price, hours of latency (see below)
batch_cache: 1h               # prompt cache lifetime in batch mode: 5m, 1h or off
batch_poll_seconds: 30        # seconds between checks of a submitted batch (backs off to 5 minutes)
batch_max_wait_seconds: 600   # submit queued requests after this long even if other tasks are still preparing theirs
max_cost: null                # stop starting tasks at this estimated cost (US dollars); run again to continue
conventions: true             # tell the FEATS agent where the treebank writes combined values (Gender=Masc,Neut)
rules:
  min_count: 5                # examples needed before code settles a cell
  min_share: 0.98             # share of them that must agree
  use_predictions: true       # confident predictions count as evidence for unseen forms
roles:
  head:
    thinking: medium                     # per-role reasoning effort
  deps:
    enabled: true                        # enhanced dependencies are off by default
```

Command-line overrides: `--strong-model`, `--efficient-model`, `--light-model`,
`--efficient-roles`, `--light-roles`, `--strong-thinking`,
`--efficient-thinking`, `--light-thinking`, `--model` (one model for every
role), `--model-for ROLE=MODEL`, `--roles`, `--rounds`, `--threshold`,
`--batch-sentences`, `--concurrency`, `--max-cost`, `--batch/--no-batch`, `--batch-poll-seconds`, `--cheap-first`, `--sentence` and `--max-sentences`.
`--cheap-first` runs round 1 on the next cheaper tier and sends only the cells that stay below the threshold to the role's own
tier in later rounds (it needs at least two rounds to pay off; the scorer, `--model` and per-role models are not downgraded).
`--max-cost 50` stops starting agent tasks once the estimate reaches $50 (tasks already running finish, so the total can
exceed it by a few tasks); the sentences not reached stay unannotated, and `run` on the same file continues.
`mcparser score` runs only the rules and the scorer.

## Cost

Each run ends with token counts and an estimated price per model (list prices
in {mod}`mcparser.pricing`). What keeps the cost down, roughly in order of
effect:

- **Rules first.** Code settles every cell the treebank makes near-certain (see
  {mod}`mcparser.rules`): on Latin PROIEL about three quarters of the LEMMA,
  UPOS, XPOS and FEATS cells, with no errors in testing, and a fifth of the
  relation labels once heads are known. Those cells never reach a model. With
  `use_predictions`, forms the treebank lacks are settled from confident
  predictions elsewhere in the file, so a long text gets cheaper as it goes.
- **Batches.** Each task covers up to `batch_sentences` sentences, and evidence
  for a repeated form or arc is shown once.
- **One request per task.** Agents (except segmentation) answer with compact
  lines (`s1 3 NOUN 0.95`) that code checks and applies; an invalid answer
  (unknown tag, broken tree) is sent back to be fixed.
- **Short output.** Output tokens cost five times as much as input. Answers
  name sentences `s1`, `s2`, ... instead of repeating their `sent_id` (about 20
  tokens each in Coptic Scriptorium), and the scorer starts with `s1 * * 0.95`
  and lists only the exceptions.
- **Short input.** Most of the cost is the task prompt, which is written to the
  prompt cache once. The cells to work on are listed as word ranges
  (`1..36 at 0.00, 40 at 0.60`), since their values are in the sentence table;
  empty columns at the end of a CoNLL-U line are left out; each form's
  analyses appear once per task, with only the columns the role needs (UPOS:
  lemma, UPOS, XPOS; LEMMA: lemma, UPOS).
- **Evidence in the prompt**: the form's analyses; for an ambiguous form, the
  readings next to the neighbouring words and tags (after the previous word or
  UPOS, before the next word or UPOS), shown only where they change the
  picture (Coptic `ⲛ` is the plural article after a preposition, 230 to 2);
  word-ending analogies for unseen forms; relation counts for each word's
  actual head.
- **Unannotated text** (`--corpus`): for forms the treebank has seen fewer than
  three times, the evidence adds how often the form occurs in the text, its
  neighbours and the longer words that contain it; for tokens the treebank lacks,
  the segmentation agent sees how often the token and its remainder after cutting
  one or two letters occur (a frequent remainder next to a rare token suggests a
  clitic). It costs a few tokens per rare form and nothing when there is no
  corpus.
- **Conventions** (`conventions`): the FEATS agent is told where the treebank writes a combined value
  (`Gender with Gen Sing: Masc 75%, Masc,Neut 25%`), computed from the trusted data for the UPOS, case and
  number combinations where at least 10% of 20+ examples have one.
- **Three model tiers**: Haiku for XPOS, Sonnet for UPOS, FEATS and scoring,
  Opus for segmentation, lemmas and syntax.
- **Cross-checks** flag (never change) a UPOS that the treebank almost never
  uses for that lemma in that relation, once the syntax is known, so the next
  round can fix it (PROIEL's *omnis* is DET as `det`, PRON as a subject).
  This, and the scorer's low scores, only pay off with at least two rounds.
- **Voice check**: after FEATS, a subject or auxiliary relation (`nsubj`, `csubj`, `aux`) that ignores the head's voice
  (plain `nsubj` on a `Voice=Pass` verb, `nsubj:pass` on an active one) is
  flagged for the syntax agent when the treebank separates the two (≥ 90% of
  the cases, at least 5 examples, and the word's own label is rare for that voice).
- **The scorer only judges what a model or person wrote**, never cells set by
  code, unannotated cells, or XPOS/FEATS of untagged words.
- **Compact tool results**, **prompt caching**, and columns the treebank never
  uses settled without an agent.

On 10 Latin sentences the first version cost $2.43; the current one costs
about $0.08 per sentence of similar length.

Every run's report (rounds, tasks, tokens and cost per model, per role and per
task) is appended to
`<file>.runs.jsonl` next to the prediction file. Before running a whole file,
try `--max-sentences` on a few sentences and look at the estimate. `--report run.json` saves the numbers.

With `concurrency` above 1, several sentences are worked on at once. That is
faster, but a sentence can then no longer rely on the more confident sentences
having been finished first.

## Message Batches (`--batch`)

For a long run that is not urgent (thousands of sentences on a server left for a day),
`--batch` sends the model requests through Anthropic's
[Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing),
which charges **50% of the live price** for input and output. The default stays the live API.

```bash
export ANTHROPIC_API_KEY=...
mcparser run pred.conllu -s train=train.conllu --batch --concurrency 200 --rounds 3 --max-cost 100
```

**How it works.** A batch is one-shot (independent requests in, answers later), but an agent task
is a conversation of several requests, and rounds depend on earlier rounds. So MCParser batches
the *requests*, not the run. Every agent, tool, retry and validator behaves as before; only the
Anthropic client is replaced by one that queues each request. When every task in flight is waiting
for its next request, the queue goes out as one batch; MCParser polls until the batch has ended
(30 seconds at first, backing off to 5 minutes), then hands each answer back to the task that asked.
Each wave of simultaneous requests is one batch, so the time is about *(dependent steps) x (batch
latency)*, not the number of tasks. Set a high `--concurrency` (hundreds): with the default of 1 every
request waits for its own batch.

**What to expect.**

- Most batches finish within an hour, but a batch can take up to 24 hours, after which its
  unfinished requests expire. A request that errors, expires or is cancelled fails only its own
  task: it is recorded as a failed task and the next `run` retries the work.
- Anthropic models only. `--batch` with another provider stops with an error before any request is made.
- Prices in the run summary are batch prices (marked in the report) and `--max-cost` works on that
  estimate. It only stops *starting* tasks, and tasks waiting for a batch count as running, so the
  total can exceed the limit by what is in flight. Batches can also slightly exceed a workspace spend limit.
- **Prompt caching in a batch** is best effort, and a batch can take longer than the 5-minute cache lifetime.
  `batch_cache` (`--batch-cache`) chooses: `1h` (the default) keeps entries for an hour, but a one-hour write costs 2x the
  input price against 1.25x for 5 minutes; `5m` is cheaper to write but entries may expire between waves; `off`
  sends no cache markers. Most of an agent task's prompt is unique to it, so with few cache reads the saving is below 50%
  and the cheaper write can win. See the measurement below.
- Features that need an immediate answer are unavailable (no streaming, no fast mode).

**Observability and recovery.** Every batch is logged as it is submitted and as it ends
(`batch msgbatch_...: submitted 120 requests`), and the run report lists each batch with its id, size,
seconds to finish and how its requests ended (`<file>.runs.jsonl`, `--report`). Edits are saved after
each task, so if the process dies, in-flight conversations are lost and running the same command again
continues from the file. Stopping with Ctrl-C cancels the batches that are running, so you are not charged for
requests that have not started. Batches that finished while no process was waiting can still be seen in the
Anthropic Console, but their results are not collected by MCParser.

**On a server.** Run under `tmux` or `nohup` with `ANTHROPIC_API_KEY` in the environment (never in a file):

```bash
nohup mcparser run pred.conllu -s train=train.conllu --batch --concurrency 200 --rounds 3 \
    --cheap-first --max-cost 200 > run.log 2>&1 &
```

The batching client is in {mod}`mcparser.agents.batching` and does not depend on MCParser, so other
pydantic-ai projects can reuse it.
