# Troubleshooting

## The run stops with a message about an API key

Set the key for the provider of your models (see [Models and providers](providers.md#credentials)),
for example `export ANTHROPIC_API_KEY=...`. Keys are read from the environment only.

## "STOPPED: the cost limit was reached"

`--max-cost` (or `max_cost` in the settings) stops *starting* tasks once the estimated cost reaches
the limit; tasks already running finish, so the total can exceed it by a few tasks. Run the same
command again to continue: the file keeps its progress. Sentences not reached stay unannotated.

## Some tasks failed

A failing task is recorded in the report (and `<file>.runs.jsonl`) and the run continues. Run again
and the work that is still needed is retried. Frequent failures usually mean a model that does
not follow the answer format or call tools reliably; try a stronger model for that role
(`--model-for ROLE=MODEL`).

## Sentences have no HEAD, or UAS is much lower than expected

`evaluate` counts unparsed sentences as wrong. Use `--annotated-only` to score only parsed
sentences, `--predicted-only` if the prediction file has fewer sentences than the gold, and
`mcparser validate pred.conllu` to see structural problems. The refine step visits
the most confident sentences first, so with a tight `--max-cost` the sentences with no value
can be reached last; raise the limit or run again.

## "modified by another process" (ConflictError)

The prediction file changed on disk after it was loaded, for example because another run, Claude
Desktop or an editor wrote to it. Run only one writer per file at a time. In a long-running
session, call `reload` and repeat the edit.

## Claude Desktop does not list the tools

Run `mcparser mcp config ...` and paste the whole `mcpServers` entry into `claude_desktop_config.json`,
then restart Claude Desktop. The entry uses the Python of the environment that printed it, so print
it from the environment where MCParser is installed. Check the server starts with
`mcparser mcp serve ...` in a terminal (it waits for a client on stdio). Logs are in Claude Desktop's
MCP log files.

## `mcparser evaluate` disagrees with the official script

The evaluator is tested against the official `conll18_ud_eval.py` on random inputs, and they
agree when the surface tokens are the same. `--official PATH_TO_SCRIPT` runs the official script on the same
files. It rejects sentences with a missing HEAD, several roots or a cycle, so those are repaired for
that run only, and listed. `--strict` adds all features and full relation labels.

## The cost estimate is missing or low

A model with no known price prints no cost, and `--max-cost` cannot count it. Estimates use list
prices. See [Models and providers](providers.md#cost-estimates-and-max-cost).

## Reporting a bug

Open an issue with `mcparser --version`, the command line, the settings (`mcparser settings --config ...`),
and the relevant lines of `<file>.runs.jsonl` or the error message. Do not paste API keys or private
text.
