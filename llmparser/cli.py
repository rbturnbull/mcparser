"""Command-line interface."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import anyio
import typer
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from .agents.roles import ROLES
from .agents.team import AgentTeam
from .baseline import run_baseline
from .corpus import CorpusError, CorpusStore, SourceMode, format_conllu, parse_conllu
from .evaluate import evaluate as evaluate_predictions
from .mcp.config import SourceSpec, claude_desktop_config, load_store
from .mcp.server import build_server
from .pipeline import Pipeline, RunReport
from .settings import Settings
from .textinput import build_sentences, read_input

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

TARGET = "pred"

app = typer.Typer(help="LLM agents for dependency parsing with CoNLL-U files.", no_args_is_help=True)
mcp_app = typer.Typer(help="The CoNLL-U MCP server.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


class Transport(StrEnum):
    STDIO = "stdio"
    HTTP = "streamable-http"


SourceOption = Annotated[
    list[str],
    typer.Option(
        "--source",
        "-s",
        help="[NAME=]PATH[:ro|rw[:CONFIDENCE]]. Repeat for several sources. Default mode is ro.",
    ),
]
RootKeyOption = Annotated[str, typer.Option(help="MISC attribute holding the word's root.")]
NameOption = Annotated[str, typer.Option(help="Server name shown to MCP clients.")]


def _specs(sources: list[str]) -> list[SourceSpec]:
    try:
        return [SourceSpec.parse(spec) for spec in sources]
    except CorpusError as error:
        raise typer.BadParameter(str(error), param_hint="--source") from error


def _load(specs: list[SourceSpec], root_key: str) -> CorpusStore:
    try:
        return load_store(specs, root_key=root_key)
    except CorpusError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@mcp_app.command("serve")
def serve(
    source: SourceOption,
    transport: Annotated[Transport, typer.Option(help="MCP transport.")] = Transport.STDIO,
    host: Annotated[str, typer.Option(help="Host for streamable-http.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for streamable-http.")] = 8000,
    read_only: Annotated[bool, typer.Option("--read-only", help="Register only the read-only tools.")] = False,
    root_key: RootKeyOption = "Root",
    name: NameOption = "conllu",
) -> None:
    """Serve CoNLL-U sources over MCP (stdio for Claude Desktop, or streamable HTTP)."""
    store = _load(_specs(source), root_key)
    server = build_server(store, name=name, allow_edits=not read_only)
    if transport is Transport.STDIO:
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)


@mcp_app.command("config")
def config(
    source: SourceOption,
    read_only: Annotated[bool, typer.Option("--read-only", help="Register only the read-only tools.")] = False,
    root_key: RootKeyOption = "Root",
    name: NameOption = "conllu",
) -> None:
    """Print the JSON to add to Claude Desktop's claude_desktop_config.json."""
    specs = _specs(source)
    for spec in specs:
        if not spec.path.is_file():
            typer.echo(f"Warning: {spec.path} does not exist.", err=True)
    typer.echo(json.dumps(claude_desktop_config(specs, name=name, read_only=read_only, root_key=root_key), indent=2))


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="CoNLL-U file to validate.")],
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="Trusted sources whose tagsets flag unseen values."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum issues to print.")] = 100,
) -> None:
    """Validate a CoNLL-U file. Exits with status 1 if there are errors."""
    specs = _specs(source or [])
    target = SourceSpec(name="__target__", path=path.expanduser().resolve())
    store = _load([*specs, target], "Root")
    report = store.validate(target.name, limit=limit)
    for item in report.issues:
        where = f"{item.sent_id}:{item.issue.token_id}" if item.issue.token_id else item.sent_id
        typer.echo(f"{item.issue.severity.upper():7} {where:20} {item.issue.code:18} {item.issue.message}")
    typer.echo(f"{report.errors} errors, {report.warnings} warnings.")
    if report.errors:
        raise typer.Exit(1)


# ---------------------------------------------------------------- annotation

TrustedOption = Annotated[
    list[str] | None,
    typer.Option("--source", "-s", help="Trusted example sources, [NAME=]PATH[:ro[:CONFIDENCE]]. Repeatable."),
]
ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Settings YAML (see `llmparser settings`).")]


def _prediction_store(prediction: Path, sources: list[str] | None) -> CorpusStore:
    specs = _specs(sources or [])
    target = SourceSpec(name=TARGET, path=prediction.expanduser().resolve(), mode=SourceMode.READ_WRITE)
    return _load([*specs, target], "Root")


def _settings(
    config: Path | None,
    model: str | None,
    model_for: list[str] | None,
    roles: str | None,
    **overrides: object,
) -> Settings:
    role_models = {}
    for item in model_for or []:
        role, sep, role_model = item.partition("=")
        if not sep or role not in ROLES:
            raise typer.BadParameter(f"Expected ROLE=MODEL with ROLE one of {', '.join(ROLES)}.", param_hint="--model-for")
        role_models[role] = role_model
    only = [role.strip() for role in roles.split(",")] if roles else None
    if only and not set(only) <= set(ROLES):
        raise typer.BadParameter(f"Roles must be among {', '.join(ROLES)}.", param_hint="--roles")
    try:
        base = Settings.from_file(config) if config else Settings()
        return base.with_overrides(model=model, role_models=role_models, only_roles=only, **overrides)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error


def make_team(store: CorpusStore, target: str, settings: Settings) -> AgentTeam:
    """Build the agent team (replaced in tests by one with scripted models)."""
    return AgentTeam(store, target, settings)


def _print_report(report: RunReport) -> None:
    for item in report.rounds:
        typer.echo(
            f"Round {item.round}: scored {item.scored} sentences, {item.tasks} agent tasks, "
            f"{item.edits} edits, {item.open_flags} open flags, mean confidence {item.confidence:.3f}"
        )
    failures = sum(task.error is not None for task in report.tasks)
    typer.echo(
        f"Final mean confidence {report.confidence:.3f} ({report.final_scored} sentences re-scored). "
        f"{len(report.tasks)} tasks ({failures} failed), {report.requests} model requests, "
        f"{report.input_tokens} input ({report.cache_read_tokens} from cache) / {report.output_tokens} output tokens."
    )


def _annotate(
    prediction: Path,
    source: list[str] | None,
    settings: Settings,
    sentence: list[str] | None,
    score_only: bool,
    report_path: Path | None,
) -> None:
    store = _prediction_store(prediction, source)
    if not any(info.mode == "ro" for info in store.list_sources()):
        typer.echo("Warning: no trusted --source given; agents will have no examples to learn from.", err=True)
    try:
        pipeline = Pipeline(
            store, TARGET, make_team(store, TARGET, settings), settings,
            sent_ids=sentence or None, progress=lambda message: typer.echo(message, err=True),
        )  # fmt: skip
        report = anyio.run(pipeline.run, None, score_only)
    except (CorpusError, UserError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    _print_report(report)
    if report_path:
        report_path.write_text(report.model_dump_json(indent=1), encoding="utf-8")


ModelOption = Annotated[str | None, typer.Option(help="Model for every role, e.g. anthropic:claude-opus-5-5.")]
ModelForOption = Annotated[
    list[str] | None, typer.Option("--model-for", help="ROLE=MODEL override, e.g. lemma=anthropic:claude-sonnet-5-5.")
]
SentenceOption = Annotated[list[str] | None, typer.Option("--sentence", help="Only work on this sent_id. Repeatable.")]
ReportOption = Annotated[Path | None, typer.Option("--report", help="Write the run report as JSON.")]
PredictionArgument = Annotated[Path, typer.Argument(help="Prediction CoNLL-U file (edited in place).")]


@app.command()
def run(
    prediction: PredictionArgument,
    source: TrustedOption = None,
    config: ConfigOption = None,
    model: ModelOption = None,
    model_for: ModelForOption = None,
    roles: Annotated[str | None, typer.Option(help=f"Comma-separated roles to run: {', '.join(r for r in ROLES if r != 'scorer')}.")] = None,
    rounds: Annotated[int | None, typer.Option(min=1, help="Maximum rounds.")] = None,
    threshold: Annotated[float | None, typer.Option(min=0, max=1, help="Refine cells below this confidence.")] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Sentences processed at once.")] = None,
    sentence: SentenceOption = None,
    report: ReportOption = None,
) -> None:
    """Score and refine a prediction file with the agents, over several rounds."""
    settings = _settings(config, model, model_for, roles, rounds=rounds, threshold=threshold, concurrency=concurrency)
    _annotate(prediction, source, settings, sentence, False, report)


@app.command()
def score(
    prediction: PredictionArgument,
    source: TrustedOption = None,
    config: ConfigOption = None,
    model: ModelOption = None,
    model_for: ModelForOption = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Sentences processed at once.")] = None,
    sentence: SentenceOption = None,
    report: ReportOption = None,
) -> None:
    """Only run the scorer: give every changed or unscored cell a confidence."""
    settings = _settings(config, model, model_for, None, concurrency=concurrency)
    _annotate(prediction, source, settings, sentence, True, report)


@app.command()
def init(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT", help="Text (one sentence per line) or YAML with 'text' fields.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="CoNLL-U file to create.")],
    source: TrustedOption = None,
    baseline: Annotated[bool, typer.Option(help="Fill in what the trusted sources make obvious (no LLM).")] = True,
    force: Annotated[bool, typer.Option("--force", help="Overwrite OUTPUT and its review/history files.")] = False,
) -> None:
    """Create a prediction file from raw text: tokenise, then apply the frequency baseline."""
    if output.exists() and not force:
        typer.echo(f"Error: {output} exists; use --force to overwrite it.", err=True)
        raise typer.Exit(1)
    try:
        sentences = build_sentences(read_input(input_path))
    except (CorpusError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    output.write_text(format_conllu(sentences), encoding="utf-8")
    for sidecar in (".review.json", ".history.jsonl"):
        output.with_name(output.name + sidecar).unlink(missing_ok=True)
    typer.echo(f"Wrote {len(sentences)} sentences to {output}.")
    if baseline and source:
        result = run_baseline(_prediction_store(output, source), TARGET)
        typer.echo(
            f"Baseline: {result.splits} multiword splits, {result.analysed} words analysed, "
            f"{result.unknown} words not found in the trusted sources."
        )


@app.command()
def evaluate(
    prediction: Annotated[Path, typer.Argument(help="Predicted CoNLL-U.")],
    gold: Annotated[Path, typer.Argument(help="Gold CoNLL-U.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """Compare predictions with gold annotation (precision, recall and F1 per metric)."""
    try:
        result = evaluate_predictions(
            parse_conllu(prediction.read_text(encoding="utf-8")), parse_conllu(gold.read_text(encoding="utf-8"))
        )
    except (CorpusError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(result.model_dump_json(indent=1))
        return
    typer.echo(f"{result.sentences} sentences matched.")
    if result.missing:
        typer.echo(f"Missing from the predictions: {', '.join(result.missing)}")
    if result.extra:
        typer.echo(f"Not in the gold file: {', '.join(result.extra)}")
    typer.echo(result.table())


@app.command("settings")
def show_settings(config: ConfigOption = None) -> None:
    """Print the settings (defaults, or CONFIG with defaults filled in) as YAML."""
    typer.echo(_settings(config, None, None, None).to_yaml(), nl=False)
