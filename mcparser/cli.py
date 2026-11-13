"""Command-line interface."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import anyio
import typer
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from . import __version__
from .agents.roles import ROLES
from .agents.team import AgentTeam, ModelUsage
from .baseline import run_baseline
from .rules import Rules
from .corpus import CorpusError, CorpusStore, SourceMode, format_conllu, parse_conllu
from .evaluate import calibration, calibration_table
from .evaluate import evaluate as evaluate_predictions
from .evaluate import run_official
from .mcp.config import AlignmentSpec, CorpusSpec, SourceSpec, claude_desktop_config, load_store
from .mcp.server import build_server
from .pipeline import Pipeline, RunReport
from .settings import (
    DEFAULT_EFFICIENT_MODEL,
    DEFAULT_EFFICIENT_ROLES,
    DEFAULT_LIGHT_MODEL,
    DEFAULT_LIGHT_ROLES,
    DEFAULT_STRONG_MODEL,
    THINKING_LEVELS,
    Settings,
    Thinking,
    parse_thinking,
)
from .textinput import blank_sentences, build_sentences, read_input

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

TARGET = "pred"


@dataclass
class TierOptions:
    """The model-tier options shared by ``run`` and ``score``."""

    strong_model: str | None = None
    efficient_model: str | None = None
    light_model: str | None = None
    efficient_roles: str | None = None
    light_roles: str | None = None
    strong_thinking: str | None = None
    efficient_thinking: str | None = None
    light_thinking: str | None = None

app = typer.Typer(help="LLM agents for dependency parsing with CoNLL-U files.", no_args_is_help=True)
mcp_app = typer.Typer(help="The CoNLL-U MCP server.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"mcparser {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """LLM agents for dependency parsing with CoNLL-U files."""


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
AlignmentOption = Annotated[
    list[str] | None,
    typer.Option(
        "--alignment",
        "-a",
        help="NAME=SOURCE_A,SOURCE_B,PATH[:ro|rw[:CONFIDENCE]]: word alignments between two sources. Repeatable.",
    ),
]
CorpusHelp = (
    "[NAME=]PATH[,PATH...]: unannotated text (one sentence or paragraph per line) that agents can search for "
    "how a word is used. Repeatable."
)
CorpusOption = Annotated[list[str] | None, typer.Option("--corpus", help=CorpusHelp)]
NameOption = Annotated[str, typer.Option(help="Server name shown to MCP clients.")]


def _specs(sources: list[str]) -> list[SourceSpec]:
    try:
        return [SourceSpec.parse(spec) for spec in sources]
    except CorpusError as error:
        raise typer.BadParameter(str(error), param_hint="--source") from error


def _alignments(alignments: list[str] | None) -> list[AlignmentSpec]:
    try:
        return [AlignmentSpec.parse(spec) for spec in alignments or []]
    except CorpusError as error:
        raise typer.BadParameter(str(error), param_hint="--alignment") from error


def _corpora(corpora: list[str] | None) -> list[CorpusSpec]:
    try:
        return [CorpusSpec.parse(spec) for spec in corpora or []]
    except CorpusError as error:
        raise typer.BadParameter(str(error), param_hint="--corpus") from error


def _load(
    specs: list[SourceSpec],
    root_key: str,
    alignments: list[AlignmentSpec] | None = None,
    corpora: list[CorpusSpec] | None = None,
) -> CorpusStore:
    try:
        return load_store(specs, root_key=root_key, alignments=alignments, corpora=corpora)
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
    alignment: AlignmentOption = None,
    corpus: CorpusOption = None,
) -> None:
    """Serve CoNLL-U sources over MCP (stdio for Claude Desktop, or streamable HTTP)."""
    store = _load(_specs(source), root_key, _alignments(alignment), _corpora(corpus))
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
    alignment: AlignmentOption = None,
    corpus: CorpusOption = None,
) -> None:
    """Print the JSON to add to Claude Desktop's claude_desktop_config.json."""
    specs = _specs(source)
    for spec in specs:
        if not spec.path.is_file():
            typer.echo(f"Warning: {spec.path} does not exist.", err=True)
    config_json = claude_desktop_config(
        specs, name=name, read_only=read_only, root_key=root_key, alignments=_alignments(alignment), corpora=_corpora(corpus)
    )
    typer.echo(json.dumps(config_json, indent=2))


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
ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Settings YAML (see `mcparser settings`).")]


def _prediction_store(prediction: Path, sources: list[str] | None, corpora: list[str] | None = None) -> CorpusStore:
    specs = _specs(sources or [])
    target = SourceSpec(name=TARGET, path=prediction.expanduser().resolve(), mode=SourceMode.READ_WRITE)
    return _load([*specs, target], "Root", corpora=_corpora(corpora))


def _role_list(value: str | None, option: str) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not set(names) <= set(ROLES):
        raise typer.BadParameter(f"Roles must be among {', '.join(ROLES)}.", param_hint=option)
    return names


def _thinking(value: str | None, option: str) -> Thinking | None:
    try:
        return None if value is None else parse_thinking(value)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint=option) from error


def _settings(
    config: Path | None,
    model: str | None,
    model_for: list[str] | None,
    roles: str | None,
    tiers: TierOptions | None = None,
    **overrides: object,
) -> Settings:
    tiers = tiers or TierOptions()
    overrides.update(
        strong_model=tiers.strong_model,
        efficient_model=tiers.efficient_model,
        light_model=tiers.light_model,
        efficient_roles=_role_list(tiers.efficient_roles, "--efficient-roles"),
        light_roles=_role_list(tiers.light_roles, "--light-roles"),
        strong_thinking=_thinking(tiers.strong_thinking, "--strong-thinking"),
        efficient_thinking=_thinking(tiers.efficient_thinking, "--efficient-thinking"),
        light_thinking=_thinking(tiers.light_thinking, "--light-thinking"),
    )
    role_models = {}
    for item in model_for or []:
        role, sep, role_model = item.partition("=")
        if not sep or role not in ROLES:
            raise typer.BadParameter(f"Expected ROLE=MODEL with ROLE one of {', '.join(ROLES)}.", param_hint="--model-for")
        role_models[role] = role_model
    only = _role_list(roles, "--roles")
    try:
        base = Settings.from_file(config) if config else Settings()
        return base.with_overrides(model=model, role_models=role_models, only_roles=only, **overrides)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error


def make_team(store: CorpusStore, target: str, settings: Settings) -> AgentTeam:
    """Build the agent team (replaced in tests by one with scripted models)."""
    return AgentTeam(store, target, settings)


def _print_report(report: RunReport) -> None:
    typer.echo(
        f"Rules settled {report.rules.total} cells without a model: {report.rules.cells}; "
        f"{report.rules.flags} cross-check flags."
    )
    for item in report.rounds:
        typer.echo(
            f"Round {item.round}: scored {item.scored} sentences, {item.tasks} agent tasks, "
            f"{item.edits} edits, {item.open_flags} open flags, mean confidence {item.confidence:.3f}"
        )
    failures = sum(task.error is not None and not task.error.startswith("skipped") for task in report.tasks)
    skipped = sum(task.error is not None and task.error.startswith("skipped") for task in report.tasks)
    rescored = f" ({report.final_scored} sentences re-scored)" if report.final_scored else ""
    typer.echo(f"Final mean confidence {report.confidence:.3f}{rescored}. {len(report.tasks)} tasks ({failures} failed).")
    if report.budget_reached:
        typer.echo(f"STOPPED: the cost limit was reached ({skipped} tasks not started). Run again to continue.")
    typer.echo("By model:")
    for name, usage in [*report.usage.items(), ("total", report.total)]:
        typer.echo(f"  {name}: {_usage_line(usage)}")
    typer.echo("By role:")
    for name, usage in report.roles.items():
        typer.echo(f"  {name}: {_usage_line(usage)}")


def _usage_line(usage: ModelUsage) -> str:
    cost = f", ≈ ${usage.cost:.2f}" if usage.cost is not None else ""
    return (
        f"{usage.tasks} tasks, {usage.requests} requests, {usage.input_tokens:,} input tokens "
        f"({usage.cache_read_tokens:,} cache reads, {usage.cache_write_tokens:,} cache writes), "
        f"{usage.output_tokens:,} output tokens{cost}"
    )


def _annotate(
    prediction: Path,
    source: list[str] | None,
    settings: Settings,
    sentence: list[str] | None,
    score_only: bool,
    report_path: Path | None,
    max_sentences: int | None = None,
    corpus: list[str] | None = None,
) -> None:
    store = _prediction_store(prediction, source, corpus)
    for text_corpus in store.text_corpora.values():
        typer.echo(f"Indexing corpus {text_corpus.name!r} ...", err=True)
        info = text_corpus.info()
        typer.echo(f"  {info.words:,} words, {info.types:,} distinct.", err=True)
    if max_sentences:
        ids = [s.sent_id for s in store.source(TARGET).sentences if not sentence or s.sent_id in sentence]
        sentence = ids[:max_sentences]
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
    log = prediction.with_name(prediction.name + ".runs.jsonl")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(report.model_dump_json() + "\n")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=1), encoding="utf-8")


ModelOption = Annotated[
    str | None, typer.Option(help="One model for every role (overrides --strong-model and --efficient-model).")
]
StrongModelOption = Annotated[
    str | None, typer.Option(help=f"Model for the demanding roles (default {DEFAULT_STRONG_MODEL}).")
]
EfficientModelOption = Annotated[
    str | None, typer.Option(help=f"Cheaper model for the --efficient-roles (default {DEFAULT_EFFICIENT_MODEL}).")
]
EfficientRolesOption = Annotated[
    str | None,
    typer.Option(help=f"Comma-separated roles that use the efficient model (default {','.join(DEFAULT_EFFICIENT_ROLES)})."),
]
StrongThinkingOption = Annotated[
    str | None, typer.Option(help=f"Reasoning for strong roles: off, on, {', '.join(THINKING_LEVELS)} (default: the model's own).")
]
EfficientThinkingOption = Annotated[
    str | None, typer.Option(help="Reasoning for efficient roles (same values as --strong-thinking).")
]
LightModelOption = Annotated[
    str | None, typer.Option(help=f"Cheapest model, for the --light-roles (default {DEFAULT_LIGHT_MODEL}).")
]
LightRolesOption = Annotated[
    str | None, typer.Option(help=f"Comma-separated roles that use the light model (default {','.join(DEFAULT_LIGHT_ROLES)}).")
]
LightThinkingOption = Annotated[
    str | None, typer.Option(help="Reasoning for light roles (same values as --strong-thinking).")
]
BatchOption = Annotated[int | None, typer.Option(min=1, help="Maximum sentences per agent task (default 8).")]
ModelForOption = Annotated[
    list[str] | None, typer.Option("--model-for", help="ROLE=MODEL override, e.g. lemma=anthropic:claude-sonnet-5-5.")
]
SentenceOption = Annotated[list[str] | None, typer.Option("--sentence", help="Only work on this sent_id. Repeatable.")]
MaxSentencesOption = Annotated[
    int | None, typer.Option(min=1, help="Only use the first N sentences (after any --sentence filter).")
]
ReportOption = Annotated[Path | None, typer.Option("--report", help="Write the run report as JSON.")]
PredictionArgument = Annotated[Path, typer.Argument(help="Prediction CoNLL-U file (edited in place).")]


@app.command()
def run(
    prediction: PredictionArgument,
    source: TrustedOption = None,
    corpus: CorpusOption = None,
    config: ConfigOption = None,
    model: ModelOption = None,
    strong_model: StrongModelOption = None,
    efficient_model: EfficientModelOption = None,
    light_model: LightModelOption = None,
    efficient_roles: EfficientRolesOption = None,
    light_roles: LightRolesOption = None,
    strong_thinking: StrongThinkingOption = None,
    efficient_thinking: EfficientThinkingOption = None,
    light_thinking: LightThinkingOption = None,
    batch_sentences: BatchOption = None,
    model_for: ModelForOption = None,
    roles: Annotated[str | None, typer.Option(help=f"Comma-separated roles to run: {', '.join(r for r in ROLES if r != 'scorer')}.")] = None,
    rounds: Annotated[int | None, typer.Option(min=1, help="Maximum rounds.")] = None,
    joint_syntax: Annotated[
        bool | None,
        typer.Option(
            "--joint-syntax/--separate-syntax",
            help="One agent sets HEAD and DEPREL together (default), or separate head and deprel agents.",
        ),
    ] = None,
    final_score: Annotated[
        bool | None,
        typer.Option("--final-score/--no-final-score", help="Re-score the cells changed in the last round (default off)."),
    ] = None,
    max_cost: Annotated[
        float | None,
        typer.Option(min=0.01, help="Stop starting agent tasks once the estimated cost reaches this many dollars (run again to continue)."),
    ] = None,
    corpus_tool: Annotated[
        bool | None,
        typer.Option(
            "--corpus-tool/--no-corpus-tool",
            help="Let agents call corpus_find (default) or give them only the corpus evidence in the prompt.",
        ),
    ] = None,
    threshold: Annotated[float | None, typer.Option(min=0, max=1, help="Refine cells below this confidence.")] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Sentences processed at once.")] = None,
    sentence: SentenceOption = None,
    max_sentences: MaxSentencesOption = None,
    report: ReportOption = None,
) -> None:
    """Score and refine a prediction file with the agents, over several rounds."""
    tiers = TierOptions(
        strong_model=strong_model, efficient_model=efficient_model, light_model=light_model,
        efficient_roles=efficient_roles, light_roles=light_roles,
        strong_thinking=strong_thinking, efficient_thinking=efficient_thinking, light_thinking=light_thinking,
    )  # fmt: skip
    settings = _settings(
        config, model, model_for, roles, tiers,
        rounds=rounds, threshold=threshold, concurrency=concurrency, batch_sentences=batch_sentences,
        final_scoring=final_score, joint_syntax=joint_syntax, corpus_tool=corpus_tool, max_cost=max_cost,
    )  # fmt: skip
    _annotate(prediction, source, settings, sentence, False, report, max_sentences, corpus)


@app.command()
def score(
    prediction: PredictionArgument,
    source: TrustedOption = None,
    corpus: CorpusOption = None,
    config: ConfigOption = None,
    model: ModelOption = None,
    efficient_model: EfficientModelOption = None,
    efficient_thinking: EfficientThinkingOption = None,
    batch_sentences: BatchOption = None,
    model_for: ModelForOption = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Tasks run at once.")] = None,
    sentence: SentenceOption = None,
    max_sentences: MaxSentencesOption = None,
    report: ReportOption = None,
) -> None:
    """Only run the scorer: give every changed or unscored cell a confidence."""
    tiers = TierOptions(efficient_model=efficient_model, efficient_thinking=efficient_thinking)
    settings = _settings(config, model, model_for, None, tiers, concurrency=concurrency, batch_sentences=batch_sentences)
    _annotate(prediction, source, settings, sentence, True, report, max_sentences, corpus)


@app.command()
def init(
    input_path: Annotated[
        Path,
        typer.Argument(
            metavar="INPUT",
            help="Text (one sentence per line), YAML with 'text' fields, or CoNLL-U (annotation is removed, segmentation kept).",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="CoNLL-U file to create.")],
    source: TrustedOption = None,
    baseline: Annotated[bool, typer.Option(help="Fill in what the trusted sources make obvious (no LLM).")] = True,
    force: Annotated[bool, typer.Option("--force", help="Overwrite OUTPUT and its review/history files.")] = False,
    max_sentences: MaxSentencesOption = None,
    skip: Annotated[int, typer.Option(min=0, help="Skip the first N sentences of INPUT (e.g. to start a new batch).")] = 0,
) -> None:
    """Create a prediction file from raw text or a gold CoNLL-U file, then apply the frequency baseline."""
    if output.exists() and not force:
        typer.echo(f"Error: {output} exists; use --force to overwrite it.", err=True)
        raise typer.Exit(1)
    try:
        if input_path.suffix.lower() == ".conllu":
            sentences = blank_sentences(parse_conllu(input_path.read_text(encoding="utf-8")))
        else:
            sentences = build_sentences(read_input(input_path))
    except (CorpusError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    sentences = sentences[skip:]
    sentences = sentences[:max_sentences] if max_sentences else sentences
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_conllu(sentences), encoding="utf-8")
    for sidecar in (".review.json", ".history.jsonl", ".runs.jsonl"):
        output.with_name(output.name + sidecar).unlink(missing_ok=True)
    typer.echo(f"Wrote {len(sentences)} sentences to {output}.")
    if baseline and source:
        store = _prediction_store(output, source)
        result = run_baseline(store, TARGET)
        typer.echo(
            f"Baseline: {result.splits} multiword splits, {result.analysed} words analysed, "
            f"{result.unknown} words not found in the trusted sources."
        )
        rules = Rules(store, TARGET).apply()
        typer.echo(f"Rules settled {rules.total} cells: {rules.cells}.")


@app.command()
def evaluate(
    prediction: Annotated[Path, typer.Argument(help="Predicted CoNLL-U.")],
    gold: Annotated[Path, typer.Argument(help="Gold CoNLL-U.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
    predicted_only: Annotated[
        bool, typer.Option("--predicted-only", help="Ignore gold sentences that are not in the predictions.")
    ] = False,
    annotated_only: Annotated[
        bool, typer.Option("--annotated-only", help="Only score predicted sentences that have been parsed (every HEAD set).")
    ] = False,
    show_calibration: Annotated[
        bool, typer.Option("--calibration", help="Also show accuracy by recorded confidence (Conf in MISC).")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Also score all features and relation subtypes (the official metrics ignore them).")
    ] = False,
    official: Annotated[
        Path | None,
        typer.Option(help="Also run the official conll18_ud_eval.py at this path (sentences it would refuse are repaired first)."),
    ] = None,
) -> None:
    """Compare predictions with gold annotation (the CoNLL 2018 metrics: precision, recall and F1)."""
    try:
        predicted_sentences = parse_conllu(prediction.read_text(encoding="utf-8"))
        gold_sentences = parse_conllu(gold.read_text(encoding="utf-8"))
        result = evaluate_predictions(
            predicted_sentences, gold_sentences, predicted_only=predicted_only, annotated_only=annotated_only, strict=strict
        )
        official_result = run_official(official, predicted_sentences, gold_sentences) if official else None
    except (CorpusError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(result.model_dump_json(indent=1))
        if official_result:
            typer.echo(official_result.model_dump_json(indent=1))
        return
    typer.echo(f"{result.sentences} sentences matched.")
    if result.missing:
        typer.echo(f"Missing from the predictions: {', '.join(result.missing)}")
    if result.extra:
        typer.echo(f"Not in the gold file: {', '.join(result.extra)}")
    if result.invalid:
        typer.echo(f"The official scorer would refuse {len(result.invalid)} predicted sentences (no valid tree): {', '.join(result.invalid[:10])}")
    typer.echo(result.table())
    if official_result:
        typer.echo(f"\nOfficial scorer, {official_result.sentences} sentences ({len(official_result.repaired)} repaired for it, {len(official_result.skipped)} gold sentences it cannot read left out):")
        typer.echo(official_result.table())
    if show_calibration:
        typer.echo("\nAccuracy of cells by confidence:")
        typer.echo(calibration_table(calibration(predicted_sentences, gold_sentences)))


@app.command("settings")
def show_settings(config: ConfigOption = None) -> None:
    """Print the settings (defaults, or CONFIG with defaults filled in) as YAML."""
    typer.echo(_settings(config, None, None, None).to_yaml(), nl=False)
