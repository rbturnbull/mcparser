"""Command-line interface."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .corpus import CorpusError, CorpusStore
from .mcp.config import SourceSpec, claude_desktop_config, load_store
from .mcp.server import build_server

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
