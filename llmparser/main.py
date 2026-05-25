from pathlib import Path
import typer
import llmloader

from .io import load_file
from .database import load_database, build_database, DEFAULT_EMBEDDING_MODEL_ID
from .parsing import parse_items
from .correction_config import build_checkers, load_correction_config
from .correction import correct_predictions
from .feats_checker import RuleBasedFeatsChecker
from .lemma_checker import RuleBasedLemmaChecker
from .mcp_server import CorrectionSession, CorrectionToolRegistry
from .pos_checker import RuleBasedPosChecker
from .syntax_checker import RuleBasedSyntaxChecker

app = typer.Typer()

@app.command()
def db(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    output:Path = typer.Argument(..., help="Path to output database"),
):
    """ Build database of examples from a CoNLL-U file. """
    build_database(
        conllu_path=input,
        output_path=output,
    )


@app.command()
def parse(
    input:Path = typer.Argument(..., help="Path to input YAML file"),
    db:Path = typer.Option(..., help="Path to database with examples (see `db` command)"),
    output:Path = typer.Option(..., help="Path to output YAML file"),
    n_examples:int = typer.Option(10, help="Number of examples to include in the prompt"),
    model:str = typer.Option("claude-sonnet-4-6", help="LLM model to use"),
    temperature:float = typer.Option(0.2, help="LLM temperature"),
    api_key:str = typer.Option(None, help="API key for LLM"),
    ignore_existing:bool = typer.Option(True, help="Ignore items that already have 'conllu' field in the output"),
):
    """ Parse items in input YAML file using LLM and example database, write output YAML file. """
    llm = llmloader.load(model, temperature=temperature, api_key=api_key)

    # Load database
    database = load_database(db)

    # Read input YAML file
    data = load_file(input)

    parse_items(data, database, llm, n_examples, output=output, ignore_existing=ignore_existing)


@app.command()
def correct(
    input:Path = typer.Argument(..., help="Path to predicted CoNLL-U file"),
    db:Path = typer.Option(..., help="Path to database with training examples (see `db` command)"),
    output:Path = typer.Option(..., help="Path to output YAML file"),
    config:Path|None = typer.Option(None, help="Path to correction YAML config"),
    n_examples:int = typer.Option(10, help="Number of examples to retrieve for correction"),
    lemma_checker:str|None = typer.Option(None, help="Lemma checker to use: 'rule' or omit for no lemma checker"),
    pos_checker:str|None = typer.Option(None, help="POS checker to use: 'rule' or omit for no POS checker"),
    feats_checker:str|None = typer.Option(None, help="FEATS checker to use: 'rule' or omit for no FEATS checker"),
    syntax_checker:str|None = typer.Option(None, help="Syntax checker to use: 'rule' or omit for no syntax checker"),
):
    """Validate and correct CoNLL-U predictions using training examples."""
    database = load_database(db)
    if config is not None:
        token_checkers, sentence_checkers, segmentation_checker = build_checkers(
            load_correction_config(config)
        )
    else:
        token_checkers = {}
        sentence_checkers = {}
        segmentation_checker = None

    if lemma_checker == "rule":
        token_checkers["lemma"] = RuleBasedLemmaChecker()
    elif lemma_checker is not None:
        raise typer.BadParameter("lemma-checker must be 'rule' when provided.")
    if pos_checker == "rule":
        checker = RuleBasedPosChecker()
        token_checkers["upos"] = checker
        token_checkers["xpos"] = checker
    elif pos_checker is not None:
        raise typer.BadParameter("pos-checker must be 'rule' when provided.")
    if feats_checker == "rule":
        token_checkers["feats"] = RuleBasedFeatsChecker()
    elif feats_checker is not None:
        raise typer.BadParameter("feats-checker must be 'rule' when provided.")
    if syntax_checker == "rule":
        checker = RuleBasedSyntaxChecker()
        sentence_checkers["head"] = checker
        sentence_checkers["deprel"] = checker
    elif syntax_checker is not None:
        raise typer.BadParameter("syntax-checker must be 'rule' when provided.")

    correct_predictions(
        prediction_path=input,
        database=database,
        segmentation_checker=segmentation_checker,
        token_checkers=token_checkers,
        sentence_checkers=sentence_checkers,
        n_examples=n_examples,
        output=output,
    )


@app.command()
def serve_mcp(
    input:Path = typer.Argument(..., help="Path to predicted CoNLL-U file"),
    db:Path = typer.Option(..., help="Path to database with training examples (see `db` command)"),
):
    """Start an MCP server exposing correction tools."""
    database = load_database(db)
    session = CorrectionSession.from_conllu_file(input, database)
    registry = CorrectionToolRegistry(session)
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        tool_names = ", ".join(registry.tool_names())
        raise typer.BadParameter(
            f"MCP SDK is not installed. Correction tools are ready to expose: {tool_names}."
        ) from exc

    raise typer.BadParameter("MCP SDK transport wiring is not implemented yet.")


@app.command()
def convert_to_yaml(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    yaml_path:Path = typer.Argument(..., help="Path to output YAML file"),
) -> dict:
    from .io import convert_to_yaml as convert_to_yaml_func
    return convert_to_yaml_func(input, yaml_path)
