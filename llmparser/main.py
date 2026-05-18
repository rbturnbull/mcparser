from pathlib import Path
import typer
import llmloader

from .io import load_file
from .database import load_database, build_database, DEFAULT_EMBEDDING_MODEL_ID
from .parsing import parse_items
from .correction import correct_predictions

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
    n_examples:int = typer.Option(10, help="Number of examples to retrieve for correction"),
):
    """Validate and correct CoNLL-U predictions using training examples."""
    database = load_database(db)
    correct_predictions(
        prediction_path=input,
        database=database,
        n_examples=n_examples,
        output=output,
    )


@app.command()
def convert_to_yaml(
    input:Path = typer.Argument(..., help="Path to input CoNLL-U file"),
    yaml_path:Path = typer.Argument(..., help="Path to output YAML file"),
) -> dict:
    from .io import convert_to_yaml as convert_to_yaml_func
    return convert_to_yaml_func(input, yaml_path)
